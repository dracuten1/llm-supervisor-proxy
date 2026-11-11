package proxy

// ─────────────────────────────────────────────────────────────────────────────
// race_coordinator.go — race lifecycle coordination (owner of attempt
// orchestration, coordinator construction, and event fan-out)
// ─────────────────────────────────────────────────────────────────────────────
//
// What this file owns:
//
//   - Coordinator construction ........ newRaceCoordinator,
//                                         newRaceCoordinatorWithEvents
//                                         (raceCoordinator constructors,
//                                         eventBus wiring, snapshot of
//                                         the ConfigSnapshot for the
//                                         race lifetime).
//
//   - Attempt orchestration ........... Start, spawn, manage (the
//                                         select-loop that ticks the
//                                         race forward — spawn-replicas,
//                                         pick-winner, hand-back via
//                                         completion channels),
//                                         handleStreamingDeadline,
//                                         handleHardDeadline,
//                                         cancelAll / cancelAllExcept.
//
//   - Event fan-out .................... publishEvent (the race state
//                                         publishes to the events.Bus
//                                         so observability / metrics
//                                         layers can subscribe without
//                                         touching the race itself).
//
//   - Finalization + introspection ..... execute (the racing proxy's
//                                         single-attempt trampoline
//                                         into race_executor.go),
//                                         GetWinner / WaitForWinner
//                                         (winner hand-back to the
//                                         caller), GetStats,
//                                         GetRequestStatuses,
//                                         GetStreamDeadlineError,
//                                         GetFinalErrorInfo (post-race
//                                         diagnostics), FinalErrorInfo,
//                                         RaceStats types.
//
// Companion file:
//
//   - race_executor.go: per-attempt EXECUTION — the upstream HTTP
//     round-trip for ONE model attempt (header construction, request
//     body hydration via providers.convertRequest, response handling,
//     tool-call repair, reasoning translation, SSE streaming). The
//     coordinator calls into the executor's runOne helper once per
//     spawned attempt and routes the per-attempt result back through
//     the orchestration loop. If you are adding a per-upstream-call
//     behavior, you are in the wrong file — race_coordinator owns the
//     RACE; race_executor owns each ATTEMPT.
//
// Size rationale: this file sits in the 1000-3000 line band because
// the race lifecycle — Case-1 rate-limit classification, credential
// failover, idle-parallel spawning, fallback bookkeeping, and the
// all-failed teardown — is ONE interleaved manage() state machine.
// The lock discipline (c.mu hold/release points threaded through the
// spawn paths) is the load-bearing invariant; splitting by concern
// would scatter those hold/release boundaries across files and make
// the deadlock reasoning much weaker than the line count saves.
// TRIPWIRE: split the event-emission helpers out FIRST if this file
// passes ~2500 lines.
// ─────────────────────────────────────────────────────────────────────────────

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/disillusioners/llm-supervisor-proxy/pkg/credentiallb"
	"github.com/disillusioners/llm-supervisor-proxy/pkg/events"
	"github.com/disillusioners/llm-supervisor-proxy/pkg/models"
	"github.com/disillusioners/llm-supervisor-proxy/pkg/providers"
	"github.com/disillusioners/llm-supervisor-proxy/pkg/stickyorder"
)

// spawnTrigger indicates why parallel requests are being spawned
type spawnTrigger string

const (
	triggerIdleTimeout spawnTrigger = "idle_timeout"
	triggerMainError   spawnTrigger = "main_error"
	// triggerRateLimit (Phase 3 / Round 3b canonical naming) identifies
	// the Case-1 rate-limit trigger that spawns a modelTypeCredFailover
	// row (same model, re-selected credential) instead of a model
	// fallback. triggerMainError continues to identify the plain
	// Case-1 model-fallback trigger.
	triggerRateLimit spawnTrigger = "rate_limit"

	// triggerQuota (layered credential fallback — Fix 1) identifies the
	// QUOTA-class Case-1 trigger: the same same-model reselect-and-retry
	// shape as triggerRateLimit, driven by a quota/credit-exhaustion
	// failure (HTTP 402 / code 2067 / "Token Plan usage limit") instead
	// of a 429. The two classes share the mechanics and differ in the
	// cooldown policy (60s vs the configured quota cooldown) and in this
	// trigger vocabulary — the race_spawn payload and the FE EventLog
	// distinguish the classes by this field.
	triggerQuota spawnTrigger = "quota_exceeded"

	// triggerPairingRetry (upstream pairing-rejection guard) identifies
	// the guard's same-model re-drive: a winner-eligible attempt
	// completed with the anchored upstream pairing-rejection signature
	// text delivered as a 200 completion (pairing_guard.go), was
	// poisoned, and the race re-drives the IDENTICAL rawBody against the
	// SAME model. The re-drive rides the modelTypeCredFailover row shape
	// (same-model machinery, exempt from model-attempt accounting) but
	// carries NO reselected credential — affinity resolution applies, so
	// the resend is the same conversation request, not a credential
	// rotation. Budget: pairingRetryMaxAttempts per race.
	triggerPairingRetry spawnTrigger = "pairing_retry"
)

// triggerForClass maps a credential-failover class onto its spawn
// trigger vocabulary. The zero-value class maps to triggerRateLimit —
// the legacy trigger — because every call site has already narrowed the
// class to an eligible one before consulting this mapping.
func triggerForClass(class credFailoverClass) spawnTrigger {
	if class == credFailoverQuota {
		return triggerQuota
	}
	return triggerRateLimit
}

// spawnTriggerInfo contains detailed information about why a spawn was triggered
type spawnTriggerInfo struct {
	trigger       spawnTrigger
	failedRequest int // Index of the failed request, -1 if not applicable

	// credentialID (Phase 3 / Round 3c C3(1)) is populated ONLY for
	// triggerRateLimit spawns: the reselected credential handed back by
	// engine.ExcludeAndReselect. It rides the trigger info from the
	// Case-1 classification site to spawn()/the executor, which applies
	// it to the new upstreamRequest BEFORE the attempt goroutine starts
	// (race-free by construction — the write happens-before `go execute`).
	// Empty for every other trigger type.
	credentialID string

	// modelID (Phase 3 / Round 3j C1 critical fix) is consumed ONLY
	// by modelTypeCredFailover spawns: the credFailover row MUST target
	// the SAME model that just 429'd (NOT c.models[0] — which is wrong
	// in a 2-model chain where the fallback row carries models[1]'s
	// modelID). It is populated at the Case-1 classification site from
	// latestReq.modelID for the rate-limit path.
	//
	// The modelTypeMain / modelTypeSecond spawn paths do NOT consume
	// this field — they read c.models[0] directly in spawn().
	// triggerMainError-driven modelTypeFallback rows left modelID
	// empty until the layered-credential-fallback fix (Fix 2), which
	// made spawn()'s modelTypeFallback arm a SECOND reader of the
	// field: the Case-1 classification site rides the ADVANCE TARGET
	// (models[i+1]) on it so the chain advances past models[1].
	// Final review A: error-path fallback rows ALWAYS carry the
	// advance target now — chain end schedules NO spawn (terminal).
	// Empty modelID reaches the modelTypeFallback arm only from the
	// idle-timeout path, which keeps the legacy defensive fallback
	// (credFailover → c.models[0], modelTypeFallback → c.models[1]).
	modelID string

	// reason (race-error-reason-classification) is the classified,
	// bounded failure reason for the spawn — populated by the Case-1
	// classification site in manage() from the failed attempt's error
	// + stored HTTP status. Zero-value (empty Class) for main spawns
	// and idle-timeout spawns (W4: no error exists to classify there).
	// Consumed by spawn()'s race_spawn publish as additive
	// reason/reason_class/reason_status payload fields.
	reason failureReason
}

// raceCoordinator manages multiple parallel upstream requests
type raceCoordinator struct {
	mu sync.RWMutex

	baseCtx context.Context
	cfg     *ConfigSnapshot
	req     *http.Request
	rawBody []byte

	// interleaved is the X-Proxy-Interleaved-Thinking flag parsed from the
	// incoming request header (handler.go entry). Threaded down through
	// executeRequest → executeInternalRequest (the sole arm since the
	// remove-external-upstream commission deleted the external arm) so
	// race-internal sites have the flag without needing *requestContext
	// (which never crosses into race_executor). See B3 (architecture §4.1).
	interleaved bool

	// Phase 3 / Round 3c C3 — engine + conversationKey reach the
	// race coordinator through the constructor; both are optional
	// (nil-safe) and consumed by execute() + manage().
	engine          *credentiallb.Engine
	conversationKey string

	// sticky is the sticky last-success order registry (constructor
	// injection; nil-safe). Read-only after construction; the attempt
	// failure funnel uses it to unpin a recorded winner whose own
	// attempt failed. Nil (tests, legacy paths) ⇒ no-op.
	sticky *stickyorder.Engine

	requests    []*upstreamRequest
	models      []string
	winner      *upstreamRequest
	winnerIdx   int
	failedCount int

	done     chan struct{} // Closed when a winner is found and finished, or all failed
	streamCh chan struct{} // Signals when streaming can start

	onceStream sync.Once
	onceDone   sync.Once

	// Metrics for logging/monitoring
	startTime     time.Time
	spawnTriggers []spawnTriggerInfo // Track detailed info about why requests were spawned

	// Phase 3 / Task 18 + R3-5 — credential failover bookkeeping. Lives
	// on the coordinator (NOT requestContext — R3-5 keeps the engine
	// stateless about retries; the coordinator is the single owner of
	// the per-race budget and is constructed fresh per race). Mutated
	// only from manage() under c.mu.
	triedCredIDs map[string]bool // mutated only from manage() under c.mu — race-free by construction

	// failoverAttempts is the GLOBAL rotation count (kept for the
	// model_credential_failover attempt_index payload); the rotation
	// BUDGET itself is per-model since the layered-credential-fallback
	// fix (failoverAttemptsByModel below) — the global counter cannot
	// express "m2 still has a rotation left after m1 spent its two".
	failoverAttempts int // mutated only from manage() under c.mu — race-free by construction

	// Layered credential fallback (Fix 2) — PER-MODEL credential
	// rotation state. The chain advances model-by-model
	// (models[0] → models[1] → …), and EACH model gets its own full
	// Layer-1 treatment, so the rotation budget and the W3
	// soonest-expiry single-shot must be tracked per model — the
	// global failoverAttempts counter (kept for the event
	// attempt_index) cannot express "m2 still has 1 rotation left
	// after m1 spent its 2".
	//
	//   failoverAttemptsByModel[m]      — rotations spent against model
	//                                     m's credential pool this
	//                                     request (budget: len(m's
	//                                     credentials)−1).
	//   soonestExpiryAttemptedByModel[m] — the W3 single-shot flag for
	//                                     m (once spent, m's next
	//                                     eligible failure routes
	//                                     straight to model advance).
	//
	// A NEW model entering the chain needs no explicit reset: both maps
	// read missing keys as zero (no rotations spent / single-shot
	// unspent) — "reset appropriately for the new model's credentials"
	// is the zero-value read, not a state mutation.
	//
	// triedCredIDs stays GLOBAL (credential-ID-keyed): a credential
	// already tried this request is the same account everywhere it
	// appears, so not re-rolling it on a later model is correct, and
	// the R3-5 no-double-try invariant stays one set.
	//
	// Mutated only from manage() under c.mu — race-free by construction.
	failoverAttemptsByModel       map[string]int
	soonestExpiryAttemptedByModel map[string]bool

	// Event publishing
	eventBus  *events.Bus
	requestID string

	// Stream deadline error info (set when stream deadline fires with no content)
	streamDeadlineError *FinalErrorInfo

	// liveFirstByteGate (Phase 2 / real-streaming-default) redefines the
	// winner-eligibility predicate from `IsCompleted()` to "first forwardable
	// data chunk" (`GetError()==nil && GetBuffer().TotalLen()>0`). Set by
	// handler.go BEFORE coordinator construction: liveFirstByteGate =
	// !rc.bufferMode && rc.isStream — isStream-scoped (NOT `!bufferMode`
	// alone, which would change non-stream deadline behavior, M3 amendment).
	// The constructor does NOT compute the gate — isStream is a
	// *requestContext field, not a *ConfigSnapshot, and the constructor
	// signature is ConfigSnapshot-based. Buffered mode (header present or
	// isStream=false) is byte-identical to today.
	liveFirstByteGate bool

	// groupPick (load-balance-groups / Q15) is the request's group
	// rewrite record, set by handler.go right after construction
	// (nil for non-group requests). The spawn path copies it onto
	// the MAIN attempt so the W-1 publish site can fire
	// model_member_selected newlyBound-only.
	groupPick *groupPickInfo

	// Upstream pairing-rejection guard state (pairing_guard.go).
	// Mutated only from manage() under c.mu — race-free by construction.
	//
	//   pairingRetriesSpawned — re-drives spawned this race (cap
	//     pairingRetryMaxAttempts).
	//   pairingRetryOwed — a poisoned attempt whose retry spawn was
	//     deferred because RaceMaxParallel slots were occupied; retried
	//     on the next tick before any new poison is considered.
	//   pairingGuardExhausted — the cap is spent and a completion still
	//     carries the signature; GetFinalErrorInfo appends the
	//     exhaustion note to the client-visible message so the all-failed
	//     path surfaces a REAL upstream error (never the signature text).
	pairingRetriesSpawned int
	pairingRetryOwed      *upstreamRequest
	pairingGuardExhausted bool
}

func newRaceCoordinator(ctx context.Context, cfg *ConfigSnapshot, req *http.Request, rawBody []byte, models []string, interleaved bool) *raceCoordinator {
	return newRaceCoordinatorWithEvents(ctx, cfg, req, rawBody, models, nil, "", interleaved, nil, "", false, nil)
}

// newRaceCoordinatorWithEvents (Phase 3 / Round 3c C3) gains two more
// parameters: engine (the Load-Balancing engine; nil-safe) and
// conversationKey (the per-request affinity key). Constructor-only
// injection per W-3 holds end-to-end (cmd/main.go → NewHandler →
// coordinator). The 7th positional arg in NewHandler is the source.
//
// Phase 2 / real-streaming-default adds an 11th positional parameter:
// liveFirstByteGate. When true, the manage-loop winner-eligibility
// predicate swaps from `IsCompleted()` to the first-byte gate
// (GetError()==nil && GetBuffer().TotalLen()>0). The 6-arg wrapper
// (`newRaceCoordinator`) passes false, so direct-constructor test
// callers stay byte-identical to buffered/legacy semantics. The
// production call site is handler.go, which computes the flag
// as `!rc.bufferMode && rc.isStream`.
//
// Sticky last-success ordering adds the 12th positional parameter:
// sticky (the stickyorder.Engine; nil-safe — nil means the feature is
// absent, so the 6-arg wrapper and any direct test construction keep
// byte-identical semantics). The production call site passes
// h.stickyEngine.
func newRaceCoordinatorWithEvents(
	ctx context.Context,
	cfg *ConfigSnapshot,
	req *http.Request,
	rawBody []byte,
	models []string,
	eventBus *events.Bus,
	requestID string,
	interleaved bool,
	engine *credentiallb.Engine,
	conversationKey string,
	liveFirstByteGate bool,
	sticky *stickyorder.Engine,
) *raceCoordinator {
	if len(models) == 0 {
		models = []string{cfg.ModelID}
	}
	return &raceCoordinator{
		baseCtx:           ctx,
		cfg:               cfg,
		req:               req,
		rawBody:           rawBody,
		interleaved:       interleaved,
		engine:            engine,
		conversationKey:   conversationKey,
		sticky:            sticky,
		models:            models,
		requests:          make([]*upstreamRequest, 0, len(models)),
		winnerIdx:         -1,
		done:              make(chan struct{}),
		streamCh:          make(chan struct{}),
		startTime:         time.Now(),
		spawnTriggers:     make([]spawnTriggerInfo, 0),
		liveFirstByteGate: liveFirstByteGate,
		// Phase 3 / Task 19 — per-request tried-set (R3-5: request
		// context, NOT engine state). Initialized here so manage()'s
		// first mutation never sees a nil map.
		triedCredIDs: make(map[string]bool),
		// Layered credential fallback (Fix 2) — per-model rotation
		// state maps (see the field docs above).
		failoverAttemptsByModel:       make(map[string]int),
		soonestExpiryAttemptedByModel: make(map[string]bool),
		eventBus:                      eventBus,
		requestID:                     requestID,
	}
}

// publishEvent publishes an event to the event bus if available
func (c *raceCoordinator) publishEvent(eventType string, data map[string]interface{}) {
	if c.eventBus == nil {
		return
	}
	// Always include request ID for correlation
	if c.requestID != "" {
		data["id"] = c.requestID
	}
	c.eventBus.Publish(events.Event{
		Type:      eventType,
		Timestamp: time.Now().Unix(),
		Data:      data,
	})
}

// Start initiates the race
func (c *raceCoordinator) Start() {
	log.Printf("[RACE] Starting race coordinator with %d models: %v", len(c.models), c.models)

	// Publish race_started event
	c.publishEvent("race_started", map[string]interface{}{
		"models": c.models,
	})

	// 1. Spawn main request (no trigger - it's the initial request)
	c.spawn(modelTypeMain, spawnTriggerInfo{
		trigger:       "",
		failedRequest: -1,
	})

	// 2. Start manager loop
	go c.manage()
}

func (c *raceCoordinator) spawn(mType upstreamModelType, triggerInfo spawnTriggerInfo) {
	c.mu.Lock()
	defer c.mu.Unlock()

	idx := len(c.requests)

	// Assign model based on request type
	var modelID string
	switch mType {
	case modelTypeMain:
		if idx >= len(c.models) {
			log.Printf("[RACE] Cannot spawn main: only %d model(s) available", len(c.models))
			return
		}
		modelID = c.models[0]
	case modelTypeSecond:
		if idx >= len(c.models) {
			log.Printf("[RACE] Cannot spawn second: only %d model(s) available", len(c.models))
			return
		}
		modelID = c.models[0]
	case modelTypeFallback:
		if len(c.models) < 2 {
			log.Printf("[RACE] Cannot spawn fallback: only %d model(s) available", len(c.models))
			return
		}
		// Layered credential fallback (Fix 2) — the fallback row targets
		// the NEXT model in the chain (models[i+1] for the model that
		// just exhausted its Layer-1 credential rotation), not the
		// hard-coded c.models[1]. The advance target is computed at the
		// Case-1 classification site (fallbackAdvanceTargetLocked) and
		// rides triggerInfo.modelID — the same mechanism as the C1 fix
		// for credFailover rows (spawn never re-derives the model).
		// Final review A: error-path rows ALWAYS carry the advance
		// target — the Case-1 dispatch skips the spawn entirely when
		// none exists (chain end ⇒ terminal, no re-spawn). The legacy
		// empty ⇒ c.models[1] mapping below now serves only the
		// idle-timeout path, whose triggerInfo legitimately carries no
		// modelID (pre-Fix-2 behavior preserved verbatim there).
		if triggerInfo.modelID != "" {
			modelID = triggerInfo.modelID
		} else {
			modelID = c.models[1]
		}
	case modelTypeCredFailover:
		// Phase 3 / Round 3j C1 critical fix — the credFailover row MUST
		// target the SAME model that just 429'd, NOT c.models[0]. The
		// prior code's hard-coded c.models[0] was wrong in a 2-model
		// chain: a fallback-row 429 would have spawned a credFailover
		// attempt running models[1]'s InternalModel on models[0]'s
		// credential pool (wrong model served + wrong-account billing).
		//
		// The trigger info carries the source modelID (C3(2) wiring,
		// populated at the Case-1 classification site from
		// latestReq.modelID). If unset (defensive — should never happen
		// for a real triggerRateLimit spawn), fall back to c.models[0]
		// to preserve the legacy single-model chain.
		//
		// NOTE (B1 separate accounting): this row does NOT consume a
		// spawn-window slot — the idx guard below main/second/fallback
		// does NOT apply here (a 3-cred single-model chain spawns idx
		// 1 and 2 past len(c.models), by design).
		if len(c.models) == 0 {
			log.Printf("[RACE] Cannot spawn cred-failover: no model available")
			return
		}
		if triggerInfo.modelID != "" {
			modelID = triggerInfo.modelID
		} else {
			modelID = c.models[0]
		}
	}

	req := newUpstreamRequest(idx, mType, modelID, c.cfg.RaceMaxBufferBytes)

	// Set flag to use secondary upstream model for second requests
	if mType == modelTypeSecond {
		req.SetUseSecondaryUpstream(true)
	}

	// load-balance-groups / Q15: the group pick rides the MAIN (and
	// second) attempt — models[0] is the sticky-picked member after
	// the Am21 rewrite. Cred-failover and fallback attempts of the
	// member's own chain do not carry it (the member event fires
	// once, on the member's own resolution).
	if (mType == modelTypeMain || mType == modelTypeSecond) && c.groupPick != nil {
		req.groupPick = c.groupPick
	}

	// Phase 3 / Round 3c C3(1) — a credFailover row carries the
	// pre-resolved credential (engine.ExcludeAndReselect output) on the
	// trigger info. Applied BEFORE `go c.execute(req)` below, so the
	// executor's resolveSpecificInternalCredential read is race-free
	// (goroutine start is the happens-before edge).
	if mType == modelTypeCredFailover && triggerInfo.credentialID != "" {
		req.reselectedCredentialID = triggerInfo.credentialID
	}

	c.requests = append(c.requests, req)
	c.spawnTriggers = append(c.spawnTriggers, triggerInfo)

	log.Printf("[RACE] Spawning %s request (id=%d, model=%s, trigger=%s)", mType, idx, modelID, triggerInfo.trigger)

	// Build event data with detailed error info if available
	eventData := map[string]interface{}{
		"request_index": idx,
		"model":         modelID,
		"type":          string(mType),
		"trigger":       string(triggerInfo.trigger),
	}

	// Add detailed error information if this spawn was triggered by an error
	if triggerInfo.trigger == triggerMainError {
		// trigger_reason keeps its key (and its role: the failed
		// attempt's error text) but is now the BOUNDED, redacted
		// detail — an upstream body can no longer flood the payload.
		eventData["trigger_reason"] = triggerInfo.reason.Detail
		eventData["failed_request_index"] = triggerInfo.failedRequest
	}
	// Phase 3 / Task 18 — surface the reselected credential on
	// credFailover spawns (operator observability via the SSE feed).
	// Layered credential fallback Fix 1: the quota class spawns the
	// same modelTypeCredFailover rows, so quota_exceeded spawns carry
	// the identical credential_id + failed_request_index payload shape
	// as rate_limit spawns.
	if (triggerInfo.trigger == triggerRateLimit || triggerInfo.trigger == triggerQuota) && triggerInfo.credentialID != "" {
		eventData["credential_id"] = triggerInfo.credentialID
		eventData["failed_request_index"] = triggerInfo.failedRequest
	}

	// race-error-reason-classification — ADDITIVE enrichment: every
	// trigger that carries a classification publishes the bounded,
	// greppable reason (reason=… status=… detail="…") plus the
	// structured class/status fields. Event name, existing keys, and
	// key semantics are unchanged; idle_timeout spawns (no error to
	// classify) publish nothing extra, byte-identical to before.
	// appendReasonFields fixes the additive key sequence (reason →
	// reason_class → conditional reason_status) in one place.
	if triggerInfo.trigger != "" && triggerInfo.reason.Class != "" {
		appendReasonFields(eventData, triggerInfo.reason)
	}

	// Publish race_spawn event
	c.publishEvent("race_spawn", eventData)

	// Execute in background
	go c.execute(req)
}

func (c *raceCoordinator) manage() {
	// HEARTBEAT / MONITORING LOOP
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	// STREAMING DEADLINE TIMER
	// When StreamDeadline is reached, pick the best buffer and continue streaming.
	// This allows us to start streaming to the client even if the upstream hasn't finished.
	//
	// PRECEDENCE (Phase 1 + Phase 2 / real-streaming-default):
	//   - Buffered mode (X-LLMProxy-Buffer-Response header present OR isStream=false):
	//     coordinator runs with the IsCompleted winner gate; StreamDeadline behaves as
	//     today — promotes a 0-byte best winner and keeps streaming until the hard
	//     deadline or upstream completion.
	//   - Live mode (header absent AND isStream=true): coordinator runs with the
	//     first-byte winner gate (liveFirstByteGate = true). StreamDeadline here
	//     MEANS "no forwardable byte within StreamDeadline ⇒ terminal in-band SSE
	//     error envelope + cancelAll()" (see handleStreamingDeadline 0-byte guard).
	streamDeadlineTimer := time.NewTimer(c.cfg.StreamDeadline)
	defer streamDeadlineTimer.Stop()

	// HARD DEADLINE TIMER
	// When MaxGenerationTime is reached, forcefully terminate all requests.
	// This is the absolute hard timeout for the entire request lifecycle.
	hardDeadlineTimer := time.NewTimer(c.cfg.MaxGenerationTime)
	defer hardDeadlineTimer.Stop()

	// Track when we started monitoring for idle
	idleCheckStart := time.Now()

	for {
		select {
		case <-c.baseCtx.Done():
			c.cancelAll()
			c.onceDone.Do(func() { close(c.done) })
			c.onceStream.Do(func() { close(c.streamCh) })
			return
		case <-streamDeadlineTimer.C:
			// Streaming deadline reached - pick best buffer and continue streaming
			c.handleStreamingDeadline()
			return
		case <-hardDeadlineTimer.C:
			// Hard deadline reached - force end everything
			log.Printf("[RACE] Hard deadline reached after %v, forcing end", time.Since(c.startTime))
			c.handleHardDeadline()
			return
		case <-ticker.C:
			c.mu.Lock()

			// Check for winner eligibility
			// IMPORTANT: Winner is only selected when request is COMPLETED (received [DONE] signal)
			// BUG FIX: Previously selected winner when IsStreaming() was true, which caused
			// premature winner selection when request finishes within idle timeout.
			// The correct behavior: buffer > wait till DONE > select winner
			//
			// Phase 2 / real-streaming-default — live-mode winner eligibility
			// is REDEFINED to "first forwardable data chunk" (the attempt's
			// streamBuffer transitions from 0 to >=1 chunk):
			// `req.GetError()==nil && req.GetBuffer().TotalLen()>0`.
			// Buffered mode (and isStream=false requests, M3) keep the
			// original IsCompleted gate verbatim — byte-identical to today.
			//
			// CONSTRAINT (pinned gate form, amendment #8): the swap lands
			// in the EXISTING manage-loop ticker arm with `onceStream`
			// single-close semantics (:431), the deadline guard's reuse of
			// the winner-nil check (:653-655), and the cancel-and-exit
			// (:440-446) untouched. NO per-attempt atomic first-byte flag.
			// Upstream pairing-rejection guard — set by the re-drive block
			// below when THIS tick spawned a same-model pairing re-drive;
			// suppresses the Case-1 fallback spawn and the all-failed
			// terminal check for the tick (same defer pattern as
			// spawnedCredFailover).
			spawnedPairingRetry := false

			if c.winner == nil {
				// pairing-rejection guard (pairing_guard.go) — attempts
				// poisoned by THIS tick's scan. One same-model re-drive
				// spawns per tick (see the block after the scan).
				var pairingPoisoned []*upstreamRequest

				for i, req := range c.requests {
					// Phase 2 / real-streaming-default — STRICT mode switch
					// (review fix; plan :572-574, :640, CONSTRAINTS :609):
					// the live-mode gate REPLACES the IsCompleted gate, it
					// is not an OR-superset over it. The OR form had a
					// corner defect: a live-mode attempt that completed
					// with err==nil but TotalLen()==0 (upstream emitted
					// only [DONE]/comments) satisfied the IsCompleted arm
					// and WON — a phantom 0-byte 200 stream that preempts
					// fallback and sidesteps the task-2.2 0-byte guard.
					// In live mode ONLY `TotalLen()>0 && err==nil` is
					// eligible; buffered mode (and isStream=false, M3)
					// keeps the IsCompleted gate verbatim.
					var eligible bool
					if c.liveFirstByteGate {
						eligible = req.GetError() == nil && req.GetBuffer().TotalLen() > 0
					} else {
						eligible = req.IsCompleted() && req.GetError() == nil
					}
					if eligible {
						// Upstream pairing-rejection guard — adjudicate the
						// attempt buffer BEFORE promotion. The verdict is
						// anchored-prefix exact over the accumulated
						// assistant text (pairing_guard.go):
						//
						//   match   → poison the attempt (flip to failed
						//             with the guard sentinel; cancel the
						//             in-flight upstream read) and never
						//             promote — the signature text must
						//             not reach the client as a 200.
						//   pending → still-streaming content that is a
						//             proper prefix of a signature: hold
						//             promotion for this tick; the next
						//             tick sees divergence (→ promotable),
						//             the full match (→ poisoned), or
						//             completion (→ final adjudication).
						//   none    → normal winner candidate.
						switch pairingGuardVerdictForBuffer(req.GetBuffer()) {
						case pairingGuardMatch:
							if req.MarkPairingRejected(errUpstreamPairingRejection) {
								// Mirrors execute()'s failure emission:
								// same event keys + additive classified
								// reason (upstream_pairing_rejection), so
								// the EventLog treats the poisoned attempt
								// as the upstream failure it is.
								failReason := classifyRequestFailure(req)
								log.Printf("[RACE] Request %d POISONED by pairing-rejection guard: %s", req.id, failReason)
								c.publishEvent("race_request_failed", map[string]interface{}{
									"request_index": req.id,
									"model":         req.modelID,
									"type":          string(req.modelType),
									"error":         failReason.Detail,
									"guard":         "pairing_rejection",
								})
								pairingPoisoned = append(pairingPoisoned, req)
							}
							continue
						case pairingGuardPending:
							continue
						}

						// We found a potential winner!
						// Preference: earlier requests (lower index)
						if c.winner == nil || i < c.winnerIdx {
							c.winner = req
							c.winnerIdx = i

							// Enhanced logging with timing and buffer stats
							elapsed := time.Since(c.startTime)
							bufferLen := req.GetBuffer().TotalLen()
							log.Printf("[RACE] Winner selected: request %d (%s, %s) after %v, buffer=%d bytes",
								i, req.modelType, req.modelID, elapsed.Round(time.Millisecond), bufferLen)

							// Publish race_winner_selected event
							c.publishEvent("race_winner_selected", map[string]interface{}{
								"winner_index": c.winnerIdx,
								"winner_type":  string(c.winner.modelType),
								"winner_model": c.winner.modelID,
								"duration_ms":  elapsed.Milliseconds(),
								"buffer_bytes": bufferLen,
							})

							c.onceStream.Do(func() { close(c.streamCh) })
						}
					}
				}

				// Upstream pairing-rejection guard — RE-DRIVE (transparent
				// retry) / EXHAUSTION. Runs only when NO winner emerged and
				// this scan poisoned at least one attempt (or a previous
				// tick's re-drive is still owed a free parallel slot).
				//
				// The re-drive re-sends the coordinator's rawBody VERBATIM
				// (never mutated — the upstream mints the tool-call ids) to
				// the SAME model, as a modelTypeCredFailover-shaped row
				// (exempt from model-attempt accounting, does not consume
				// the model chain). Precedence: the pairing re-drive beats
				// the Case-1 credential/model fallback for a poisoned
				// attempt — the upstream answered 200 and the request is
				// healable by replay; when the budget is spent, the normal
				// failure machinery (Case-1 chain advance / all-failed
				// terminal) takes over untouched.
				if c.winner == nil {
					retryReq := c.pairingRetryOwed
					if retryReq == nil && len(pairingPoisoned) > 0 {
						retryReq = pairingPoisoned[len(pairingPoisoned)-1]
					}
					if retryReq != nil {
						running := 0
						for _, r := range c.requests {
							if !r.IsDone() {
								running++
							}
						}
						if running >= c.cfg.RaceMaxParallel {
							// No parallel slot free — owe the re-drive to
							// the next tick (the poisoned attempt stays
							// failed; nothing is relayed meanwhile).
							c.pairingRetryOwed = retryReq
						} else {
							c.pairingRetryOwed = nil
							if c.pairingRetriesSpawned < pairingRetryMaxAttempts {
								c.pairingRetriesSpawned++
								spawnedPairingRetry = true
								failReason := classifyRequestFailure(retryReq)
								log.Printf("[RACE] Pairing-rejection guard: re-driving identical rawBody against %s (retry %d/%d, failed attempt %d)",
									retryReq.modelID, c.pairingRetriesSpawned, pairingRetryMaxAttempts, retryReq.id)
								c.mu.Unlock()
								c.spawn(modelTypeCredFailover, spawnTriggerInfo{
									trigger:       triggerPairingRetry,
									failedRequest: retryReq.id,
									modelID:       retryReq.modelID,
									reason:        failReason,
								})
								c.mu.Lock()
							} else {
								// EXHAUSTION: cap spent and a completion
								// still carries the signature. No re-drive —
								// fall through to the normal terminal
								// machinery (all-failed → GetFinalErrorInfo
								// → real 502 upstream_error to the client;
								// the sentinel message + this note replace
								// the signature text).
								if !c.pairingGuardExhausted {
									c.pairingGuardExhausted = true
									log.Printf("[RACE] Pairing-rejection guard exhausted after %d re-drive(s); surfacing upstream error", pairingRetryMaxAttempts)
								}
							}
						}
					}
				}
			}

			// If we have a winner, stop all other attempts and exit management loop.
			// The winning attempt will continue to stream into its buffer,
			// and the handler will read from that buffer.
			if c.winner != nil {
				c.mu.Unlock()
				c.cancelAllExcept(c.winner)
				// We don't close c.done here yet, but we can exit the manager loop.
				// c.done will be closed when the context is cancelled.
				return
			}
			// Spawning logic (on failure or idle)
			//
			// Phase 3 / Round 3c — C1 HOISTED ADMISSION GATE (replaces the
			// pre-Phase-3 window check `len(c.requests) < len(c.models)`):
			//    gate := modelAttempts < len(models) || credFailoverEligibleWithBudget()
			// modelAttempts counts ONLY modelType ∈ {main, second, fallback}
			// (B1 separate accounting — modelTypeCredFailover rows are EXEMPT
			// from the window count and from the all-failed accounting below).
			// The hoisted form is what lets the core scenario (3 creds /
			// 1 model / no fallback chain: `1 < 1` false from the first
			// failure) get its FULL failover chain before terminal.
			modelAttempts := c.modelAttemptCountLocked()
			credEligible := c.credFailoverEligibleLocked()
			gate := modelAttempts < len(c.models) || credEligible
			spawnedCredFailover := false
			// Pairing-rejection guard precedence: while this tick spawned
			// a pairing re-drive (or one is owed a free parallel slot),
			// the Case-1 credential/model fallback spawn is suppressed —
			// the poisoned attempt is being healed by replay, and letting
			// Case-1 fire in the same tick would double-spawn against it.
			// When the guard is exhausted, pairingRetryOwed is nil and
			// Case-1 proceeds untouched (chain advance / terminal).
			guardOwnsSpawn := spawnedPairingRetry || c.pairingRetryOwed != nil
			if c.winner == nil && gate && !guardOwnsSpawn {
				running := 0
				for _, r := range c.requests {
					if !r.IsDone() {
						running++
					}
				}

				shouldSpawn := false
				var triggerInfo spawnTriggerInfo

				if running < c.cfg.RaceMaxParallel {
					// Case 1: Latest request failed - spawn fallback directly (skip retry with same model)
					latestReq := c.requests[len(c.requests)-1]
					if latestReq.IsDone() && latestReq.GetError() != nil {

						// race-error-reason-classification — classify ONCE
						// per failed attempt (bounded + redacted detail),
						// then reuse across the credFailover branch, the
						// model-fallback branch, and the log line. The raw
						// error string never reaches an event payload or a
						// [RACE] log line unbounded again.
						failReason := classifyRequestFailure(latestReq)

						// ── HOISTED CRED-FAILOVER BRANCH (Round 3c C1;
						// Tasks 18-20; layered credential fallback Fix 1) ──
						// rate-limit OR quota-class failure AND internal
						// multi-credential model AND pre-first-byte (race
						// guard: c.winner == nil per Task 23 — framing
						// headers + ": connected" carry no model content)
						// AND per-model retry budget remaining ⇒
						// ExcludeAndReselect + same-model
						// modelTypeCredFailover spawn INSTEAD of model
						// fallback. Precedence: credential failover
						// BEFORE model switching — per model, not just
						// for models[0] (Fix 2).
						plan := c.credFailoverPlanLocked(latestReq)
						if plan != nil {
							c.mu.Unlock()
							c.publishCredentialFailover(latestReq.modelID, plan.fromCredentialID, plan.credentialID, plan)
							c.spawn(modelTypeCredFailover, spawnTriggerInfo{
								trigger:       triggerForClass(plan.class),
								failedRequest: latestReq.id,
								credentialID:  plan.credentialID,
								// Phase 3 / Round 3j C1 critical fix —
								// ride the source modelID on the trigger
								// info so spawn() targets the SAME model
								// that failed (not c.models[0]).
								modelID: latestReq.modelID,
								// race-error-reason-classification — the
								// classified reason rides the trigger
								// info into the race_spawn payload.
								reason: failReason,
							})
							c.mu.Lock()
							// Budget bookkeeping (Task 19 / R3-5 — mutated only
							// from manage() under c.mu — race-free by construction).
							c.failoverAttempts++
							c.triedCredIDs[plan.credentialID] = true
							c.failoverAttemptsByModel[latestReq.modelID]++
							if plan.soonestExpiry {
								// W3 single-shot: the SoonestExpiry attempt is
								// spent for THIS model; the NEXT eligible
								// failure of the same model routes straight to
								// model advance (no double spawn, no loop).
								// Fix 2: per-model flag — a later model in the
								// chain starts with its own unspent single-shot.
								c.soonestExpiryAttemptedByModel[latestReq.modelID] = true
							}
							spawnedCredFailover = true
							// Skip the model-fallback spawn this tick — the
							// credFailover row IS this tick's spawn.
						} else {
							// Bounded classified log line — the raw errMsg
							// (which can embed a whole upstream error body
							// on the context-overflow path) stays out of
							// the server log.
							// Layered credential fallback (Fix 2) — the
							// chain advances to the NEXT model after the
							// failed one, so every model in the chain gets
							// its Layer-1 credential rotation.
							// Final review A: hasAdvance distinguishes a
							// real advance target from chain-end /
							// not-in-chain. NO advance target ⇒ genuinely
							// TERMINAL — no spawn is scheduled (the
							// legacy empty ⇒ c.models[1] mapping would
							// re-spawn an already-exhausted model, a
							// wrong-model retry in the C1 family); the
							// race resolves as failed for this path.
							advanceTarget, hasAdvance := c.fallbackAdvanceTargetLocked(latestReq.modelID)
							if hasAdvance {
								log.Printf("[RACE] Latest request %d failed: %s, spawning fallback directly", latestReq.id, failReason)
								shouldSpawn = true
								triggerInfo = spawnTriggerInfo{
									trigger:       triggerMainError,
									failedRequest: latestReq.id,
									reason:        failReason,
									modelID:       advanceTarget,
								}
							} else {
								log.Printf("[RACE] Latest request %d failed: %s; no advance target — chain end, terminal", latestReq.id, failReason)
							}
						}
					}

					// Case 2: Main request is idle (Parallel race retry)
					// Phase 3 / Round 3c W4 (RE-SCOPE of Round-3b ruling iii):
					// this path spawns second/fallback UNCHANGED — at
					// idle-spawn time NO error exists to classify
					// (latestReq.GetError() is nil for a stalled-but-
					// unfinished request). The stalled attempt's eventual
					// 429 is adjudicated by Case 1 above, which the C1
					// hoisted gate makes reachable in the multi-attempt
					// environment this spawn creates.
					// FIXED: Now uses IsIdle() which tracks last activity time during streaming
					// This correctly detects idle even after the request has started streaming
					if !shouldSpawn && !spawnedCredFailover && c.cfg.RaceParallelOnIdle && len(c.requests) == 1 {
						mainReq := c.requests[0]
						// Check for idle in two ways:
						// 1. statusRunning: hasn't received first byte yet (use start time)
						// 2. statusStreaming: has received data but is now idle (use last activity time)
						if mainReq.GetStatus() == statusRunning {
							// Haven't received first byte yet - check from start time
							if time.Since(idleCheckStart) > time.Duration(c.cfg.IdleTimeout) {
								log.Printf("[RACE] Main request idle (no first byte), spawning parallel request")
								shouldSpawn = true
								triggerInfo = spawnTriggerInfo{
									trigger:       triggerIdleTimeout,
									failedRequest: -1,
								}
							}
						} else if mainReq.IsIdle(time.Duration(c.cfg.IdleTimeout)) {
							// Has received data but is now idle
							log.Printf("[RACE] Main request idle (no data for %v), spawning parallel request", time.Duration(c.cfg.IdleTimeout))
							shouldSpawn = true
							triggerInfo = spawnTriggerInfo{
								trigger:       triggerIdleTimeout,
								failedRequest: -1,
							}
						}
					}
				}

				if spawnedCredFailover {
					// credFailover row already spawned this tick — skip the
					// model-fallback spawn path below.
				} else if shouldSpawn {

					c.mu.Unlock()

					if triggerInfo.trigger == triggerIdleTimeout {
						// On idle timeout: spawn both second AND fallback together
						c.spawn(modelTypeSecond, triggerInfo)
						if len(c.models) > 1 {
							c.spawn(modelTypeFallback, triggerInfo)
						}
					} else {
						// On error: spawn fallback directly (skip retry with same model)
						if len(c.models) > 1 {
							c.spawn(modelTypeFallback, triggerInfo)
						} else {
							log.Printf("[RACE] Main failed but no fallback available")
						}
					}

					c.mu.Lock()
				}
			}

			// If no winner and reached max parallel attempts, check if all failed
			//
			// Phase 3 / B1 amendment: the accounting counts MODEL attempts only
			// (modelTypeCredFailover rows are exempt). Terminal fires when every
			// request has failed AND no credFailover row was just spawned (that
			// row is still running); after credential exhaustion the flow falls
			// through to the model-fallback spawn above when len(c.models) > 1
			// — the spawn advances to the next model in the chain (see
			// fallbackAdvanceTargetLocked). Final review A: an error-path
			// tick with NO advance target (failed model is the chain's
			// last) is TERMINAL — no spawn is scheduled and the
			// all-failed resolution below fires; the legacy c.models[1]
			// mapping survives only on the idle-timeout path.
			//
			// Pairing-rejection guard: a tick that just spawned a pairing
			// re-drive also defers the terminal check (the re-drive row is
			// still running — same reasoning as spawnedCredFailover).
			if c.winner == nil && !spawnedCredFailover && !spawnedPairingRetry && modelAttempts >= len(c.models) {
				allFailed := true
				for _, r := range c.requests {
					if !r.IsDone() || r.GetError() == nil {
						allFailed = false
						break
					}
				}
				if allFailed {
					log.Printf("[RACE] All requests failed")

					// Collect all error details. Per-row `error` is the
					// bounded/redacted text; `reason` is the additive
					// classified form (race-error-reason-classification).
					errors := make([]map[string]interface{}, 0, len(c.requests))
					for _, r := range c.requests {
						if r.GetError() != nil {
							rReason := classifyRequestFailure(r)
							row := map[string]interface{}{
								"request_index": r.id,
								"model":         r.modelID,
								"type":          string(r.modelType),
								"error":         rReason.Detail,
							}
							appendReasonFields(row, rReason)
							errors = append(errors, row)
						}
					}

					// Publish race_all_failed event with detailed error info
					c.publishEvent("race_all_failed", map[string]interface{}{
						"total_attempts": len(c.requests),
						"duration_ms":    time.Since(c.startTime).Milliseconds(),
						"errors":         errors,
					})

					c.mu.Unlock()
					c.onceDone.Do(func() { close(c.done) })
					c.onceStream.Do(func() { close(c.streamCh) })
					return
				}
			}

			c.mu.Unlock()
		}
	}
}

// handleStreamingDeadline picks the best buffer when StreamDeadline is reached
// Per the unified race retry design:
// - Pick the request with the most content (best candidate to continue)
// - DON'T cancel the winner - let it continue streaming until complete or hard deadline
// - Cancel only the other requests
//
// Phase 2 / real-streaming-default — live-mode 0-byte guard (M3 + amendment #8).
// When liveFirstByteGate is true, the predicate inside the manage-loop ticker
// arm is the first-byte gate, so a deadline fire with all buffers empty MUST
// surface as an in-band SSE error envelope + all attempts cancelled (today's
// behavior promotes a 0-byte winner and keeps streaming — wrong for live mode).
// Buffered mode keeps today's behavior byte-identical.
func (c *raceCoordinator) handleStreamingDeadline() {
	c.mu.Lock()

	if c.winner != nil {
		c.mu.Unlock()
		return // Already have a winner
	}

	log.Printf("[RACE] Streaming deadline reached after %v, picking best buffer", time.Since(c.startTime))

	// Find request with most content (best candidate to continue)
	// If all buffers are equal, prefer the first (main) request
	//
	// Upstream pairing-rejection guard: match/pending candidates are NOT
	// deadline-pickable. A match candidate is poisoned-bound (delivering
	// it would relay the rejection text); a pending candidate is not yet
	// adjudicable, and manage() returns after this method — so a held
	// candidate can never be re-adjudicated later. Neither may win the
	// deadline pick; if every candidate is guard-bound, the no-content
	// error envelope below fires (a REAL upstream error beats a
	// maybe-signature 200).
	var best *upstreamRequest
	var bestLen int64 = -1 // Start at -1 so first request always gets selected

	for _, req := range c.requests {
		if req != nil && !req.IsDone() {
			if v := pairingGuardVerdictForBuffer(req.GetBuffer()); v != pairingGuardNone {
				if v == pairingGuardMatch {
					// Poison now — manage() exits after this method, so
					// this is the last chance to mark the attempt failed.
					req.MarkPairingRejected(errUpstreamPairingRejection)
					log.Printf("[RACE] Deadline pick skipped request %d: pairing-rejection signature (poisoned)", req.id)
				} else {
					log.Printf("[RACE] Deadline pick skipped request %d: guard verdict pending", req.id)
				}
				continue
			}
			bufferLen := req.GetBuffer().TotalLen()
			if bufferLen > bestLen {
				best = req
				bestLen = bufferLen
			}
		}
	}

	// If we have a winner (even with 0 bytes), stream it
	// Only error if no requests exist at all
	if best != nil {
		// Phase 2 / real-streaming-default — LIVE-MODE 0-BYTE GUARD.
		// In buffered mode, today promotes a 0-byte winner and keeps
		// streaming (unaffected). In live mode, a 0-byte deadline fire
		// means "no forwardable byte within StreamDeadline" — surface as
		// an in-band SSE error envelope + cancelAll (the no-content
		// branch below), instead of forwarding a phantom 0-byte winner.
		if c.liveFirstByteGate && bestLen == 0 {
			log.Printf("[RACE] Live-mode deadline 0-byte guard firing: no forwardable byte within StreamDeadline")

			// Build error info for the handler to use
			errInfo := c.getFinalErrorInfoLocked()
			errInfo.Message = "Request timeout - no response received"
			c.streamDeadlineError = &errInfo

			c.publishEvent("race_all_failed", map[string]interface{}{
				"total_attempts": len(c.requests),
				"duration_ms":    time.Since(c.startTime).Milliseconds(),
				"reason":         "streaming_deadline_no_content_live",
			})

			c.onceDone.Do(func() { close(c.done) })
			c.onceStream.Do(func() { close(c.streamCh) })

			// PINNED cancellation form (C2 / W1): capture the request
			// set under the held c.mu, RELEASE the lock, THEN cancel.
			// cancelAll → cancelAllExcept → c.mu re-acquires at :760;
			// `req.Cancel()` itself only acquires `r.mu` and is not the
			// deadlock source, but the pinned form keeps one shape.
			// Releasing first also matches the codebase pattern at
			// :441-446 (cancel-and-exit). The deferred unlock is
			// REMOVED in this live-guard branch (W1) — the explicit
			// Unlock() below replaces it; double-unlock would fail-fast.
			c.mu.Unlock()
			c.cancelAll()
			return
		}

		c.winner = best
		c.winnerIdx = best.id
		// Sticky last-success: a deadline pick is a best-partial
		// artifact, never a delivered success — mark it so the
		// record sites skip it.
		best.deadlinePick = true

		log.Printf("[RACE] Picked best buffer: request %d (%s, %s) with %d bytes",
			best.id, best.modelType, best.modelID, bestLen)

		// Publish race_deadline_pick event
		c.publishEvent("race_deadline_pick", map[string]interface{}{
			"winner_index": c.winnerIdx,
			"winner_type":  string(best.modelType),
			"winner_model": best.modelID,
			"buffer_bytes": bestLen,
			"duration_ms":  time.Since(c.startTime).Milliseconds(),
		})

		// Signal that streaming can start
		c.onceStream.Do(func() { close(c.streamCh) })

		// DON'T cancel the winner - let it continue streaming until complete or hardDeadline
		// This is per the unified race retry design: "Continue streaming winner until complete or hard deadline"
		// The winner was picked because it has the most content, so we want to keep receiving more data.

		// Cancel only the other requests
		for _, req := range c.requests {
			if req != nil && req != best {
				req.Cancel()
			}
		}
		c.mu.Unlock()
	} else {
		// No content at all - all failed or no requests started
		log.Printf("[RACE] Streaming deadline reached, no content available")

		// Build error info for the handler to use
		errInfo := c.getFinalErrorInfoLocked()
		errInfo.Message = "Request timeout - no response received"
		c.streamDeadlineError = &errInfo

		c.publishEvent("race_all_failed", map[string]interface{}{
			"total_attempts": len(c.requests),
			"duration_ms":    time.Since(c.startTime).Milliseconds(),
			"reason":         "streaming_deadline_no_content",
		})

		c.onceDone.Do(func() { close(c.done) })
		c.onceStream.Do(func() { close(c.streamCh) })

		// Match the live-guard sequencing: release c.mu before cancelAll
		// to avoid the c.mu re-entrance chain (cancelAllExcept :760).
		// Buffered mode callers benefit from the same safe ordering
		// (today's defer-form Unlock would also be safe, but this
		// single form keeps both branches consistent).
		c.mu.Unlock()
		c.cancelAll()
	}
}

// handleHardDeadline forcefully terminates all requests when MaxGenerationTime is reached
// This is the absolute hard timeout - no requests are allowed to continue past this point.
func (c *raceCoordinator) handleHardDeadline() {
	c.mu.Lock()
	defer c.mu.Unlock()

	log.Printf("[RACE] Hard deadline reached, cancelling all requests")

	// Publish race_hard_deadline event
	c.publishEvent("race_hard_deadline", map[string]interface{}{
		"duration_ms": time.Since(c.startTime).Milliseconds(),
	})

	// Cancel ALL requests immediately (including winner if any)
	for _, req := range c.requests {
		if req != nil {
			req.Cancel()
		}
	}

	// Signal done
	c.onceDone.Do(func() { close(c.done) })
	c.onceStream.Do(func() { close(c.streamCh) })
}

func (c *raceCoordinator) cancelAll() {
	c.cancelAllExcept(nil)
}

// cancelAllExcept cancels all requests except the winner.
// Each call to req.Cancel() performs full cleanup:
//   - Cancels the request context
//   - Drains and closes the HTTP response body
//   - Releases the stream buffer
func (c *raceCoordinator) cancelAllExcept(winner *upstreamRequest) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, req := range c.requests {
		if req != winner {
			req.Cancel()
		}
	}
}

// execute is a wrapper for executeRequest.
//
// Phase 3 / Tasks 4 + 6 + race-coordinator-level wiring (Round 3c C3):
// receive the per-coordinator engine + conversationKey from the
// constructor; forward them down to executeRequest on every call so
// the publish site has them.
func (c *raceCoordinator) execute(req *upstreamRequest) {
	// Create context for this specific attempt
	ctx, cancel := context.WithCancel(c.baseCtx)
	req.SetContext(ctx, cancel)

	// Build the per-request "publish model_credential_selected" callback
	// bound to this coordinator's eventBus. The callback reads
	// NewlyBound off the resolved struct (already populated by
	// executeInternalRequest) and fires the event on the bus.
	var publish func(modelID string, cred models.ResolvedCredential)
	if c.eventBus != nil && c.engine != nil {
		publish = func(modelID string, cred models.ResolvedCredential) {
			keyPrefix := c.conversationKey
			if len(keyPrefix) > 8 {
				keyPrefix = keyPrefix[:8]
			}
			c.eventBus.Publish(events.Event{
				Type:      "model_credential_selected",
				Timestamp: time.Now().Unix(),
				Data: map[string]interface{}{
					"model_id":                modelID,
					"credential_id":           cred.CredentialID,
					"conversation_key_prefix": keyPrefix,
					"id":                      c.requestID,
				},
			})
		}
	}

	// load-balance-groups / Q15: the member-resolution observability
	// event, published at the SAME W-1 site — newlyBound-ONLY (the
	// GROUP pick's W-1 signal, mirroring the
	// model_credential_selected first-binding asymmetry; do not spam
	// every request). Payload per the plan: {group_id, member
	// model_id, upstream_provider, upstream_model}. Non-group
	// traffic keeps model_credential_selected unchanged.
	var publishMember func(memberID string, cred models.ResolvedCredential)
	if c.eventBus != nil && c.groupPick != nil {
		gp := c.groupPick
		publishMember = func(memberID string, cred models.ResolvedCredential) {
			payload := map[string]interface{}{
				"group_id":          gp.GroupID,
				"member_id":         memberID,
				"upstream_provider": cred.Provider,
				"upstream_model":    cred.InternalModel,
				"id":                c.requestID,
			}
			// group-backup-members / D5 observability: the
			// additive `pool` field rides on the same event as
			// reason / reason_class. It is emitted ONLY for a
			// BACKUP-pool serve ("backup" — every default member
			// in the run's exclusion set had failed); a
			// default-pool serve leaves the field ABSENT (the
			// FE's existing default branch — no "default" value
			// is ever emitted). The enclosing `c.groupPick != nil`
			// guard means a group rewrite always happened.
			if gp.Pool == models.PoolBackup {
				payload["pool"] = models.PoolBackup
			}
			// group-backup-members / D8 observability: the
			// additive `binding` field marks a selection-time
			// repair — the conversation's sticky binding pointed
			// at a member that left the group, so the binding was
			// discarded and the member re-picked. Omitted unless
			// the condition occurs (same additive style as the
			// `pool` field above: present only on the condition,
			// absent otherwise).
			if gp.BindingStale {
				payload["binding"] = "stale"
			}
			c.eventBus.Publish(events.Event{
				Type:      "model_member_selected",
				Timestamp: time.Now().Unix(),
				Data:      payload,
			})
		}
	}
	req.onMemberSelected = publishMember

	err := executeRequest(ctx, c.cfg, c.req, c.rawBody, req, c.interleaved, c.conversationKey, c.engine, publish)

	// Upstream pairing-rejection guard: a poisoned attempt's terminal
	// state belongs to the guard (manage() already flipped it to failed
	// with errUpstreamPairingRejection and published race_request_failed).
	// The executor here exits either mid-stream (poison cancelled the
	// context) or with err==nil having buffered the signature frames —
	// either way neither MarkFailed nor MarkCompleted may run: a late
	// MarkFailed would overwrite the guard sentinel (breaking the
	// exhaustion path's classified reason), and a late MarkCompleted
	// would resurrect a poisoned attempt toward winner eligibility.
	if req.IsPoisoned() {
		if err != nil {
			log.Printf("[RACE] Request %d executor exited after pairing-rejection poison (post-poison error discarded: %v)", req.id, err)
		} else {
			log.Printf("[RACE] Request %d buffered pairing-rejection frames before poison took effect — guard owns the terminal state", req.id)
		}
		return
	}

	if err != nil {
		req.MarkFailed(err)
		// Sticky last-success invalidation (winner-matched inside the
		// engine): a failed attempt of the RECORDED winner unpins the
		// slot; any other model's failure is a no-op. Cred-failover
		// exhaustion needs no separate hook — every failover attempt
		// already flowed through here. Mid-attempt cancels surface as
		// ctx.Err()/context.Canceled through executeRequest and land
		// here too (accepted transient unpin, design §8.12).
		if c.cfg.stickyActive(c.conversationKey) {
			c.sticky.InvalidateOnFailure(c.conversationKey, req.modelID)
		}
		// race-error-reason-classification — the [RACE] log line and the
		// event payload carry the classified, bounded reason; the raw
		// error (which can embed an entire upstream error body on the
		// context-overflow path) never reaches a log or payload
		// unbounded.
		failReason := classifyRequestFailure(req)
		log.Printf("[RACE] Request %d failed: %s", req.id, failReason)

		// Publish race_request_failed event with detailed error info.
		// Existing keys unchanged (`error` is the same error text, now
		// bounded + redacted); reason/reason_class/reason_status are
		// additive (appendReasonFields fixes their key sequence).
		failEventData := map[string]interface{}{
			"request_index": req.id,
			"model":         req.modelID,
			"type":          string(req.modelType),
			"error":         failReason.Detail,
		}
		appendReasonFields(failEventData, failReason)
		c.publishEvent("race_request_failed", failEventData)
	} else {
		req.MarkCompleted()
		log.Printf("[RACE] Request %d completed successfully", req.id)
	}
}

// GetWinner returns the winner request
func (c *raceCoordinator) GetWinner() *upstreamRequest {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.winner
}

// WaitForWinner blocks until a winner is found or all requests fail
func (c *raceCoordinator) WaitForWinner() *upstreamRequest {
	select {
	case <-c.streamCh:
		return c.GetWinner()
	case <-c.done:
		return c.GetWinner()
	case <-c.baseCtx.Done():
		return nil
	}
}

// GetCommonFailureStatus returns the HTTP status code if all failed requests share the same status, else 0
func (c *raceCoordinator) GetCommonFailureStatus() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.requests) == 0 {
		return 0
	}

	var commonStatus int
	for _, req := range c.requests {
		err := req.GetError()
		if err == nil {
			return 0 // Not failed yet or didn't fail
		}

		// First check if we have the HTTP status stored
		httpStatus := req.GetHTTPStatus()
		if httpStatus >= 400 {
			// Use the stored HTTP status directly
			if commonStatus == 0 {
				commonStatus = httpStatus
			} else if commonStatus != httpStatus {
				return 0 // Mismatch
			}
			continue
		}

		// Fallback: Parse status text like "upstream returned error: 429 Too Many Requests"
		errStr := err.Error()
		var status int
		if strings.Contains(errStr, "upstream returned error: ") {
			fmt.Sscanf(errStr, "upstream returned error: %d", &status)
		} else if strings.Contains(errStr, "idle timeout") || strings.Contains(errStr, "context") || strings.Contains(errStr, "timeout") {
			status = http.StatusGatewayTimeout
		} else if strings.Contains(errStr, "buffer limit") {
			status = http.StatusRequestEntityTooLarge
		} else {
			status = http.StatusBadGateway
		}

		if status == 0 {
			status = http.StatusBadGateway
		}

		if commonStatus == 0 {
			commonStatus = status
		} else if commonStatus != status {
			return 0 // Mismatch
		}
	}

	return commonStatus
}

// RaceStats contains statistics about a completed race
type RaceStats struct {
	TotalRequests   int           `json:"total_requests"`
	WinnerType      string        `json:"winner_type"`
	WinnerModel     string        `json:"winner_model"`
	WinnerIndex     int           `json:"winner_index"`
	Duration        time.Duration `json:"duration"`
	SpawnTriggers   []string      `json:"spawn_triggers"`
	FailedCount     int           `json:"failed_count"`
	WinnerBufferLen int64         `json:"winner_buffer_bytes"`
}

// GetStats returns statistics about the race for logging/metrics
func (c *raceCoordinator) GetStats() RaceStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	stats := RaceStats{
		TotalRequests: len(c.requests),
		WinnerIndex:   c.winnerIdx,
		Duration:      time.Since(c.startTime),
		FailedCount:   c.failedCount,
	}

	// Convert spawn triggers to strings
	for _, t := range c.spawnTriggers {
		stats.SpawnTriggers = append(stats.SpawnTriggers, string(t.trigger))
	}

	// Winner info
	if c.winner != nil {
		stats.WinnerType = string(c.winner.modelType)
		stats.WinnerModel = c.winner.modelID
		stats.WinnerBufferLen = c.winner.GetBuffer().TotalLen()
	}

	return stats
}

// GetRequestStatuses now lives with the Phase 3 Task 18 helpers at the
// bottom of this file (S4 — credFailover rows surfacing). The original
// three-key shape (main/second/fallback) is preserved; the additive
// cred_failover_<n> keys extend it without breaking legacy readers.

// GetStreamDeadlineError returns the error info if stream deadline fired with no content
func (c *raceCoordinator) GetStreamDeadlineError() *FinalErrorInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.streamDeadlineError
}

// FinalErrorInfo contains information for building the final error response
type FinalErrorInfo struct {
	HTTPStatus int    // HTTP status code to return
	ErrorType  string // Error type (e.g., "rate_limit", "upstream_error")
	ErrorCode  string // Error code for retry detection (e.g., "rate_limit", "unavailable")
	Message    string // Human-readable error message
}

// GetFinalErrorInfo builds the final error info based on all failed requests.
// This implements the OpenCode-compatible error format with proper rate_limit detection.
// Key rules:
// - Rate limit code is only added after ALL models are exhausted with 429
// - Context overflow errors never get rate_limit code (triggers compaction instead)
// - HTTP 503 maps to type "too_many_requests" with code "unavailable"
func (c *raceCoordinator) GetFinalErrorInfo() FinalErrorInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.requests) == 0 {
		return FinalErrorInfo{
			HTTPStatus: http.StatusBadGateway,
			ErrorType:  models.ErrorTypeServerError,
			ErrorCode:  "",
			Message:    "No upstream requests were made",
		}
	}

	// Collect all failed requests and their errors
	var failedRequests []*upstreamRequest
	var lastError error
	anyModel429 := false
	allModelsExhausted := true
	hasContextOverflow := false

	for _, req := range c.requests {
		err := req.GetError()
		if err != nil {
			failedRequests = append(failedRequests, req)
			lastError = err

			// Check for 429 rate limit
			if req.GetHTTPStatus() == http.StatusTooManyRequests {
				anyModel429 = true
			}

			// Check for context overflow
			if models.IsContextOverflowError(err) {
				hasContextOverflow = true
			}
		} else {
			// A request succeeded (shouldn't happen if we're building error, but check)
			allModelsExhausted = false
		}
	}

	// If no errors, this shouldn't be called
	if lastError == nil {
		return FinalErrorInfo{
			HTTPStatus: http.StatusBadGateway,
			ErrorType:  models.ErrorTypeServerError,
			ErrorCode:  "",
			Message:    "Unknown error",
		}
	}

	// Determine HTTP status from common status
	httpStatus := c.getCommonFailureStatusLocked()

	// Build error message
	message := "All upstream models failed"
	if lastError != nil {
		message = lastError.Error()
	}

	// Upstream pairing-rejection guard exhaustion — the final error is
	// the guard sentinel (a poisoned attempt's text), so annotate WHY the
	// client is seeing a real upstream error instead of a completion: the
	// transparent re-drive budget was spent and every completion still
	// carried the rejection signature.
	if c.pairingGuardExhausted && lastError != nil && errors.Is(lastError, errUpstreamPairingRejection) {
		message = lastError.Error() + " [guarded pairing-retry budget exhausted]"
	}

	// CRITICAL: Never add rate_limit code to context overflow errors
	// OpenCode checks context overflow BEFORE retry logic to trigger compaction
	if hasContextOverflow {
		return FinalErrorInfo{
			HTTPStatus: http.StatusBadRequest,
			ErrorType:  models.ErrorTypeContextOverflow,
			ErrorCode:  "", // No code - let OpenCode trigger compaction
			Message:    message,
		}
	}

	// Determine error type and code based on HTTP status
	var errType string
	var errCode string

	switch httpStatus {
	case http.StatusTooManyRequests: // 429
		// Only add rate_limit code if ALL models exhausted
		// If we have working models, no need to signal retry
		if allModelsExhausted && anyModel429 {
			errType = models.ErrorTypeRateLimit
			errCode = models.ErrorCodeRateLimit
			message = "All models rate limited"
		} else {
			errType = models.ErrorTypeRateLimit
			// No code if fallback worked (but this shouldn't happen since all failed)
			errCode = ""
		}
	case http.StatusBadGateway: // 502
		errType = models.ErrorTypeUpstreamError
		errCode = models.ErrorCodeUnavailable
	case http.StatusNotFound: // 404
		errType = models.ErrorTypeServerError
		errCode = "model_not_found"
	case http.StatusInternalServerError: // 500
		errType = models.ErrorTypeServerError
		errCode = ""
	case http.StatusServiceUnavailable: // 503
		errType = models.ErrorTypeTooManyRequests
		errCode = models.ErrorCodeUnavailable
	case http.StatusGatewayTimeout: // 504
		errType = models.ErrorTypeUpstreamError
		errCode = ""
	default:
		// httpStatus 0 means mismatched statuses (e.g., 429 + context canceled)
		// Default to BadGateway to ensure valid HTTP status
		httpStatus = http.StatusBadGateway
		errType = models.ErrorTypeServerError
		errCode = ""
	}

	return FinalErrorInfo{
		HTTPStatus: httpStatus,
		ErrorType:  errType,
		ErrorCode:  errCode,
		Message:    message,
	}
}

// getFinalErrorInfoLocked is the internal version of GetFinalErrorInfo that assumes lock is held
func (c *raceCoordinator) getFinalErrorInfoLocked() FinalErrorInfo {
	if len(c.requests) == 0 {
		return FinalErrorInfo{
			HTTPStatus: http.StatusBadGateway,
			ErrorType:  models.ErrorTypeServerError,
			ErrorCode:  "",
			Message:    "No upstream requests were made",
		}
	}

	// Collect all failed requests and their errors
	var failedRequests []*upstreamRequest
	var lastError error
	anyModel429 := false
	allModelsExhausted := true
	hasContextOverflow := false

	for _, req := range c.requests {
		err := req.GetError()
		if err != nil {
			failedRequests = append(failedRequests, req)
			lastError = err

			// Check for 429 rate limit
			if req.GetHTTPStatus() == http.StatusTooManyRequests {
				anyModel429 = true
			}

			// Check for context overflow
			if models.IsContextOverflowError(err) {
				hasContextOverflow = true
			}
		} else {
			allModelsExhausted = false
		}
	}

	if lastError == nil {
		return FinalErrorInfo{
			HTTPStatus: http.StatusBadGateway,
			ErrorType:  models.ErrorTypeServerError,
			ErrorCode:  "",
			Message:    "Unknown error",
		}
	}

	httpStatus := c.getCommonFailureStatusLocked()

	message := "All upstream models failed"
	if lastError != nil {
		message = lastError.Error()
	}

	if hasContextOverflow {
		return FinalErrorInfo{
			HTTPStatus: http.StatusBadRequest,
			ErrorType:  models.ErrorTypeContextOverflow,
			ErrorCode:  "",
			Message:    message,
		}
	}

	var errType string
	var errCode string

	switch httpStatus {
	case http.StatusTooManyRequests:
		if allModelsExhausted && anyModel429 {
			errType = models.ErrorTypeRateLimit
			errCode = models.ErrorCodeRateLimit
			message = "All models rate limited"
		} else {
			errType = models.ErrorTypeRateLimit
			errCode = ""
		}
	case http.StatusBadGateway:
		errType = models.ErrorTypeUpstreamError
		errCode = models.ErrorCodeUnavailable
	case http.StatusNotFound: // 404
		errType = models.ErrorTypeServerError
		errCode = "model_not_found"
	case http.StatusInternalServerError: // 500
		errType = models.ErrorTypeServerError
		errCode = ""
	case http.StatusServiceUnavailable:
		errType = models.ErrorTypeTooManyRequests
		errCode = models.ErrorCodeUnavailable
	case http.StatusGatewayTimeout:
		errType = models.ErrorTypeUpstreamError
		errCode = ""
	default:
		// httpStatus 0 means mismatched statuses (e.g., 429 + context canceled)
		// Default to BadGateway to ensure valid HTTP status
		httpStatus = http.StatusBadGateway
		errType = models.ErrorTypeServerError
		errCode = ""
	}

	return FinalErrorInfo{
		HTTPStatus: httpStatus,
		ErrorType:  errType,
		ErrorCode:  errCode,
		Message:    message,
	}
}

// getCommonFailureStatusLocked is the internal version of GetCommonFailureStatus that assumes lock is held
func (c *raceCoordinator) getCommonFailureStatusLocked() int {
	if len(c.requests) == 0 {
		return 0
	}

	var commonStatus int
	for _, req := range c.requests {
		err := req.GetError()
		if err == nil {
			return 0 // Not failed yet or didn't fail
		}

		// First check if we have the HTTP status stored
		httpStatus := req.GetHTTPStatus()
		if httpStatus >= 400 {
			// Use the stored HTTP status directly
			if commonStatus == 0 {
				commonStatus = httpStatus
			} else if commonStatus != httpStatus {
				return 0 // Mismatch
			}
			continue
		}

		// Fallback: Parse status text
		errStr := err.Error()
		var status int
		if strings.Contains(errStr, "upstream returned error: ") {
			fmt.Sscanf(errStr, "upstream returned error: %d", &status)
		} else if strings.Contains(errStr, "idle timeout") || strings.Contains(errStr, "context") || strings.Contains(errStr, "timeout") {
			status = http.StatusGatewayTimeout
		} else if strings.Contains(errStr, "buffer limit") {
			status = http.StatusRequestEntityTooLarge
		} else {
			status = http.StatusBadGateway
		}

		if status == 0 {
			status = http.StatusBadGateway
		}

		if commonStatus == 0 {
			commonStatus = status
		} else if commonStatus != status {
			return 0 // Mismatch
		}
	}

	return commonStatus
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3 / Task 18 (Round 3b B1/B3 + Round 3c C1/C3/W3) — coordinator
// credential-failover helpers. All *Locked functions run under c.mu from
// the manage() loop (single-goroutine mutation invariant, R3-5).
// ─────────────────────────────────────────────────────────────────────────────

// credFailoverPlan is the decision output of credFailoverPlanLocked: the
// spawn the Case-1 branch should perform INSTEAD of a model fallback.
type credFailoverPlan struct {
	credentialID     string            // reselected credential (engine output) — rides spawnTriggerInfo.credentialID
	fromCredentialID string            // the credential that 429'd / quota-failed
	class            credFailoverClass // failure class — drives the trigger vocabulary + event reason + cooldown policy
	retryAfter       time.Duration     // from the upstream Retry-After header (0 ⇒ class default cooldown)
	// cooldown is the EFFECTIVE seed the engine applied for this
	// failover (CooldownFor(class, retryAfter)) — mirrors
	// model_credential_failover's cooldown_ms payload key.
	cooldown      time.Duration
	soonestExpiry bool // W3: single-shot SoonestExpiry attempt
}

// modelAttemptCountLocked returns the B1 separate-accounting count of
// MODEL attempts: rows with modelType ∈ {main, second, fallback} only.
// modelTypeCredFailover rows are EXEMPT from the spawn-window gate and
// from the all-failed accounting (Round 3b B1 / Round 3c C1).
func (c *raceCoordinator) modelAttemptCountLocked() int {
	n := 0
	for _, r := range c.requests {
		switch r.modelType {
		case modelTypeMain, modelTypeSecond, modelTypeFallback:
			n++
		}
	}
	return n
}

// credFailoverEligibleLocked is the cheap (side-effect-free) form of
// credFailoverEligibleWithBudget() used by the C1-hoisted admission
// gate. It performs NO engine calls (ExcludeAndReselect has cooldown
// side effects and must run at most once per Case-1 decision).
//
// Layered credential fallback (Fix 1) widens the error class from
// rate-limit-only to rate-limit + quota (classifyCredFailure), and
// (Fix 2) moves the rotation budget from the GLOBAL failoverAttempts
// counter to a PER-MODEL counter — every model in the chain gets its
// own full len(credentials)−1 Layer-1 budget.
func (c *raceCoordinator) credFailoverEligibleLocked() bool {
	if c.engine == nil || c.cfg == nil || c.cfg.ModelsConfig == nil || len(c.requests) == 0 {
		return false
	}
	latest := c.requests[len(c.requests)-1]
	if !latest.IsDone() || latest.GetError() == nil {
		return false
	}
	// Fix 1 — BOTH eligible classes gate here: rate_limit (429-style)
	// and quota_exceeded (402 / insufficient credit). Everything else
	// (401/auth, 5xx, timeouts) stays ineligible — eligibility is
	// never widened beyond these two classes.
	if classifyCredFailure(latest.GetError()) == credFailoverNone {
		return false
	}
	// Task 23 pre-first-byte guard (race path): c.winner == nil. Framing
	// headers + ": connected\n\n" precede coordinator.Start() but carry
	// NO model content (handler.go:783-797); the guard is the winner,
	// not headersSent.
	if c.winner != nil {
		return false
	}
	// P0 shadowfix — resolve by ROW IDENTITY (directGetModel →
	// GetModelNative), not by the client-facing claims-first GetModel.
	// This read decides the CREDENTIAL-FAILOVER eligibility of the
	// failed attempt: the credential count comes from the attempt's
	// own model row. For a member ID claimed by its own group (the
	// day-1 self-shadow shape coding-group[coding]) the claims-first
	// read returned the GROUP row (zero credentials) ⇒ the count
	// check `len(Credentials) <= 1` failed ⇒ credential failover was
	// SILENTLY DISABLED for that member and the upstream 429 bubbled
	// out of the member instead of being absorbed by the member's own
	// credential LB. Execution-resolution reads are row-identity reads
	// (Am22 L1); the client-facing seam is untouched.
	modelCfg := directGetModel(c.cfg, latest.modelID)
	// remove-external-upstream P1 (A20): the `!modelCfg.Internal`
	// conjunct is GONE — every resolvable model is internal by
	// construction (the boundary 404 gate rejects the rest), so
	// credential failover eligibility keys on credential count only.
	if modelCfg == nil || len(modelCfg.Credentials) <= 1 {
		return false
	}
	// W3: once the single SoonestExpiry attempt is spent FOR THIS
	// MODEL, the next eligible failure routes straight to model
	// advance. Fix 2: per-model flag — a later model in the chain
	// starts with its own unspent single-shot.
	if c.soonestExpiryAttemptedByModel[latest.modelID] {
		return false
	}
	// R3-5 budget, per model (Fix 2): each remaining credential of THE
	// FAILED MODEL at most once per request; the model's total
	// failover attempts ≤ len(its credentials)−1. The legacy global
	// failoverAttempts counter (kept for the event attempt_index) no
	// longer bounds eligibility — under it, a second model in the
	// chain inherited the first model's spend and got ZERO rotations
	// (the documented WIP limitation in
	// TestHandler_TwoModelFallback429_WrongModelRegression).
	return c.failoverAttemptsByModel[latest.modelID] < len(modelCfg.Credentials)-1
}

// fallbackAdvanceTargetLocked (layered credential fallback — Fix 2)
// returns the NEXT model in the chain after failedModelID — the model a
// modelTypeFallback row must target once failedModelID's Layer-1
// credential rotation is exhausted. The bool reports whether an advance
// target exists at all: false ⇒ the failed model is the chain's LAST
// (or isn't in the chain) and the error path is genuinely TERMINAL —
// the Case-1 site schedules no spawn (final review A; the legacy
// empty ⇒ c.models[1] mapping would re-spawn an already-exhausted
// model). The legacy default survives ONLY on the idle-timeout path,
// whose triggerInfo legitimately carries no modelID.
//
// Derived from the chain (index-of-failed + 1), not stored: the failed
// attempt's own modelID is the single source of truth, so the order is
// exactly the layered spec (models[0] → models[1] → models[2] → …)
// with no drift between a tracked index and the requests actually
// spawned. Runs under c.mu from the manage() loop.
func (c *raceCoordinator) fallbackAdvanceTargetLocked(failedModelID string) (target string, hasAdvance bool) {
	if len(c.models) < 2 {
		return "", false
	}
	for i, m := range c.models {
		if m == failedModelID {
			if i+1 < len(c.models) {
				return c.models[i+1], true
			}
			return "", false
		}
	}
	return "", false
}

// credFailoverPlanLocked runs the full Round-3b/3c decision chain for
// the failed latestReq: classify the failure class (rate-limit vs
// quota — Fix 1), extract Retry-After (errors.As — the executor
// propagates the raw *ProviderError up to the coordinator), resolve the
// class's effective cooldown (CooldownFor), call
// engine.ExcludeAndReselectForClass (mode-aware per B3/C2), and map the mode:
//
//   - ReselectHealthy (incl. B2 no-op + empty-key fresh pick) → proceed
//     with the returned credential (spawn below).
//   - ReselectSoonestExpiry → single attempt (soonestExpiry=true; the
//     caller sets the W3 single-shot flag after spawning).
//   - ReselectNone → fall through to model fallback (plan == nil).
//
// The tried-set intersection (R3-5) makes re-selection of a
// tried-this-request credential impossible — no loops by construction.
func (c *raceCoordinator) credFailoverPlanLocked(latestReq *upstreamRequest) *credFailoverPlan {
	if !c.credFailoverEligibleLocked() {
		return nil
	}

	err := latestReq.GetError()

	// Fix 1 — classify ONCE here: the class drives the trigger
	// vocabulary, the event reason, AND the cooldown policy.
	class := classifyCredFailure(err)

	// Retry-After from the extended ProviderError (Task 17 — survives
	// wrapping; errors.As sees through fmt.Errorf("%w", ...) up the
	// executor's return path). Zero ⇒ the class default applies (the
	// classifier itself does NOT default).
	var retryAfter time.Duration
	var providerErr *providers.ProviderError
	if errors.As(err, &providerErr) {
		retryAfter = providerErr.RetryAfter
	}
	cooldown := effectiveCooldown(c.engine, class, retryAfter)

	// The credential that failed: the attempt's resolved credential.
	// modelTypeSecond rows resolve the PRIMARY credential at the
	// executor layer (race_executor.go secondary branch) — a failure
	// there re-keys failover to the primary credential of c.models[0]
	// (specified behavior per Round 3b, not an accident).
	fromCred := latestReq.resolved.CredentialID
	if fromCred == "" {
		// Resolution itself failed (no credential was used) — not a
		// credential failover candidate.
		return nil
	}

	reselected, mode := c.engine.ExcludeAndReselectForClass(
		latestReq.modelID,
		c.conversationKey, // Round 3c C3(3): injected via constructor
		fromCred,
		class.toEngineClass(),
		retryAfter,
	)

	switch mode {
	case credentiallb.ReselectHealthy:
		if reselected == "" || reselected == fromCred || c.triedCredIDs[reselected] {
			// Engine returned nothing new (or a credential this request
			// already tried — belt-and-braces on top of the engine's
			// cooldown skip): fall through to model advance.
			log.Printf("[LB-FAILOVER] model=%s %s on cred=%s but no healthy untried credential (reselected=%q); advancing to next model",
				latestReq.modelID, class.readable(), fromCred, reselected)
			return nil
		}
		log.Printf("[LB-FAILOVER] model=%s %s on cred=%s; failing over to cred=%s (class=%s, cooldown=%s, rotation %d/%d)",
			latestReq.modelID, class.readable(), fromCred, reselected, class, cooldown,
			c.failoverAttemptsByModel[latestReq.modelID]+1, modelBudget(c, latestReq.modelID))
		return &credFailoverPlan{
			credentialID:     reselected,
			fromCredentialID: fromCred,
			class:            class,
			retryAfter:       retryAfter,
			cooldown:         cooldown,
		}

	case credentiallb.ReselectSoonestExpiry:
		// 0-of-N healthy (Round 3b 0-of-N pin): the engine already
		// emitted its WARN; caller performs a SINGLE attempt then falls
		// through (W3 flag + F.4 contract).
		if reselected == "" || reselected == fromCred || c.triedCredIDs[reselected] {
			return nil
		}
		log.Printf("[WARN] [LB-FAILOVER] model=%s all credentials cooling (%s); single attempt with soonest-expiring cred=%s then model advance",
			latestReq.modelID, class, reselected)
		return &credFailoverPlan{
			credentialID:     reselected,
			fromCredentialID: fromCred,
			class:            class,
			retryAfter:       retryAfter,
			cooldown:         cooldown,
			soonestExpiry:    true,
		}

	default: // credentiallb.ReselectNone — no credential available
		// (single-credential model or 0 valid credentials; B2 no-op and
		// empty-key are ReselectHealthy per the C2 unified enum).
		log.Printf("[LB-FAILOVER] model=%s no reselectable credential (mode=%v, class=%s); advancing to next model",
			latestReq.modelID, mode, class)
		return nil
	}
}

// modelBudget returns len(credentials)−1 (R3-5 budget cap) for the
// INFO log line; 0 when the model config is unavailable.
//
// P0 shadowfix — row-identity read (directGetModel), matching the
// eligibility gate above: the budget is derived from the ATTEMPT's
// own model row, so a member ID claimed by its own group is counted
// from the member (not from the credential-less group row).
func modelBudget(c *raceCoordinator, modelID string) int {
	if c.cfg == nil || c.cfg.ModelsConfig == nil {
		return 0
	}
	mc := directGetModel(c.cfg, modelID)
	if mc == nil {
		return 0
	}
	return len(mc.Credentials) - 1
}

// publishCredentialFailover (Task 20 / R3-8) emits the canonical
// model_credential_failover event with the payload
// {model_id, from_credential_id, to_credential_id, reason,
// retry_after_ms, cooldown_ms, attempt_index}. reason carries the
// FAILURE CLASS vocabulary ("rate_limit" | "quota_exceeded" — Fix 1);
// cooldown_ms mirrors the engine's effective seed for that class
// (plan.cooldown — CooldownFor output: retryAfter when the upstream
// supplied it [rate-limit], the configured quota cooldown otherwise
// [quota]) — the engine itself never publishes (it must not import
// pkg/events).
func (c *raceCoordinator) publishCredentialFailover(modelID, fromID, toID string, plan *credFailoverPlan) {
	if c.eventBus == nil {
		return
	}
	class := credFailoverNone
	retryAfter := time.Duration(0)
	cooldown := credentiallb.DefaultCooldown
	if plan != nil {
		class = plan.class
		retryAfter = plan.retryAfter
		cooldown = plan.cooldown
		if cooldown <= 0 {
			cooldown = credentiallb.DefaultCooldown
		}
	}
	data := map[string]interface{}{
		"model_id":           modelID,
		"from_credential_id": fromID,
		"to_credential_id":   toID,
		"reason":             class.String(),
		"retry_after_ms":     retryAfter.Milliseconds(),
		"cooldown_ms":        cooldown.Milliseconds(),
		"attempt_index":      c.failoverAttempts + 1,
	}
	if c.requestID != "" {
		data["id"] = c.requestID
	}
	c.eventBus.Publish(events.Event{
		Type:      credentiallb.EventCredentialFailover,
		Timestamp: time.Now().Unix(),
		Data:      data,
	})
}

// GetRequestStatuses (Phase 3 / Round 3c — S4): credFailover rows are
// surfaced as additive "cred_failover_<n>" keys alongside the legacy
// main/second/fallback slots (B1 separate accounting keeps the legacy
// three keys model-attempt-only). Each row's reselected credential is
// exposed as "cred_failover_<n>_credential" for operator visibility.
//
// Round 3j W4 — the fallback to `req.resolved.CredentialID` was DROPPED
// rather than reading it under `req.mu`: the only meaningful credential
// identity for a credFailover row is the one the coordinator pre-resolved
// at the Case-1 classification site (`req.reselectedCredentialID`, set
// under c.mu before `go execute(req)` — race-free by construction).
// Reading `req.resolved.CredentialID` here without locking would race
// against `executeInternalRequest`'s lock-free write at
// race_executor.go:293. The belt-and-braces check in
// resolveSpecificInternalCredential (Round 3j C1 step 4) already
// guarantees that a row whose reselected credential was rejected never
// gets past the executor — so the resolved-fallback was both racy AND
// redundant.
func (c *raceCoordinator) GetRequestStatuses() map[string]string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	statuses := make(map[string]string)
	// Initialize all as not_started
	statuses["main"] = "not_started"
	statuses["second"] = "not_started"
	statuses["fallback"] = "not_started"

	credIdx := 0
	for _, req := range c.requests {
		var status string
		switch req.GetStatus() {
		case statusCompleted:
			status = "success"
		case statusFailed:
			status = "failed"
		default:
			// pending, running, streaming - treat as not_started (cancelled or in-progress)
			status = "not_started"
		}

		switch req.modelType {
		case modelTypeMain:
			statuses["main"] = status
		case modelTypeSecond:
			statuses["second"] = status
		case modelTypeFallback:
			statuses["fallback"] = status
		case modelTypeCredFailover:
			key := fmt.Sprintf("cred_failover_%d", credIdx)
			statuses[key] = status
			// Round 3j W4 — use ONLY reselectedCredentialID.
			// Reading req.resolved.CredentialID here would race the
			// executor's lock-free write (race_executor.go:293); the
			// reselected credential IS the row's identity for operator
			// observability.
			if req.reselectedCredentialID != "" {
				statuses[key+"_credential"] = req.reselectedCredentialID
			}
			credIdx++
		}
	}

	return statuses
}
