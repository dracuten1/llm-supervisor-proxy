package proxy

// ─────────────────────────────────────────────────────────────────────────────
// race_reason.go — classified, bounded failure reasons for race/fallback
// event-log emission (race-error-reason-classification).
//
// What this file owns:
//
//   - failureReason ........ the classified reason record: a stable
//     greppable class token, the upstream HTTP status when one is
//     known, and a SANITIZED + DETERMINISTICALLY-BOUNDED detail
//     fragment.
//
//   - classifyAttemptFailure / classifyRequestFailure — pure
//     classification over (error, stored HTTP status). Reuses the
//     existing classifiers where they exist
//     (providers.ProviderError / providers.IsRateLimitError) and only
//     falls back to the executor's own stable string shapes
//     ("upstream returned error: %d …", "failed to resolve internal
//     config for model …", "buffer limit exceeded: …") for the
//     external-path errors that never materialize as ProviderError.
//
//   - sanitizeReasonDetail — single-line flattening + rune-safe
//     truncation at maxReasonDetailRunes + best-effort secret
//     redaction. Every free-text fragment classified BY THIS FILE
//     goes through it (see the scope note below) — raw err.Error()
//     emissions elsewhere on the request path are out of scope, so
//     this is a scoped guarantee for this file's output, not a
//     log-wide one.
//
// Design constraints honored here:
//
//   - Event names and the observable event shape are UNCHANGED. The
//     enrichment is additive: payloads GAIN `reason` / `reason_class`
//     / `reason_status`; existing keys keep their meaning
//     (`trigger_reason` carries the same detail it always carried,
//     now bounded + redacted).
//
//   - Redaction is BEST-EFFORT defense-in-depth, NOT a guarantee. The
//     reasonSecret* patterns strip a FIXED set of high-confidence
//     credential shapes only: sk- API keys, Bearer / Basic
//     Authorization values, api_key / api-key / x-api-key /
//     access_token / refresh_token assignments, GitHub
//     ghp_/gho_/ghu_/ghs_/ghr_ tokens, and Google AIza… keys. It
//     covers the `detail` / `error` fragments routed through
//     sanitizeReasonDetail and NOTHING ELSE — unrecognized key shapes
//     pass through, and the raw error-text emissions on the wider
//     request path (e.g. handler.go:890/898/906/910,
//     handler_anthropic.go:788-818,
//     race_coordinator.go:1094/1242/1355/1437, race_executor.go:1013)
//     are out of scope for this file. Broadening the assignment
//     alternation to bare \bkey\b / \btoken\b is deliberately deferred
//     (over-redaction would eat useful error detail) — see the defect
//     ledger note at the redaction pattern block below.
// ─────────────────────────────────────────────────────────────────────────────

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/disillusioners/llm-supervisor-proxy/pkg/providers"
)

// Stable classification tokens. These are the `reason_class` values
// that land in event payloads and the `reason=<class>` segment of log
// lines — treat them as a public vocabulary (grep-able, FE-renderable).
const (
	reasonClassRateLimit     = "rate_limit"     // 429 or rate-limit error-body vocabulary
	reasonClassAuth          = "auth"           // 401 / 403
	reasonClassConflict      = "conflict"       // 409
	reasonClassUpstream4xx   = "upstream_4xx"   // other 4xx (status emitted alongside)
	reasonClassUpstream5xx   = "upstream_5xx"   // 5xx (status emitted alongside)
	reasonClassTimeout       = "timeout"        // deadlines, idle timeouts, net timeouts
	reasonClassConnection    = "connection"     // dial/DNS/reset/closed-connection errors
	reasonClassProxyInternal = "proxy_internal" // proxy-side: resolution, no-credential, provider build, buffer cap
	reasonClassCancelled     = "cancelled"      // context cancelled (race teardown — loser of the race)
	reasonClassUnknown       = "unknown"        // nothing matched

	// reasonClassPairingRejection (upstream pairing-rejection guard) —
	// the upstream returned the tool-call pairing rejection AS a 200
	// completion; the guard poisoned the attempt and re-drives the
	// identical request. See pairing_guard.go.
	reasonClassPairingRejection = "upstream_pairing_rejection"
)

// maxReasonDetailRunes caps the free-text `detail` segment. 160 was
// chosen because: (a) the full reason segment — `reason=` + class
// (≤13 chars) + ` status=` + ≤3 digits + ` detail="…"` — stays under
// ~200 chars, inside a single terminal/UI log line; (b) it preserves
// the head of a typical upstream error message or status line, which
// is all triage needs (the full body stays in the request detail
// store, not the event log); (c) truncation is RUNE-safe so a
// multi-byte upstream body cannot split mid-character. Enforced on
// every emission path — no caller can leak an unbounded line.
const maxReasonDetailRunes = 160

// basicCredentialMinRun is the length floor for the Basic-auth
// credential run: a real base64 credential ("user:pass" encoded) is
// never shorter than 8 alphabet chars, while digit-bearing prose
// fragments ("basic v2", "basic 401 challenge") always are. Enforced
// on the captured run in sanitizeReasonDetail (RE2 cannot express
// "≥8 chars AND contains a digit" in one pattern without an
// enumeration of the digit position).
const basicCredentialMinRun = 8

// failureReason is the classified reason for one race/fallback step.
// Status is 0 when no HTTP status is attributable (timeouts,
// connection errors, proxy-side failures). Detail is already
// sanitized and bounded — never store raw upstream text here.
type failureReason struct {
	Class  string
	Status int
	Detail string
}

// String renders the stable one-line form, e.g.
//
//	reason=upstream_5xx status=502 detail="upstream returned error: 502 Bad Gateway"
//
// Field order is fixed (class, status, detail) so lines are greppable
// and diff-able across runs.
func (f failureReason) String() string {
	var b strings.Builder
	b.WriteString("reason=")
	if f.Class == "" {
		b.WriteString(reasonClassUnknown)
	} else {
		b.WriteString(f.Class)
	}
	if f.Status > 0 {
		fmt.Fprintf(&b, " status=%d", f.Status)
	}
	if f.Detail != "" {
		fmt.Fprintf(&b, " detail=%q", f.Detail)
	}
	return b.String()
}

// appendReasonFields adds the ADDITIVE classified-reason payload
// fields to an event/row map. The key sequence is the fixed
// convention shared by every classified emission site (race_spawn,
// race_all_failed per-row, race_request_failed) and mirrors
// failureReason.String()'s segment order:
//
//	reason → reason_class → reason_status (ONLY when Status > 0)
//
// The order is source-level (Go maps are unordered on the wire) but
// it is greppable and MUST NOT drift — this helper is now the single
// place that defines it. Callers keep their own gating (e.g. the
// race_spawn site publishes nothing extra for idle-timeout spawns).
func appendReasonFields(m map[string]interface{}, fr failureReason) {
	m["reason"] = fr.String()
	m["reason_class"] = fr.Class
	if fr.Status > 0 {
		m["reason_status"] = fr.Status
	}
}

// classifyAttemptFailure classifies one failed attempt from its error
// and the HTTP status stored on the attempt (0 when none). Pure
// function — safe from any goroutine.
//
// Precedence (matches the repo's own classifier semantics where they
// exist):
//  1. context cancellation (race teardown — not a real upstream fault)
//  2. rate-limit vocabulary (IsRateLimitError sees 503+rate-limit-body
//     as rate limit, per the Round-3b matrix — kept)
//  3. HTTP status family (from stored status, ProviderError, or the
//     executor's "upstream returned error: %d" string shape)
//  4. timeout family (net.Error.Timeout, DeadlineExceeded, idle-timeout strings)
//  5. connection family (dial/DNS/reset/closed-connection shapes)
//  6. proxy-internal shapes (resolution, provider build, buffer cap)
//  7. unknown
func classifyAttemptFailure(err error, storedStatus int) failureReason {
	if err == nil {
		return failureReason{Class: reasonClassUnknown}
	}

	fr := failureReason{Detail: sanitizeReasonDetail(err.Error())}

	// 1. Race teardown cancellation — the attempt lost the race, its
	// context was cancelled by cancelAllExcept. Not an upstream fault.
	if errors.Is(err, context.Canceled) {
		fr.Class = reasonClassCancelled
		return fr
	}

	// 1b. Upstream pairing-rejection guard sentinel (pairing_guard.go) —
	// the upstream answered 200 but the completion body is the anchored
	// tool-call pairing-rejection signature text. Classified BEFORE the
	// status families: the stored HTTP status is 200 (success-shaped) and
	// must not route this to reasonClassUnknown, and there is no
	// attributable HTTP error status (Status stays 0). Public vocabulary:
	// `reason_class=upstream_pairing_rejection` greps both the poisoned
	// attempt's race_request_failed payload and the pairing_retry
	// re-drive's race_spawn payload.
	if errors.Is(err, errUpstreamPairingRejection) {
		fr.Class = reasonClassPairingRejection
		return fr
	}

	// Status resolution happens BEFORE class dispatch so every branch
	// (rate limit included) can carry the concrete HTTP status: stored
	// attempt status wins, then ProviderError, then the external
	// executor's "upstream returned error: %d" string shape.
	status := storedStatus
	if status == 0 {
		var pe *providers.ProviderError
		if errors.As(err, &pe) && pe != nil {
			status = pe.StatusCode
		}
	}
	if status == 0 {
		status = parseUpstreamReturnedStatus(err.Error())
	}
	fr.Status = status

	// 2. Rate-limit vocabulary (Round-3b semantics: 429 OR rate-limit
	// error-body code/type, including the 503+rate-limit-body row).
	if providers.IsRateLimitError(err) {
		fr.Class = reasonClassRateLimit
		return fr
	}

	// 3. HTTP status family — dispatch on the status resolved above
	// (stored attempt status → ProviderError → executor string shape).
	if status > 0 {
		switch {
		case status == http.StatusTooManyRequests:
			fr.Class = reasonClassRateLimit
		case status == http.StatusUnauthorized || status == http.StatusForbidden:
			fr.Class = reasonClassAuth
		case status == http.StatusConflict:
			fr.Class = reasonClassConflict
		case status >= 500:
			fr.Class = reasonClassUpstream5xx
		case status >= 400:
			fr.Class = reasonClassUpstream4xx
		default:
			fr.Class = reasonClassUnknown
		}
		return fr
	}

	// 4. Timeout family.
	if classifyIsTimeout(err) {
		fr.Class = reasonClassTimeout
		return fr
	}

	// 5. Connection family.
	if classifyIsConnection(err) {
		fr.Class = reasonClassConnection
		return fr
	}

	// 6. Proxy-internal shapes (executor's own stable error strings —
	// these are proxy-side failures, not upstream faults).
	if classifyIsProxyInternal(err.Error()) {
		fr.Class = reasonClassProxyInternal
		return fr
	}

	// 7. Unknown — class token still emitted, detail still bounded.
	fr.Class = reasonClassUnknown
	return fr
}

// classifyRequestFailure classifies a failed upstreamRequest using its
// stored error and HTTP status. Zero-value safe: a request with no
// error classifies as unknown with no detail.
func classifyRequestFailure(req *upstreamRequest) failureReason {
	if req == nil {
		return failureReason{Class: reasonClassUnknown}
	}
	return classifyAttemptFailure(req.GetError(), req.GetHTTPStatus())
}

// parseUpstreamReturnedStatus extracts the numeric status from the
// external executor's error shape "upstream returned error: <code>
// <reason>" (with or without the context-overflow body suffix).
func parseUpstreamReturnedStatus(errStr string) int {
	const prefix = "upstream returned error: "
	i := strings.Index(errStr, prefix)
	if i < 0 {
		return 0
	}
	// strconv.Atoi requires the ENTIRE string to be numeric, so the
	// leading digit run is extracted first (the shape carries a
	// trailing reason phrase: "502 Bad Gateway").
	rest := errStr[i+len(prefix):]
	end := 0
	for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
		end++
	}
	if end == 0 {
		return 0
	}
	// Parse failure (e.g. overflow) degrades to 0 = "no attributable
	// status".
	code, err := strconv.Atoi(rest[:end])
	if err != nil {
		return 0
	}
	return code
}

// classifyIsTimeout reports whether the error is deadline/timeout
// shaped. Covers context deadlines, net.Error timeouts, and the
// proxy's own idle-timeout / stream-deadline strings.
func classifyIsTimeout(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr != nil && netErr.Timeout() {
		return true
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "timeout") ||
		strings.Contains(s, "timed out") ||
		strings.Contains(s, "deadline")
}

// classifyIsConnection reports whether the error is a transport-level
// connection failure: dial/DNS failures, refused/reset connections,
// premature upstream close, mid-stream read errors.
//
// Deferral rule: when the message matches a proxy-internal marker,
// defer to the proxy-internal cascade. Material for the
// race_executor.go:478 `url.Parse(cfg.UpstreamURL)` failure wrapped as
// "invalid upstream URL: %w" — a CONFIG defect (proxy-internal), not a
// transport fault. Without this deferral, the *url.Error / *net.OpError
// type checks would misroute every config-malformed URL to
// `connection` and the proxy-internal marker fix couldn't take effect.
func classifyIsConnection(err error) bool {
	if err == nil {
		return false
	}
	// Top-of-function deferral: a proxy-internal-shaped error is by
	// definition NOT a transport fault — short-circuit before the
	// type-based checks so the cascade in classifyAttemptFailure can
	// route it correctly.
	if classifyIsProxyInternal(err.Error()) {
		return false
	}
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		// url.Error.Timeout()/Temporary cover the timeout half; anything
		// else surfacing as url.Error is a dial/transport failure.
		if !urlErr.Timeout() {
			return true
		}
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr != nil && !netErr.Timeout() {
		return true
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		return true
	}
	s := strings.ToLower(err.Error())
	for _, marker := range []string{
		"connection refused",
		"connection reset",
		"broken pipe",
		"no such host",
		"dial tcp",
		"dial unix",
		"closed connection", // "upstream closed connection prematurely"
		"read error:",       // mid-stream read failure
		"eof",
	} {
		if strings.Contains(s, marker) {
			return true
		}
	}
	return false
}

// classifyIsProxyInternal reports whether the error string is one of
// the executor/coordinator's own proxy-side failure shapes (config
// resolution, credential absence, provider construction, buffer cap).
//
// Matching is case-INSENSITIVE: the haystack is lowercased first,
// mirroring classifyIsTimeout / classifyIsConnection. The producers
// emit MIXED-CASE literals — "invalid upstream URL"
// (race_executor.go:478) and "failed to create upstream request"
// (:511) — so the previous case-sensitive compare never matched them
// and those failures degraded to reasonClassUnknown.
func classifyIsProxyInternal(errStr string) bool {
	s := strings.ToLower(errStr)
	for _, marker := range []string{
		"failed to resolve internal config", // model missing or no usable credentials
		"failed to create provider",         // provider factory failure
		"failed to parse request body",      // client body (proxy-side handling)
		"failed to convert request",         // internal translation failure
		"failed to create upstream request", // external request construction (race_executor.go:511)
		"invalid upstream url",              // upstream URL configuration (race_executor.go:478)
		"failed to marshal response",        // proxy-side marshaling
		"buffer limit exceeded",             // RaceMaxBufferBytes cap
		"streaming not supported",           // provider capability gap
		"credential not found",              // dangling credential reference
	} {
		if strings.Contains(s, marker) {
			return true
		}
	}
	return false
}

// Precompiled redaction patterns. Applied in order; each replaces the
// secret-bearing span with a fixed *** marker so no key/token value
// can ever reach the event log.
//
// SCOPE / residual risk: this is a fixed high-confidence shape list,
// not a general secret detector. Broadening the assignment
// alternation to bare `\bkey\b` / `\btoken\b` was deliberately DEFERRED
// (2026-10-18): it would swallow ordinary error vocabulary such as
// "token budget exhausted" / "key not found", destroying the triage
// value of the detail this file exists to preserve. Unrecognized key
// shapes therefore survive, and raw error text emitted OUTSIDE
// sanitizeReasonDetail (see the file header) is not covered at all.
var (
	// sk- prefixed API keys (OpenAI-style; covers sk-proj-… etc).
	reasonSecretSK = regexp.MustCompile(`sk-[A-Za-z0-9][A-Za-z0-9_-]{4,}`)
	// Bearer tokens in Authorization headers echoed by upstreams.
	reasonSecretBearer = regexp.MustCompile(`(?i:bearer)\s+[A-Za-z0-9._~+/=-]{8,}`)
	// api_key / api-key / apikey / x-api-key / token assignment shapes
	// in bodies and echoed headers.
	reasonSecretAssign = regexp.MustCompile(`(?i:"?[x-]?api[_-]?key"?|"?access[_-]?token"?|"?refresh[_-]?token"?)["']?\s*[:=]\s*["']?[A-Za-z0-9._~+/=-]{6,}`)
	// GitHub tokens: ghp_ (PAT), gho_ (OAuth), ghu_ (user-to-server),
	// ghs_ (server-to-server), ghr_ (refresh) followed by ≥20 base62
	// chars. Genuine tokens carry 36+; the floor keeps prose fragments
	// from half-matching.
	reasonSecretGitHub = regexp.MustCompile(`\bgh[pousr]_[A-Za-z0-9]{20,}`)
	// Google API keys: the literal "AIza" prefix + exactly 35 key chars
	// (39 total). The trailing \b avoids emitting a partially-redacted
	// key prefix if the shape is not a full key.
	reasonSecretGoogle = regexp.MustCompile(`\bAIza[A-Za-z0-9_-]{35}\b`)
	// HTTP Basic credentials echoed in Authorization headers:
	// "Basic <base64>". Two guards keep prose out: (a) the credential
	// run must span ≥8 credential-alphabet chars (real base64 creds
	// are never shorter) — enforced at replace time via the capture
	// group + basicCredentialMinRun, because RE2 has no lookahead;
	// (b) the run must carry at least one digit or base64 symbol
	// (+ / =), so ordinary vocabulary after the word "Basic" (e.g.
	// "basic authentication") is NOT eaten. Together they also keep
	// short digit-bearing fragments ("basic v2", "basic 401
	// challenge") intact.
	reasonSecretBasic = regexp.MustCompile(`(?i:\bbasic)\s+([A-Za-z0-9+/=]*[0-9+/=][A-Za-z0-9+/=]*)`)
)

// sanitizeReasonDetail makes an upstream error string safe for the
// event log: flattens it to a single line, redacts credential-shaped
// material, and truncates to maxReasonDetailRunes (rune-safe, fixed
// "…" suffix). Deterministic: same input ⇒ same output, always.
//
// Redaction is best-effort (see the pattern block above for the exact
// covered shapes and the deferred broad-match note).
func sanitizeReasonDetail(s string) string {
	if s == "" {
		return ""
	}
	// Single-line flattening first so regex windows can't straddle a
	// newline and so the emitted reason can never break the log line.
	s = strings.Join(strings.Fields(s), " ")

	// Each pattern collapses to a marker that says WHAT was redacted —
	// never the value. Applied on flattened text, in order.
	s = reasonSecretSK.ReplaceAllString(s, `sk-***`)
	s = reasonSecretBearer.ReplaceAllString(s, `Bearer ***`)
	s = reasonSecretAssign.ReplaceAllString(s, `api_key=***`)
	s = reasonSecretGitHub.ReplaceAllString(s, `gh_***`)
	s = reasonSecretGoogle.ReplaceAllString(s, `AIza***`)
	// Basic: the digit/symbol guard alone still half-matches short
	// digit-bearing prose ("basic v2"), so the captured credential run
	// must ALSO clear basicCredentialMinRun before it is redacted;
	// runs below the floor survive untouched.
	s = reasonSecretBasic.ReplaceAllStringFunc(s, func(m string) string {
		if sub := reasonSecretBasic.FindStringSubmatch(m); len(sub[1]) >= basicCredentialMinRun {
			return `Basic ***`
		}
		return m
	})

	return truncateReasonRunes(s, maxReasonDetailRunes)
}

// truncateReasonRunes cuts s to at most max runes, appending "…" when
// truncated. Rune-safe (never splits a multi-byte character).
//
// Contract: a non-positive max is a NO-OP (s returned unchanged, no
// suffix). The sole call site (sanitizeReasonDetail) passes the
// maxReasonDetailRunes constant (160), so the branch is defensive.
func truncateReasonRunes(s string, max int) string {
	if max <= 0 || utf8.RuneCountInString(s) <= max {
		return s
	}
	runes := []rune(s)
	return string(runes[:max]) + "…"
}
