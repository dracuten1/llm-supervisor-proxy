package proxy

import (
	"context"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/disillusioners/llm-supervisor-proxy/pkg/credentiallb"
	"github.com/disillusioners/llm-supervisor-proxy/pkg/models"
)

type upstreamModelType string

const (
	modelTypeMain         upstreamModelType = "main"
	modelTypeSecond       upstreamModelType = "second"
	modelTypeFallback     upstreamModelType = "fallback"
	modelTypeCredFailover upstreamModelType = "cred_failover" // Phase 3 / Task 18 (Round 3c canonical)
)

type upstreamStatus string

const (
	statusPending   upstreamStatus = "pending"
	statusRunning   upstreamStatus = "running"
	statusStreaming upstreamStatus = "streaming"
	statusCompleted upstreamStatus = "completed"
	statusFailed    upstreamStatus = "failed"
)

// TokenUsage represents token usage from an upstream response
type TokenUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// upstreamRequest represents a single attempt to an upstream provider
type upstreamRequest struct {
	mu sync.RWMutex

	id        int               // Sequence ID (0=main, 1=second, 2=fallback)
	modelType upstreamModelType // main, second, fallback
	modelID   string            // Specific model ID used

	ctx      context.Context    // Request-specific context
	cancel   context.CancelFunc // To cancel this request specifically
	buffer   *streamBuffer      // Buffer for response chunks
	resp     *http.Response     // Set when response received
	status   upstreamStatus     // Current status
	err      error              // Final error (if any)
	attempts int                // Number of HTTP retries (for connectivity)

	// Phase 3 / Tasks 4 + 6 — race-executor plumbing.
	// conversationKey is threaded from handler.go:401+ through the
	// race coordinator + executeRequest down to the resolver seam
	// (ResolveInternalConfigWithAffinity). required for
	// ResolvedCredential.CredentialID / .NewlyBound propagation.
	conversationKey string

	// resolved is the struct form (Phase 3 / #3) populated by
	// executeInternalRequest after the resolver call. .NewlyBound is
	// the W-1 sole signal for the model_credential_selected event.
	resolved   models.ResolvedCredential
	resolvedOK bool

	// engine + onCredentialSelected (Phase 3 / Task 6) are
	// captured by executeRequest and forwarded via this attempt
	// struct so the publish site (executeRequest after
	// executeInternalRequest returns) can fire the event.
	//
	// engine is the optional Load-Balancing engine (nil-safe per
	// W-3 / E-3); onCredentialSelected is the caller-supplied callback
	// that publishes model_credential_selected with the result struct.
	engine               *credentiallb.Engine
	onCredentialSelected func(modelID string, cred models.ResolvedCredential)

	// onMemberSelected (load-balance-groups / Q15) publishes
	// model_member_selected at the W-1 site when the group pick was
	// newly bound. Built by the coordinator alongside
	// onCredentialSelected; nil for non-group attempts.
	onMemberSelected func(memberID string, cred models.ResolvedCredential)

	// groupPick (load-balance-groups / Q15) is the request's group
	// rewrite record (nil for non-group attempts). Set at spawn for
	// the MAIN attempt; the W-1 publish site fires
	// model_member_selected when NewlyBound is true.
	groupPick *groupPickInfo

	// reselectedCredentialID (Phase 3 / R3-2/R3-3) — populated only
	// for modelTypeCredFailover rows from Round-3b ExcludeAndReselect:
	// the SPECIFIC reselected credential (not a fresh engine call).
	// Used by handleNonStreamResult / streamResult to skip the
	// secondary upstream-set call and use this credential directly.
	reselectedCredentialID string

	// deadlinePick (sticky last-success ordering) marks an attempt
	// promoted to winner by handleStreamingDeadline's best-buffer pick
	// — a BEST-PARTIAL artifact, not a delivered success. The sticky
	// record sites skip it (design §4.2: never record on
	// race_deadline_pick).
	//
	// Written under c.mu in (*raceCoordinator).handleStreamingDeadline
	// (its best-buffer promotion site), where the load-bearing publish
	// is the c.streamCh close in that same method — that is what wakes
	// WaitForWinner's streamCh arm; c.done is closed by the LATER
	// completion/all-failed path, not by the deadline pick. The write
	// happens-before the close and the reader takes c.mu in GetWinner,
	// so the handler observes it without a lock of its own.
	deadlinePick bool

	// poisoned (upstream pairing-rejection guard) — set ONLY by
	// MarkPairingRejected when the attempt's buffer carries the anchored
	// upstream pairing-rejection signature. Once set: the attempt can
	// never be promoted to winner (GetError() != nil excludes it from
	// every eligibility predicate) and execute() must not re-mark it
	// completed (the executor goroutine may still be draining its
	// upstream stream when the poison lands in live mode). Guarded by mu.
	poisoned bool

	// Cancellation state
	cancelled  bool      // Set when Cancel() has been called
	cancelOnce sync.Once // Ensures cleanup is only performed once

	// Timing
	startTime        time.Time
	firstByteTime    time.Time
	completionTime   time.Time
	lastActivityTime time.Time // Last time we received data (for idle detection)

	// Stats
	totalChunks int
	totalBytes  int64

	// Token usage (extracted from non-streaming responses)
	usage *TokenUsage

	// HTTP status code from upstream (0 if not an HTTP error)
	httpStatusCode int

	// Flag to use secondary upstream model for modelTypeSecond requests
	// Only applicable to internal models
	useSecondaryUpstream bool
}

func newUpstreamRequest(id int, mType upstreamModelType, modelID string, maxBuffer int) *upstreamRequest {
	return &upstreamRequest{
		id:        id,
		modelType: mType,
		modelID:   modelID,
		status:    statusPending,
		buffer:    newStreamBuffer(int64(maxBuffer)),
	}
}

// MarkStarted transitions status to running
func (r *upstreamRequest) MarkStarted() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.status = statusRunning
	r.startTime = time.Now()
}

func (r *upstreamRequest) MarkStreaming() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.status = statusStreaming
	r.firstByteTime = time.Now()
	r.lastActivityTime = time.Now() // Initialize activity time when streaming starts
}

func (r *upstreamRequest) MarkCompleted() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.status = statusCompleted
	r.completionTime = time.Now()
	// Signal buffer completion (only if buffer hasn't been released by Cancel())
	if r.buffer != nil {
		r.buffer.Close(nil)
	}
}

func (r *upstreamRequest) MarkFailed(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.status = statusFailed
	r.err = err
	r.completionTime = time.Now()
	// Signal buffer completion with error (only if buffer hasn't been released by Cancel())
	if r.buffer != nil {
		r.buffer.Close(err)
	}
}

// MarkPairingRejected poisons a winner-eligible attempt whose buffer
// carries the upstream tool-call pairing-rejection signature
// (pairing_guard.go). The attempt is flipped to failed with the
// errUpstreamPairingRejection sentinel BEFORE it can be promoted to
// winner or deadline-picked, so the signature text can never be relayed
// to the client as a successful completion.
//
// Semantics beyond MarkFailed:
//   - sets the poisoned flag — execute() checks it so a still-running
//     executor goroutine cannot re-MarkCompleted (live mode: the poison
//     lands on the FIRST buffered byte, while the executor is mid-stream)
//     and cannot overwrite the sentinel via a late MarkFailed;
//   - cancels the attempt context so the in-flight upstream read stops
//     promptly (mirrors Cancel's ordering: cancel outside r.mu);
//   - closes the buffer with the sentinel (CAS-guarded — if MarkCompleted
//     already closed it with nil, the close is a no-op and the sentinel
//     still lives in r.err).
//
// Idempotent + one-shot: returns false when the attempt was already
// poisoned or already carries an error (never re-poison a loser).
func (r *upstreamRequest) MarkPairingRejected(reason error) bool {
	r.mu.Lock()
	if r.poisoned || r.err != nil {
		r.mu.Unlock()
		return false
	}
	r.poisoned = true
	r.status = statusFailed
	r.err = reason
	r.completionTime = time.Now()
	cancel := r.cancel
	r.mu.Unlock()

	if cancel != nil {
		cancel()
	}

	if buf := r.GetBuffer(); buf != nil {
		buf.Close(reason)
	}
	return true
}

// IsPoisoned reports whether the pairing guard claimed this attempt
// (MarkPairingRejected). Lock-protected: manage() writes it while the
// executor goroutine reads it.
func (r *upstreamRequest) IsPoisoned() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.poisoned
}

func (r *upstreamRequest) SetContext(ctx context.Context, cancel context.CancelFunc) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ctx = ctx
	r.cancel = cancel
}

func (r *upstreamRequest) GetStatus() upstreamStatus {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.status
}

func (r *upstreamRequest) GetID() int {
	return r.id
}

func (r *upstreamRequest) GetModelType() upstreamModelType {
	return r.modelType
}

func (r *upstreamRequest) GetModelID() string {
	return r.modelID
}

// GetBuffer returns the current stream buffer, or nil after Cancel()
// released it. Callers must either nil-check the result or rely on the
// nil-safe streamBuffer methods. Lock-protected: buffer release (Cancel →
// cleanup) writes this field concurrently.
func (r *upstreamRequest) GetBuffer() *streamBuffer {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.buffer
}

func (r *upstreamRequest) GetError() error {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.err
}

func (r *upstreamRequest) IsDone() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.status == statusCompleted || r.status == statusFailed
}

// IsCompleted returns true only if the request completed successfully (received [DONE])
func (r *upstreamRequest) IsCompleted() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.status == statusCompleted
}

func (r *upstreamRequest) IsStreaming() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.status == statusStreaming
}

// IsCancelled returns true if the request has been cancelled
func (r *upstreamRequest) IsCancelled() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.cancelled
}

// Cancel safely cancels the request's context and performs full cleanup.
// This method is idempotent - calling it multiple times is safe.
// Cancel performs the following cleanup:
//   - Cancels the request context
//   - Drains and closes the HTTP response body
//   - Releases the stream buffer
func (r *upstreamRequest) Cancel() {
	// Use sync.Once to ensure cleanup only happens once
	r.cancelOnce.Do(func() {
		r.mu.Lock()
		r.cancelled = true
		cancel := r.cancel
		r.mu.Unlock()

		// Cancel the context first
		if cancel != nil {
			cancel()
		}

		// Perform full cleanup
		r.cleanup()
	})
}

// cleanup drains and closes the response body and releases the buffer.
// Called by Cancel() without the mutex held (after Cancel releases it) to avoid
// deadlock. sync.Once guarantees single-threaded access, so no additional
// synchronization is needed here.
func (r *upstreamRequest) cleanup() {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Drain and close response body if present
	if r.resp != nil && r.resp.Body != nil {
		// Drain any remaining data to prevent connection from being closed prematurely
		// This is important for HTTP connection reuse
		io.Copy(io.Discard, r.resp.Body)
		r.resp.Body.Close()
		r.resp = nil
	}

	// Release the stream buffer
	if r.buffer != nil {
		r.buffer.Close(context.Canceled)
		r.buffer = nil
	}
}

// TrackActivity updates the last activity time to now
// This is used by the executor to signal that data is still being received
func (r *upstreamRequest) TrackActivity() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lastActivityTime = time.Now()
}

// GetLastActivity returns the time of last activity
func (r *upstreamRequest) GetLastActivity() time.Time {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.lastActivityTime
}

// IsIdle returns true if the request has been idle for longer than the given duration
func (r *upstreamRequest) IsIdle(idleTimeout time.Duration) bool {
	r.mu.RLock()
	lastActivity := r.lastActivityTime
	status := r.status
	r.mu.RUnlock()

	// Only consider idle if we're streaming (have received at least some data)
	// and haven't received data for idleTimeout duration
	if status != statusStreaming {
		return false
	}
	return time.Since(lastActivity) > idleTimeout
}

// SetUsage sets the token usage for this request
func (r *upstreamRequest) SetUsage(usage *TokenUsage) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.usage = usage
}

// GetUsage returns the token usage for this request
func (r *upstreamRequest) GetUsage() *TokenUsage {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.usage
}

// SetResp sets the HTTP response received from upstream
func (r *upstreamRequest) SetResp(resp *http.Response) {
	r.mu.Lock()
	r.resp = resp
	r.mu.Unlock()
}

// SetHTTPStatus sets the HTTP status code from upstream
func (r *upstreamRequest) SetHTTPStatus(code int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.httpStatusCode = code
}

// GetHTTPStatus returns the HTTP status code
func (r *upstreamRequest) GetHTTPStatus() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.httpStatusCode
}

// SetUseSecondaryUpstream sets whether to use secondary upstream model
func (r *upstreamRequest) SetUseSecondaryUpstream(val bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.useSecondaryUpstream = val
}

// UseSecondaryUpstream returns whether to use secondary upstream model
func (r *upstreamRequest) UseSecondaryUpstream() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.useSecondaryUpstream
}
