package proxy

// pairing_guard_test.go — upstream tool-call pairing-rejection guard
// (pairing_guard.go + the manage-loop hook in race_coordinator.go).
//
// Coverage:
//   1. Detector: both signature variants, near-miss negatives
//      ("[Error] Unrelated"), embedded-in-longer-text negatives, pending
//      (proper-prefix) holds, buffer shapes (SSE split across frames,
//      non-stream single JSON, reasoning-then-signature, >threshold
//      probe path).
//   2. Retry path: identical rawBody re-sent (byte-equal — no id
//      mutation), cap enforced, pairing_retry trigger vocabulary.
//   3. Exhaustion path: real upstream error surfaced via
//      GetFinalErrorInfo (502 / upstream_error / unavailable), NEVER
//      the signature text as a successful completion.
//   4. Live-mode (first-byte gate): poisoned before promotion, pending
//      hold releases on divergence, normal content unaffected.
//   5. Deadline-pick guard: match/pending candidates are never picked.
//
// No live upstream calls — every test drives an httptest mock.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// 1. Detector unit tests
// ─────────────────────────────────────────────────────────────────────────────

func TestPairingGuardVerdictForContent(t *testing.T) {
	sig1Full := pairingSigNoToolCallForOutput + "call_abc123"
	sig2Full := pairingSigNoToolOutputForCall + "abc123-def456"

	cases := []struct {
		name      string
		content   string
		completed bool
		want      pairingGuardVerdict
	}{
		{"sig1 variant 1 full", sig1Full, true, pairingGuardMatch},
		{"sig1 with trailing text", sig1Full + " Please retry.", true, pairingGuardMatch},
		{"sig2 variant 2 full", sig2Full, true, pairingGuardMatch},
		{"sig2 exact prefix boundary", pairingSigNoToolOutputForCall, true, pairingGuardMatch},
		// Near-miss negatives — the pattern must stay NARROW.
		{"near miss unrelated", "[Error] Unrelated", true, pairingGuardNone},
		{"near miss other error", "[Error] No tool call found somewhere else", true, pairingGuardNone},
		{"bare error token", "[Error]", true, pairingGuardNone},
		{"embedded in longer text", "Here is your answer. [Error] No tool call found for function call output with call_id x", true, pairingGuardNone},
		{"prefix mid-text", "Sure! [Error] No tool call found", true, pairingGuardNone},
		{"empty content", "", true, pairingGuardNone},
		// Pending: streaming, non-empty proper prefix.
		{"sig1 partial prefix", "[Error] No too", false, pairingGuardPending},
		{"sig1 shared head", pairingGuardProbe, false, pairingGuardPending},
		{"sig2 partial prefix", "[Error] No tool output found for function call ca", false, pairingGuardPending},
		// Pending resolves at completion: a completed body shorter than a
		// signature can never grow into a match.
		{"sig1 partial prefix completed", "[Error] No too", true, pairingGuardNone},
		// Divergence past the shared head with room to grow: still none.
		{"sig1 diverged", "[Error] No tool call found for function call input with call_id x", false, pairingGuardNone},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := pairingGuardVerdictForContent(tc.content, tc.completed)
			if got != tc.want {
				t.Fatalf("pairingGuardVerdictForContent(%q, completed=%v) = %v, want %v",
					tc.content, tc.completed, got, tc.want)
			}
		})
	}
}

// sseFrame builds one executor-shaped "data: {chunk}" frame.
func sseFrame(content string) []byte {
	return []byte("data: " + makeOpenAITextChunk(content) + "\n")
}

func newTestBuffer() *streamBuffer {
	return newStreamBuffer(1024 * 1024)
}

func TestPairingGuardVerdictForBuffer_Shapes(t *testing.T) {
	sigText := pairingSigNoToolCallForOutput + "call_abc123"

	t.Run("streaming signature split across two frames", func(t *testing.T) {
		buf := newTestBuffer()
		buf.Add(sseFrame("[Error] No tool call found for function call out"))
		buf.Add(sseFrame("put with call_id call_abc123"))
		buf.Close(nil)
		if got := pairingGuardVerdictForBuffer(buf); got != pairingGuardMatch {
			t.Fatalf("split signature = %v, want match", got)
		}
	})

	t.Run("non-stream single JSON body", func(t *testing.T) {
		resp := map[string]interface{}{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"choices": []interface{}{map[string]interface{}{"index": 0, "message": map[string]interface{}{"role": "assistant", "content": sigText}, "finish_reason": "stop"}},
		}
		b, _ := json.Marshal(resp)
		buf := newTestBuffer()
		if !buf.Add(b) {
			t.Fatalf("buffer add failed")
		}
		buf.Close(nil)
		if got := pairingGuardVerdictForBuffer(buf); got != pairingGuardMatch {
			t.Fatalf("non-stream signature body = %v, want match", got)
		}
	})

	t.Run("normal content buffer", func(t *testing.T) {
		buf := newTestBuffer()
		buf.Add(sseFrame("The weather is sunny."))
		buf.Add(sseFrame(" Have a nice day!"))
		buf.Close(nil)
		if got := pairingGuardVerdictForBuffer(buf); got != pairingGuardNone {
			t.Fatalf("normal content = %v, want none", got)
		}
	})

	t.Run("tool-call chunk buffer", func(t *testing.T) {
		chunk := map[string]interface{}{
			"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1700000000, "model": "test-upstream-model",
			"choices": []interface{}{map[string]interface{}{"index": 0, "delta": map[string]interface{}{
				"tool_calls": []interface{}{map[string]interface{}{
					"index": 0, "id": "call_abc123", "type": "function",
					"function": map[string]interface{}{"name": "get_weather", "arguments": "{}"},
				}},
			}}},
		}
		b, _ := json.Marshal(chunk)
		buf := newTestBuffer()
		buf.Add([]byte("data: " + string(b) + "\n"))
		buf.Close(nil)
		if got := pairingGuardVerdictForBuffer(buf); got != pairingGuardNone {
			t.Fatalf("tool-call buffer = %v, want none", got)
		}
	})

	t.Run("reasoning frames then signature content", func(t *testing.T) {
		think := map[string]interface{}{
			"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1700000000, "model": "test-upstream-model",
			"choices": []interface{}{map[string]interface{}{"index": 0, "delta": map[string]interface{}{"reasoning_content": "Let me think about this..."}}},
		}
		tb, _ := json.Marshal(think)
		buf := newTestBuffer()
		buf.Add([]byte("data: " + string(tb) + "\n"))
		buf.Add(sseFrame(sigText))
		buf.Close(nil)
		if got := pairingGuardVerdictForBuffer(buf); got != pairingGuardMatch {
			t.Fatalf("reasoning-then-signature = %v, want match", got)
		}
	})

	t.Run("large healthy buffer short-circuits without parse", func(t *testing.T) {
		buf := newTestBuffer()
		pad := strings.Repeat("x", pairingGuardParseThreshold) // > threshold after framing
		buf.Add(sseFrame(pad))
		buf.Add(sseFrame("more text"))
		if buf.TotalLen() <= pairingGuardParseThreshold {
			t.Fatalf("fixture setup: buffer not above threshold (%d)", buf.TotalLen())
		}
		if got := pairingGuardVerdictForBuffer(buf); got != pairingGuardNone {
			t.Fatalf("large healthy buffer = %v, want none", got)
		}
	})

	t.Run("large buffer with probe hit still parses", func(t *testing.T) {
		buf := newTestBuffer()
		pad := strings.Repeat("x", pairingGuardParseThreshold)
		// Signature FIRST (anchored at content position 0), padding after.
		// The probe hits in a later chunk only if it scans forward — but
		// the match decision is on the accumulated content, which starts
		// with the signature.
		buf.Add(sseFrame(sigText))
		buf.Add(sseFrame(pad))
		buf.Close(nil)
		if got := pairingGuardVerdictForBuffer(buf); got != pairingGuardMatch {
			t.Fatalf("large buffer with anchored signature = %v, want match", got)
		}
	})

	t.Run("large buffer with mid-text signature is not anchored", func(t *testing.T) {
		// The probe hits (signature bytes present somewhere), but the
		// accumulated content does not START with it → none. Anchoring is
		// load-bearing even on the parse path.
		buf := newTestBuffer()
		pad := strings.Repeat("x", pairingGuardParseThreshold)
		buf.Add(sseFrame(pad))
		buf.Add(sseFrame(sigText))
		buf.Close(nil)
		if got := pairingGuardVerdictForBuffer(buf); got != pairingGuardNone {
			t.Fatalf("large buffer with mid-text signature = %v, want none", got)
		}
	})

	t.Run("nil and empty buffers", func(t *testing.T) {
		if got := pairingGuardVerdictForBuffer(nil); got != pairingGuardNone {
			t.Fatalf("nil buffer = %v, want none", got)
		}
		buf := newTestBuffer()
		if got := pairingGuardVerdictForBuffer(buf); got != pairingGuardNone {
			t.Fatalf("empty buffer = %v, want none", got)
		}
	})

	t.Run("pending on open buffer", func(t *testing.T) {
		buf := newTestBuffer()
		buf.Add(sseFrame("[Error] No too"))
		if got := pairingGuardVerdictForBuffer(buf); got != pairingGuardPending {
			t.Fatalf("open partial buffer = %v, want pending", got)
		}
	})
}

// ─────────────────────────────────────────────────────────────────────────────
// Coordinator-level tests (httptest mock upstream — no live calls)
// ─────────────────────────────────────────────────────────────────────────────

// pairingFixtureBody carries a realistic assistant-tool-call → tool-result
// pair so the "identical rawBody resent" assertion exercises the actual
// pairing payload shape.
const pairingFixtureBody = `{"model":"test-model","stream":true,"messages":[` +
	`{"role":"user","content":"what is the weather in paris"},` +
	`{"role":"assistant","content":"","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"paris\"}"}}]},` +
	`{"role":"tool","tool_call_id":"call_abc123","content":"sunny, 21C"}]}`

// capturedUpstream records every upstream call's body for resend-identity
// assertions.
type capturedUpstream struct {
	mu     sync.Mutex
	bodies [][]byte
}

func (c *capturedUpstream) snapshot() [][]byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	bodies := make([][]byte, len(c.bodies))
	copy(bodies, c.bodies)
	return bodies
}

func (c *capturedUpstream) append(b []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	cp := make([]byte, len(b))
	copy(cp, b)
	c.bodies = append(c.bodies, cp)
}

func pairingRaceConfig(t *testing.T, serverURL string, idleTimeout time.Duration) *ConfigSnapshot {
	t.Helper()
	return &ConfigSnapshot{
		RaceRetryEnabled:   true,
		RaceMaxParallel:    4,
		RaceMaxBufferBytes: 1024 * 1024,
		IdleTimeout:        idleTimeout,
		StreamDeadline:     5 * time.Second,
		MaxGenerationTime:  5 * time.Second,
		ModelsConfig:       seededModelsConfig(t, serverURL),
		ModelID:            "test-model",
	}
}

func newPairingRaceRequest(t *testing.T) (*http.Request, []byte) {
	t.Helper()
	req, _ := http.NewRequest("POST", "/v1/chat/completions", strings.NewReader(pairingFixtureBody))
	return req, []byte(pairingFixtureBody)
}

func TestPairingRetry_ResendsIdenticalBody(t *testing.T) {
	captured := &capturedUpstream{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured.append(body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if len(captured.snapshot()) == 1 {
			// Attempt 1: upstream answers 200 but "completes" with the
			// pairing-rejection text as normal content.
			fmt.Fprintf(w, "data: %s\n\n", makeOpenAITextChunk(pairingSigNoToolCallForOutput+"call_abc123"))
			w.Write([]byte("data: [DONE]\n\n"))
			return
		}
		// Attempt 2 (the re-drive): heals — normal content.
		fmt.Fprintf(w, "data: %s\n\n", makeOpenAITextChunk("The weather in Paris is sunny."))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, rawBody := newPairingRaceRequest(t)
	coordinator := newRaceCoordinator(ctx, pairingRaceConfig(t, server.URL, time.Second), req, rawBody, []string{"test-model"}, false)
	coordinator.Start()

	winner := coordinator.WaitForWinner()
	if winner == nil {
		t.Fatalf("no winner — guard retry path failed to heal")
	}
	if winner.id == 0 {
		t.Fatalf("poisoned attempt 0 won — guard did not gate promotion")
	}

	// The re-drive resent the IDENTICAL request — attempt-to-attempt byte
	// equality of the upstream bodies (same deterministic provider
	// conversion, no per-attempt divergence), the tool-call ids carried
	// through VERBATIM (never rewritten: the upstream mints ids itself),
	// and the coordinator's rawBody still stored verbatim.
	//
	// NOTE: the upstream body is not byte-equal to rawBody by design —
	// every attempt (including the poisoned one) goes through the same
	// rawBody → bodyMap → provider conversion (internal model-name swap,
	// field ordering). Identity that matters: both attempts received the
	// SAME converted body.
	bodies := captured.snapshot()
	if len(bodies) != 2 {
		t.Fatalf("expected exactly 2 upstream calls (poisoned + re-drive), got %d", len(bodies))
	}
	if string(bodies[0]) != string(bodies[1]) {
		t.Fatalf("re-drive upstream body differs from the poisoned attempt's body:\nwant %q\ngot  %q", bodies[0], bodies[1])
	}
	if !strings.Contains(string(bodies[1]), `"id":"call_abc123"`) || !strings.Contains(string(bodies[1]), `"tool_call_id":"call_abc123"`) {
		t.Fatalf("re-drive lost or rewrote the tool-call ids: %q", bodies[1])
	}
	if string(coordinator.rawBody) != pairingFixtureBody {
		t.Fatalf("coordinator rawBody mutated in place: %q", coordinator.rawBody)
	}

	// The winner's relayable buffer must NOT contain the signature.
	raw := winner.GetBuffer().GetAllRawBytes()
	if strings.Contains(string(raw), "[Error] No tool call found") {
		t.Fatalf("winner buffer carries the rejection signature: %q", raw)
	}

	// Trigger vocabulary: the re-drive published as pairing_retry.
	found := false
	for _, trig := range coordinator.spawnTriggers {
		if trig.trigger == triggerPairingRetry {
			found = true
		}
	}
	if !found {
		t.Fatalf("no %s spawn trigger recorded: %+v", triggerPairingRetry, coordinator.spawnTriggers)
	}
}

func TestPairingRetry_CapAndExhaustion(t *testing.T) {
	var calls int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&calls, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "data: %s\n\n", makeOpenAITextChunk(pairingSigNoToolCallForOutput+"call_abc123"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	req, rawBody := newPairingRaceRequest(t)
	coordinator := newRaceCoordinator(ctx, pairingRaceConfig(t, server.URL, time.Second), req, rawBody, []string{"test-model"}, false)
	coordinator.Start()

	winner := coordinator.WaitForWinner()
	if winner != nil {
		t.Fatalf("exhausted guard must not surface a winner carrying the signature text")
	}

	// Cap: main attempt + exactly pairingRetryMaxAttempts re-drives.
	if got := atomic.LoadInt64(&calls); got != int64(1+pairingRetryMaxAttempts) {
		t.Fatalf("upstream calls = %d, want %d (1 main + %d re-drives)", got, 1+pairingRetryMaxAttempts, pairingRetryMaxAttempts)
	}

	// EXHAUSTION PATH: a REAL upstream error, never the signature text as
	// a success. 502 + upstream_error/unavailable so client-side error
	// classification engages.
	errInfo := coordinator.GetFinalErrorInfo()
	if errInfo.HTTPStatus != http.StatusBadGateway {
		t.Fatalf("exhaustion HTTPStatus = %d, want 502", errInfo.HTTPStatus)
	}
	if errInfo.ErrorType != "upstream_error" || errInfo.ErrorCode != "unavailable" {
		t.Fatalf("exhaustion error shape = %s/%s, want upstream_error/unavailable", errInfo.ErrorType, errInfo.ErrorCode)
	}
	if !strings.Contains(errInfo.Message, "pairing") {
		t.Fatalf("exhaustion message lost the pairing context: %q", errInfo.Message)
	}
	if strings.Contains(errInfo.Message, "[Error] No tool call found") {
		t.Fatalf("exhaustion message leaks the raw signature text: %q", errInfo.Message)
	}

	// Status resolution: the sentinel must route to 502 (its text avoids
	// the GetCommonFailureStatus fallback keywords by design).
	if got := coordinator.GetCommonFailureStatus(); got != http.StatusBadGateway {
		t.Fatalf("GetCommonFailureStatus = %d, want 502", got)
	}
}

func TestPairingGuard_LiveMode_SignatureFirstThenHeal(t *testing.T) {
	captured := &capturedUpstream{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured.append(body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if len(captured.snapshot()) == 1 {
			fmt.Fprintf(w, "data: %s\n\n", makeOpenAITextChunk(pairingSigNoToolCallForOutput+"call_abc123"))
			w.Write([]byte("data: [DONE]\n\n"))
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", makeOpenAITextChunk("Healed response."))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, rawBody := newPairingRaceRequest(t)
	// LIVE gate: winner eligibility is the first buffered byte.
	coordinator := newRaceCoordinatorWithEvents(ctx, pairingRaceConfig(t, server.URL, time.Second), req, rawBody, []string{"test-model"}, nil, "", false, nil, "", true, nil)
	coordinator.Start()

	winner := coordinator.WaitForWinner()
	if winner == nil || winner.id == 0 {
		t.Fatalf("live mode: expected the re-drive (id 1) to win, got %+v", winner)
	}
	if got := len(captured.snapshot()); got != 2 {
		t.Fatalf("upstream calls = %d, want 2", got)
	}
}

func TestPairingGuard_LiveMode_PendingHoldReleasesOnDivergence(t *testing.T) {
	var calls int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&calls, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// First delta: a proper prefix of the signature — the guard must
		// HOLD the promotion, not win and not poison.
		fmt.Fprintf(w, "data: %s\n\n", makeOpenAITextChunk("[Error] No too"))
		// Second delta DIVERGES from the signature → never a match; the
		// hold releases and the attempt wins normally.
		time.Sleep(150 * time.Millisecond) // force ≥1 manage tick into the pending state
		fmt.Fprintf(w, "data: %s\n\n", makeOpenAITextChunk("[Error] Nothing matched your filters."))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, rawBody := newPairingRaceRequest(t)
	coordinator := newRaceCoordinatorWithEvents(ctx, pairingRaceConfig(t, server.URL, time.Second), req, rawBody, []string{"test-model"}, nil, "", false, nil, "", true, nil)
	coordinator.Start()

	winner := coordinator.WaitForWinner()
	if winner == nil {
		t.Fatalf("diverging content must eventually win")
	}
	if winner.id != 0 {
		t.Fatalf("diverging content should win as the ORIGINAL attempt (id 0), got id %d", winner.id)
	}
	if got := atomic.LoadInt64(&calls); got != 1 {
		t.Fatalf("upstream calls = %d, want 1 (divergence must NOT trigger a re-drive)", got)
	}
}

func TestPairingGuard_LiveMode_NormalContentUnaffected(t *testing.T) {
	var calls int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&calls, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "data: %s\n\n", makeOpenAITextChunk("A perfectly normal answer."))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, rawBody := newPairingRaceRequest(t)
	coordinator := newRaceCoordinatorWithEvents(ctx, pairingRaceConfig(t, server.URL, time.Second), req, rawBody, []string{"test-model"}, nil, "", false, nil, "", true, nil)
	coordinator.Start()

	winner := coordinator.WaitForWinner()
	if winner == nil || winner.id != 0 {
		t.Fatalf("normal content must win as attempt 0, got %+v", winner)
	}
	if got := atomic.LoadInt64(&calls); got != 1 {
		t.Fatalf("upstream calls = %d, want 1 — guard must not disturb healthy traffic", got)
	}
	if len(coordinator.spawnTriggers) != 1 {
		t.Fatalf("spawnTriggers = %+v, want exactly the main spawn", coordinator.spawnTriggers)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Deadline-pick guard + classifier unit tests
// ─────────────────────────────────────────────────────────────────────────────

func TestHandleStreamingDeadline_SkipsGuardCandidates(t *testing.T) {
	cases := []struct {
		name    string
		frames  []string
		wantEvt string // expected log path: poisoned vs pending skip
	}{
		{
			name:    "match candidate poisoned, error envelope fires",
			frames:  []string{pairingSigNoToolCallForOutput + "call_abc123"},
			wantEvt: "poisoned",
		},
		{
			name:    "pending candidate skipped, error envelope fires",
			frames:  []string{"[Error] No too"},
			wantEvt: "pending",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := pairingRaceConfig(t, "http://unused.invalid", time.Second)
			c := newRaceCoordinator(context.Background(), cfg, nil, nil, []string{"test-model"}, false)
			req := newUpstreamRequest(0, modelTypeMain, "test-model", 1024*1024)
			for _, f := range tc.frames {
				req.GetBuffer().Add(sseFrame(f))
			}
			c.requests = append(c.requests, req)

			c.handleStreamingDeadline()

			if c.GetWinner() != nil {
				t.Fatalf("guard-bound candidate must never be deadline-picked")
			}
			if c.GetStreamDeadlineError() == nil {
				t.Fatalf("expected the no-content error envelope (streamDeadlineError) to fire")
			}
			if !req.IsPoisoned() && tc.wantEvt == "poisoned" {
				t.Fatalf("match candidate must be poisoned by the deadline path")
			}
			if req.IsPoisoned() && tc.wantEvt == "pending" {
				t.Fatalf("pending candidate must NOT be poisoned (not yet adjudicable)")
			}
		})
	}
}

func TestClassifyAttemptFailure_PairingSentinel(t *testing.T) {
	fr := classifyAttemptFailure(errUpstreamPairingRejection, 0)
	if fr.Class != reasonClassPairingRejection {
		t.Fatalf("class = %q, want %q", fr.Class, reasonClassPairingRejection)
	}
	if fr.Status != 0 {
		t.Fatalf("status = %d, want 0 (upstream answered 200 — no attributable error status)", fr.Status)
	}
	if fr.Detail == "" {
		t.Fatalf("detail must carry the bounded sentinel text")
	}
}

func TestPairingGuardSentinel_StatusResolution(t *testing.T) {
	// The sentinel wording avoids every GetCommonFailureStatus fallback
	// keyword ("upstream returned error: ", "idle timeout", "timeout",
	// "context", "buffer limit") so an all-poisoned race resolves to 502.
	coordinator := newRaceCoordinator(context.Background(), pairingRaceConfig(t, "http://unused.invalid", time.Second), nil, nil, []string{"test-model"}, false)
	for i := 0; i < 2; i++ {
		req := newUpstreamRequest(i, modelTypeMain, "test-model", 1024*1024)
		req.MarkFailed(errUpstreamPairingRejection)
		coordinator.requests = append(coordinator.requests, req)
	}
	if got := coordinator.GetCommonFailureStatus(); got != http.StatusBadGateway {
		t.Fatalf("GetCommonFailureStatus = %d, want 502", got)
	}
	errInfo := coordinator.GetFinalErrorInfo()
	if errInfo.HTTPStatus != http.StatusBadGateway {
		t.Fatalf("HTTPStatus = %d, want 502", errInfo.HTTPStatus)
	}
}
