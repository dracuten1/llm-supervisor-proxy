package proxy

// ─────────────────────────────────────────────────────────────────────────────
// pairing_guard.go — upstream tool-call pairing-rejection guard
//
// Problem this file solves: some agentic upstreams (Z.ai GLM coding
// endpoint, Antigravity/Gemini pool) validate tool-call pairing against
// their OWN server-side conversation state, not the delivered messages
// array. Stochastically (~3/day field rate) that state drops the
// declaration and the upstream rejects a pair whose call_id it itself
// minted — returning the rejection AS a normal 200-SSE completion
// (finish_reason=stop, no error frame, no usage block). Clients receive
// HTTP 200 text → no exception → no retry → the rejection persists as
// assistant content.
//
// Replaying the IDENTICAL request heals (eventually-consistent upstream
// state), so the guard:
//
//   1. DETECTS the anchored rejection signature in a completed /
//      completing attempt buffer (both verbatim variants below; prefix
//      match on the ACCUMULATED assistant text content only — never a
//      loose substring scan, so false positives are negligible).
//
//   2. POISONS the attempt (MarkPairingRejected → failed with the
//      errUpstreamPairingRejection sentinel) BEFORE it can be promoted
//      to winner, and the coordinator re-drives the request with the
//      IDENTICAL rawBody (race_coordinator.go manage loop; trigger
//      pairing_retry; cap pairingRetryMaxAttempts). Tool-call ids are
//      NEVER rewritten on resend — the raw body is re-sent verbatim and
//      the upstream mints its own ids.
//
//   3. EXHAUSTION: when the cap is spent and the latest completion still
//      carries the signature, the race resolves through the normal
//      all-failed machinery — the client receives a REAL upstream error
//      (502 / upstream_error / unavailable via GetFinalErrorInfo), never
//      the signature text as a successful completion.
//
// Where the hook lives: attempts are BUFFERED per-attempt (the executor
// adds SSE frames to the attempt's streamBuffer), but in live-streaming
// mode the manage loop promotes a winner on the FIRST buffered byte and
// the handler relays that buffer to the client incrementally. Detection
// therefore runs at winner-promotion time (manage loop ticker arm +
// handleStreamingDeadline), i.e. strictly BEFORE the first client byte —
// a poisoned attempt can never reach the relay.
// ─────────────────────────────────────────────────────────────────────────────

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
)

// Verbatim upstream rejection signatures. BOTH variants are matched as
// ANCHORED PREFIXES of the accumulated assistant text content. Keep this
// list narrow: "[Error]" alone or loose substrings must NOT match
// ("[Error] Unrelated" is a designed negative test).
const (
	pairingSigNoToolCallForOutput = "[Error] No tool call found for function call output with call_id "
	pairingSigNoToolOutputForCall = "[Error] No tool output found for function call call_"
)

// pairingGuardSignatures is the match table (both variants).
var pairingGuardSignatures = []string{
	pairingSigNoToolCallForOutput,
	pairingSigNoToolOutputForCall,
}

// pairingGuardProbe is the shortest stable shared head of both
// signatures. It is a CHEAP prefilter for large buffers only (see
// pairingGuardParseThreshold): if the probe bytes appear nowhere in the
// raw buffer, the anchored signature cannot be present either, and the
// (allocation-bearing) exact parse is skipped. The exact verdict always
// runs on the accumulated CONTENT — the probe never decides.
const pairingGuardProbe = "[Error] No tool"

// pairingGuardParseThreshold caps the no-parse fast path. Buffers at or
// below this size are always parsed exactly (a rejection completion is
// ~500 bytes, so every realistic candidate sits far below the
// threshold). Larger buffers only parse on a probe hit — except when
// completed, where the final adjudication must be exact.
const pairingGuardParseThreshold = 64 * 1024

// pairingRetryMaxAttempts caps the transparent re-drives per race: 2
// extra attempts beyond the poisoned one. The re-drives reuse the
// modelTypeCredFailover row shape (same-model re-drive machinery), so
// they are EXEMPT from the model-attempt accounting (B1) exactly like
// credential-failover rows and do NOT consume the model chain.
const pairingRetryMaxAttempts = 2

// errUpstreamPairingRejection is the sentinel error carried by a
// poisoned attempt. Wording constraint: the text must NOT contain
// "upstream returned error: ", "timeout", "context", or "buffer limit"
// — GetCommonFailureStatus's fallback parser keys on those fragments to
// pick the client status, and this sentinel must resolve to 502
// (StatusBadGateway → upstream_error/unavailable), which it does.
var errUpstreamPairingRejection = errors.New(
	"upstream rejected tool-call pairing (rejection text delivered as a 200 completion)")

// pairingGuardVerdict is the detector's outcome for one attempt buffer.
type pairingGuardVerdict int

const (
	// pairingGuardNone — no signature material (or the accumulated
	// content has diverged from every signature). The attempt is a
	// normal winner candidate.
	pairingGuardNone pairingGuardVerdict = iota

	// pairingGuardPending — the attempt is still streaming and its
	// accumulated content is a NON-EMPTY proper prefix of a signature
	// (e.g. the first delta carried "[Error] No too"). Not adjudicable
	// yet: hold promotion for a tick (the next tick sees either
	// divergence → none, the full match → poisoned, or completion →
	// final adjudication). Empty content is NEVER pending (role-only
	// first frames must not stall every live stream).
	pairingGuardPending

	// pairingGuardMatch — the accumulated content starts with a full
	// signature. The attempt is poisoned-bound; it must never be
	// promoted, deadline-picked, or relayed.
	pairingGuardMatch
)

// pairingGuardVerdictForContent classifies an accumulated assistant text
// content string. completed=true (buffer closed — the content can never
// grow) resolves pending to none: a completed body shorter than a
// signature cannot be a match, and holding forever would be wrong.
// Pure function.
func pairingGuardVerdictForContent(content string, completed bool) pairingGuardVerdict {
	if content == "" {
		return pairingGuardNone
	}
	for _, sig := range pairingGuardSignatures {
		if strings.HasPrefix(content, sig) {
			return pairingGuardMatch
		}
	}
	if !completed {
		for _, sig := range pairingGuardSignatures {
			if len(content) < len(sig) && strings.HasPrefix(sig, content) {
				return pairingGuardPending
			}
		}
	}
	return pairingGuardNone
}

// pairingGuardVerdictForBuffer adjudicates one attempt's stream buffer.
// Nil-safe (a released buffer holds no data). Read-only over the buffer:
// chunk slices are shared with the writer, so they are only read.
//
// Cost shape: buffers ≤ pairingGuardParseThreshold and completed buffers
// always go through the exact join+parse path (a rejection completion is
// ~500 bytes, and the exact path runs at most once per completed
// attempt per remaining tick — bounded by the race ending on winner or
// terminal). Buffers above the threshold parse only when the probe
// bytes appear somewhere in the raw chunks, which keeps very long
// healthy streams on a no-allocation per-tick path.
func pairingGuardVerdictForBuffer(buf *streamBuffer) pairingGuardVerdict {
	if buf == nil || buf.TotalLen() == 0 {
		return pairingGuardNone
	}
	completed := buf.IsComplete()
	chunks, _ := buf.GetChunksFrom(0)
	if len(chunks) == 0 {
		return pairingGuardNone
	}

	needParse := completed || buf.TotalLen() <= pairingGuardParseThreshold
	if !needParse {
		probe := []byte(pairingGuardProbe)
		hit := false
		for _, ch := range chunks {
			if ch != nil && bytes.Contains(ch, probe) {
				hit = true
				break
			}
		}
		if !hit {
			return pairingGuardNone
		}
	}

	joined := make([]byte, 0, buf.TotalLen()+int64(len(chunks)))
	for _, ch := range chunks {
		if ch != nil {
			joined = append(joined, ch...)
		}
	}
	content := extractPairingGuardContent(joined)
	return pairingGuardVerdictForContent(content, completed)
}

// extractPairingGuardContent accumulates the assistant TEXT content from
// a proxy-shaped attempt buffer. Two shapes reach an attempt buffer:
//
//   - Streaming (the executor's OpenAI-style relay): "data: {...chunk}"
//     frames where each carries choices[0].delta.content (the executor
//     builds these; upstream chunks are normalized into them), plus the
//     trailing "data: [DONE]".
//
//   - Non-streaming (handleInternalNonStream): ONE marshaled
//     ChatCompletionResponse JSON object with choices[0].message.content.
//
// Everything else (reasoning_content, tool_calls, usage frames, [DONE],
// error frames without choices) contributes nothing — the detector only
// ever reasons about the delivered assistant text.
func extractPairingGuardContent(raw []byte) string {
	trimmed := bytes.TrimLeft(raw, " \t\r\n")
	var sb strings.Builder
	if bytes.HasPrefix(trimmed, []byte("data: ")) {
		for _, line := range bytes.Split(trimmed, []byte("\n")) {
			line = bytes.TrimSpace(line)
			if !bytes.HasPrefix(line, []byte("data: ")) {
				continue
			}
			payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data: ")))
			if bytes.Equal(payload, []byte("[DONE]")) || len(payload) == 0 {
				continue
			}
			appendPairingGuardChoiceContent(payload, &sb)
		}
		return sb.String()
	}
	// Non-stream single-JSON body.
	appendPairingGuardChoiceContent(trimmed, &sb)
	return sb.String()
}

// appendPairingGuardChoiceContent appends the text content of ONE
// OpenAI-shaped frame payload (a full chunk object), preferring the
// streaming delta shape and falling back to the non-stream message
// shape. Malformed payloads contribute nothing (never panic, never
// guess).
func appendPairingGuardChoiceContent(payload []byte, sb *strings.Builder) {
	var chunk map[string]interface{}
	if err := json.Unmarshal(payload, &chunk); err != nil {
		return
	}
	choices, ok := chunk["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return
	}
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return
	}
	if delta, ok := choice["delta"].(map[string]interface{}); ok {
		if c, ok := delta["content"].(string); ok {
			sb.WriteString(c)
		}
		return
	}
	if msg, ok := choice["message"].(map[string]interface{}); ok {
		if c, ok := msg["content"].(string); ok {
			sb.WriteString(c)
		}
	}
}
