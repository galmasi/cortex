package roleflatten

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/rossoctl/cortex/authbridge/authlib/pipeline"
)

func TestFlattenRoles_OpenAI_SystemAndTool(t *testing.T) {
	body := `{
		"model": "gpt-4",
		"messages": [
			{"role": "system", "content": "You are a helper."},
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": "Hi there!"},
			{"role": "tool", "content": "result data", "tool_call_id": "call_123"}
		],
		"temperature": 0.7
	}`

	out, changed := flattenRoles([]byte(body))
	if !changed {
		t.Fatal("expected change")
	}

	var result map[string]json.RawMessage
	if err := json.Unmarshal(out, &result); err != nil {
		t.Fatal(err)
	}

	var msgs []map[string]json.RawMessage
	if err := json.Unmarshal(result["messages"], &msgs); err != nil {
		t.Fatal(err)
	}

	expectedRoles := []string{"user", "user", "assistant", "user"}
	for i, msg := range msgs {
		var role string
		json.Unmarshal(msg["role"], &role)
		if role != expectedRoles[i] {
			t.Errorf("message %d: got role %q, want %q", i, role, expectedRoles[i])
		}
	}

	// Verify tool_call_id is preserved on the tool message
	var toolCallID string
	json.Unmarshal(msgs[3]["tool_call_id"], &toolCallID)
	if toolCallID != "call_123" {
		t.Errorf("tool_call_id lost: got %q", toolCallID)
	}

	// Verify temperature preserved
	if _, ok := result["temperature"]; !ok {
		t.Error("temperature field lost")
	}
}

func TestFlattenRoles_Anthropic_SystemFieldUntouched(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-20250514",
		"system": "You are a helpful assistant.",
		"messages": [
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": "Hi!"},
			{"role": "developer", "content": "extra context"}
		],
		"max_tokens": 1024
	}`

	out, changed := flattenRoles([]byte(body))
	if !changed {
		t.Fatal("expected change")
	}

	var result map[string]json.RawMessage
	json.Unmarshal(out, &result)

	// System field should be preserved as-is
	var sys string
	json.Unmarshal(result["system"], &sys)
	if sys != "You are a helpful assistant." {
		t.Errorf("system field modified: got %q", sys)
	}

	var msgs []map[string]json.RawMessage
	json.Unmarshal(result["messages"], &msgs)

	expectedRoles := []string{"user", "assistant", "user"}
	for i, msg := range msgs {
		var role string
		json.Unmarshal(msg["role"], &role)
		if role != expectedRoles[i] {
			t.Errorf("message %d: got role %q, want %q", i, role, expectedRoles[i])
		}
	}
}

func TestFlattenRoles_NoChange(t *testing.T) {
	body := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "Hello"},
			{"role": "assistant", "content": "Hi!"}
		]
	}`

	_, changed := flattenRoles([]byte(body))
	if changed {
		t.Error("expected no change when only user/assistant roles present")
	}
}

func TestFlattenRoles_EmptyBody(t *testing.T) {
	_, changed := flattenRoles(nil)
	if changed {
		t.Error("expected no change on nil body")
	}

	_, changed = flattenRoles([]byte{})
	if changed {
		t.Error("expected no change on empty body")
	}
}

func TestFlattenRoles_NoMessages(t *testing.T) {
	body := `{"model": "gpt-4", "prompt": "hello"}`
	_, changed := flattenRoles([]byte(body))
	if changed {
		t.Error("expected no change when no messages field")
	}
}

func TestFlattenRoles_DeveloperAndFunction(t *testing.T) {
	body := `{
		"model": "gpt-4",
		"messages": [
			{"role": "developer", "content": "instructions"},
			{"role": "function", "name": "get_weather", "content": "{\"temp\": 72}"},
			{"role": "assistant", "content": "The temperature is 72F."}
		]
	}`

	out, changed := flattenRoles([]byte(body))
	if !changed {
		t.Fatal("expected change")
	}

	var result map[string]json.RawMessage
	json.Unmarshal(out, &result)

	var msgs []map[string]json.RawMessage
	json.Unmarshal(result["messages"], &msgs)

	expectedRoles := []string{"user", "user", "assistant"}
	for i, msg := range msgs {
		var role string
		json.Unmarshal(msg["role"], &role)
		if role != expectedRoles[i] {
			t.Errorf("message %d: got role %q, want %q", i, role, expectedRoles[i])
		}
	}

	// Verify name field preserved on former function message
	var name string
	json.Unmarshal(msgs[1]["name"], &name)
	if name != "get_weather" {
		t.Errorf("name field lost: got %q", name)
	}
}

func TestOnRequest_NonInferencePath(t *testing.T) {
	p := &RoleFlatten{}
	pctx := &pipeline.Context{
		Path: "/v1/embeddings",
		Body: []byte(`{"model":"x","messages":[{"role":"system","content":"hi"}]}`),
	}
	action := p.OnRequest(context.Background(), pctx)
	if action.Type != pipeline.Continue {
		t.Error("expected Continue")
	}
}

func TestOnRequest_InferencePath(t *testing.T) {
	p := &RoleFlatten{}
	pctx := &pipeline.Context{
		Path: "/v1/chat/completions",
		Body: []byte(`{"model":"gpt-4","messages":[{"role":"system","content":"hi"},{"role":"user","content":"hello"}]}`),
	}
	pctx.SetCurrentPlugin("role-flatten", pipeline.InvocationPhaseRequest)

	action := p.OnRequest(context.Background(), pctx)
	if action.Type != pipeline.Continue {
		t.Error("expected Continue")
	}
	if !pctx.BodyMutated() {
		t.Error("body should be mutated")
	}

	// Verify the body was rewritten
	var result map[string]json.RawMessage
	json.Unmarshal(pctx.Body, &result)
	var msgs []map[string]json.RawMessage
	json.Unmarshal(result["messages"], &msgs)
	var role string
	json.Unmarshal(msgs[0]["role"], &role)
	if role != "user" {
		t.Errorf("first message role not flattened: got %q", role)
	}
}

func TestOnRequest_QueryStringInPath(t *testing.T) {
	p := &RoleFlatten{}
	pctx := &pipeline.Context{
		Path: "/v1/messages?beta=true",
		Body: []byte(`{"model":"claude-sonnet-4-20250514","messages":[{"role":"developer","content":"x"},{"role":"user","content":"hi"}]}`),
	}
	pctx.SetCurrentPlugin("role-flatten", pipeline.InvocationPhaseRequest)

	action := p.OnRequest(context.Background(), pctx)
	if action.Type != pipeline.Continue {
		t.Error("expected Continue")
	}
	if !pctx.BodyMutated() {
		t.Error("body should be mutated even with query string in path")
	}
}

func TestOnRequest_EmptyBody(t *testing.T) {
	p := &RoleFlatten{}
	pctx := &pipeline.Context{
		Path: "/v1/chat/completions",
		Body: nil,
	}
	action := p.OnRequest(context.Background(), pctx)
	if action.Type != pipeline.Continue {
		t.Error("expected Continue")
	}
}
