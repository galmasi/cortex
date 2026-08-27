package roleflatten

import (
	"context"
	"encoding/json"
	"log/slog"
	"strings"

	"github.com/rossoctl/cortex/authbridge/authlib/pipeline"
	"github.com/rossoctl/cortex/authbridge/authlib/plugins"
)

func init() {
	plugins.RegisterPlugin("role-flatten", func() pipeline.Plugin { return &RoleFlatten{} })
}

type RoleFlatten struct{}

func (p *RoleFlatten) Name() string { return "role-flatten" }

func (p *RoleFlatten) Capabilities() pipeline.PluginCapabilities {
	return pipeline.PluginCapabilities{
		WritesBody:  true,
		RequiresAny: []string{"inference-parser"},
		Description: "Rewrites outbound LLM requests so only user and assistant roles remain.",
	}
}

func (p *RoleFlatten) OnRequest(_ context.Context, pctx *pipeline.Context) pipeline.Action {
	cont := pipeline.Action{Type: pipeline.Continue}

	if len(pctx.Body) == 0 {
		return cont
	}

	path, _, _ := strings.Cut(pctx.Path, "?")
	switch path {
	case "/v1/chat/completions", "/v1/completions", "/v1/messages":
	default:
		return cont
	}

	out, changed := flattenRoles(pctx.Body)
	if changed {
		pctx.SetBody(out)
		slog.Debug("role-flatten: rewrote message roles", "path", path)
	}

	return cont
}

func (p *RoleFlatten) OnResponse(_ context.Context, _ *pipeline.Context) pipeline.Action {
	return pipeline.Action{Type: pipeline.Continue}
}

// flattenRoles rewrites any message role that isn't "user" or "assistant" to
// "user". It preserves all other fields in the request body verbatim. Returns
// the rewritten body and true if any role was changed.
func flattenRoles(body []byte) ([]byte, bool) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(body, &top); err != nil {
		return nil, false
	}

	rawMsgs, ok := top["messages"]
	if !ok || len(rawMsgs) == 0 {
		return nil, false
	}

	var msgs []map[string]json.RawMessage
	if err := json.Unmarshal(rawMsgs, &msgs); err != nil {
		return nil, false
	}

	changed := false
	userRole, _ := json.Marshal("user")

	for i := range msgs {
		roleRaw, exists := msgs[i]["role"]
		if !exists {
			continue
		}
		var role string
		if err := json.Unmarshal(roleRaw, &role); err != nil {
			continue
		}
		if role != "user" && role != "assistant" {
			msgs[i]["role"] = userRole
			changed = true
		}
	}

	if !changed {
		return nil, false
	}

	newMsgs, err := json.Marshal(msgs)
	if err != nil {
		return nil, false
	}
	top["messages"] = newMsgs

	out, err := json.Marshal(top)
	if err != nil {
		return nil, false
	}
	return out, true
}
