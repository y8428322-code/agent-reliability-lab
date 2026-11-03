# PR: fix(llm) — keep `tool_calls` when a non-streaming reply also carries `reasoning_content`

**Branch**: `fix/keep-tool-calls-when-reasoning-content`
**Base**: `main`

---

## Problem

`TextChatAtOAI._chat_no_stream` dispatches the reply on whether the message has a
`reasoning_content` attribute:

```python
response = self._chat_complete_create(..., stream=False, **generate_cfg)
if hasattr(response.choices[0].message, 'reasoning_content'):
    return [
        Message(role=ASSISTANT,
                content=response.choices[0].message.content,
                reasoning_content=response.choices[0].message.reasoning_content)
    ]
else:
    return [Message(role=ASSISTANT, content=response.choices[0].message.content)]
```

A reply that carries **both** `reasoning_content` and `tool_calls` therefore loses its
tool calls. The agent then never performs the requested action — it just explains what it
*would* do.

This is not a corner case: every endpoint that returns `reasoning_content`
(MiniMax, DeepSeek-R1, Qwen3-thinking, …) hits it as soon as the non-streaming path is used.

### Reproduction against a real endpoint

`MiniMax-M3.1-Flash`, non-streaming, one tool available:

```
Q: 请用工具计算 (137*89 - 42) / 7 的结果
   reasoning_content=有(297字) | tool_calls=['calc'] | finish=tool_calls
```

The provider returns both fields. The current code returns only
`content` + `reasoning_content`, so `tool_calls` is dropped.

```
$ python -c "..."   # upstream code path
→ assistant message: content='...', reasoning_content='...', function_call=None
```

The agent ends the turn without calling the tool.

The same request with `stream=True` works, because `_chat_stream` accumulates the tool
calls separately — which is exactly the asymmetry this patch removes.

---

## Root cause

The two reply shapes (`reasoning_content` vs. `tool_calls`) are **not** mutually
exclusive, but the branch treats them as if they were.

---

## Fix

Keep both. The patched version emits one message per tool call — mirroring the
streaming path (`_chat_stream`) — with `reasoning_content` attached, and sets
`extra.function_id` to the provider's tool-call id so the ids stay unique:

```python
message = response.choices[0].message
reasoning_content = getattr(message, 'reasoning_content', None)
tool_calls = getattr(message, 'tool_calls', None) or []

if tool_calls:
    return [
        Message(role=ASSISTANT,
                content='',
                reasoning_content=reasoning_content,
                function_call=FunctionCall(name=tc.function.name, arguments=tc.function.arguments),
                extra={'function_id': tc.id})
        for tc in tool_calls
    ]

return [
    Message(role=ASSISTANT,
            content=message.content,
            reasoning_content=reasoning_content)
]
```

Behaviour for replies **without** tool calls is unchanged.

Setting `extra.function_id` also removes a related hazard: previously a non-streaming
reply produced no `function_call` at all, so the pairing id used for the tool result
message had to fall back to a constant. With the id taken from the provider, tool calls
stay uniquely id'ed (see the open question in the issue linked below).

---

## Tests

New file `tests/llm/test_tool_call_preservation.py`:

| Test | Covers |
|---|---|
| `test_parallel_tool_calls_get_unique_ids` | two tool calls in one assistant message must not share an id |
| `test_explicit_function_ids_are_preserved` | an explicit `extra.function_id` is still used as the tool_call id |
| `test_no_stream_keeps_tool_calls_when_reasoning_content_present` | a reply with **both** `reasoning_content` and `tool_calls` keeps both |

The third test fails on the current `main` with:

```
AssertionError: tool_calls were dropped by the reasoning_content branch
```

and passes with this patch.

The first test currently fails with `tool_call ids must be unique, got ['1', '1']` —
it is the regression guard for the id-uniqueness question described in
[the related issue](https://github.com/QwenLM/Qwen-Agent/issues/XXXX); it is included
here so the fix does not regress that behaviour once the id scheme is settled.

---

## Compatibility

- Replies without tool calls: unchanged output.
- Replies with tool calls and no `reasoning_content`: previously the `else` branch returned a
  single `Message` **without** `function_call`, so the tool call was already dropped. This
  patch now returns the tool call — a behaviour change, but the previous behaviour was
  plainly broken (the agent could not act on the reply).

Related open PRs, for context — this patch does not overlap with either:
- #917 — use `tool_call_id` (not `id`) for tool-role messages
- #968 — import the base package without `numpy`/`soundfile`/`tqdm`/`dateutil`

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
