# Issue: parallel tool calls without `extra.function_id` produce duplicate tool_call ids

**Title**: Parallel tool calls without `extra.function_id` get the same id (`'1'`), which strict OpenAI-compatible endpoints reject with `400 duplicate tool_call id`

---

## Problem

`BaseChatModel._conv_qwen_agent_messages_to_oai` assigns the tool-call id like this:

```python
if msg.get('function_call'):
    if not new_messages[-1].get('tool_calls'):
        new_messages[-1]['tool_calls'] = []
    new_messages[-1]['tool_calls'].append({
        'id': msg.get('extra', {}).get('function_id', '1'),
        'type': 'function',
        'function': {...},
    })
```

Consecutive assistant messages are merged into a single assistant message carrying several
`tool_calls`. When those messages have **no** `extra.function_id`, every tool call falls
back to the constant `'1'`, so one assistant message ends up with duplicated ids.

The same fallback is used for the tool-result messages, so the ids stay paired — but the
provider rejects the request before that ever matters.

### Reproduction against a real endpoint

`MiniMax-M3.1-Flash`, two tool calls in one assistant message sharing id `1`:

```
POST /v1/chat/completions
[
  {"role": "user", "content": "查两个东西"},
  {"role": "assistant", "tool_calls": [
      {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{\"x\":\"1\"}"}},
      {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{\"x\":\"2\"}"}}]},
  {"role": "tool", "tool_call_id": "1", "content": "r1"},
  {"role": "tool", "tool_call_id": "1", "content": "r2"}
]
```

```
400 {"type": "error", "error": {"type": "bad_request_error",
     "message": "invalid params, duplicate tool_call id: 1 (2013)", ...}}
```

With distinct ids (`1` and `2`) the same request succeeds.

---

## How reachable is this?

Inside the agent loop the ids are always set — `_chat_stream` and (after
[the fix for the non-streaming path](#)) attach `extra={'function_id': tc.id}` from the
provider's tool-call id. So the failure needs messages that reach the converter **without**
`extra.function_id`:

- a `Message(role=ASSISTANT, function_call=...)` built by user code or a custom agent;
- a checkpoint / persisted conversation whose `extra` was dropped;
- any code path that reconstructs assistant messages from scratch.

That makes it a latent hazard rather than a common failure — but the failure mode is a
hard `400` from the provider, and the constant `'1'` is a silent trap: it looks like a
valid id and only breaks under a strict provider.

---

## Two possible fixes — and a pairing constraint

The ids of a tool call and its tool-result message must match, so the fallback has to be
derived **symmetrically** on both sides.

**Option A — derive the fallback from the position**

- assistant side: the k-th `tool_call` of the merged assistant message gets `call_{k}`
- tool side: the j-th tool-result message after that assistant gets `call_{j}`

This keeps the pairing intact, but it relies on tool-result messages immediately
following their assistant message — which holds in the agent loop, but is an assumption
about the message list.

**Option B — require `extra.function_id`**

Fail loudly instead of silently emitting `'1'`: raise if a `function_call` message has no
`extra.function_id`. This is the smallest and safest change, and it turns a distant
`400` into a local, actionable error — but it is a breaking change for anyone who
constructs such messages today.

**Option C — keep the constant but deduplicate on the assistant side only**

Not recommended: the tool-result side would still emit `'1'` twice, and the ids would
silently **mis-pair** (a tool result attached to the wrong tool call), which is worse than
a `400`.

---

## Suggested test

```python
def test_parallel_tool_calls_get_unique_ids():
    messages = [
        Message(role='user', content='hi'),
        Message(role='assistant', content='', function_call={'name': 'f', 'arguments': '{}'}),
        Message(role='assistant', content='', function_call={'name': 'g', 'arguments': '{}'}),
    ]
    converted = BaseChatModel._conv_qwen_agent_messages_to_oai(messages)
    ids = [tc['id'] for tc in converted[1]['tool_calls']]
    assert len(set(ids)) == 2, ids
```

This test is included in the PR that fixes the non-streaming `reasoning_content` path, so
that the id scheme can be settled without the behaviour regressing.

---

Would maintainers prefer **Option A** or **Option B**? Happy to prepare either.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
