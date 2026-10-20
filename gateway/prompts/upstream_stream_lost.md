[SYSTEM GUARD]: The model server closed the connection in the middle of this response. The generation was lost — nothing was written and no tool call from this turn ran. This is an upstream failure (the model process crashed or restarted), not a fault in the task or in the last tool result.

Repeat the step that was interrupted. If the request that triggered it was a large one, split it into smaller stages first: an oversized prompt is the most common way to bring that server down, and repeating it unchanged is the most common way to bring it down again.
