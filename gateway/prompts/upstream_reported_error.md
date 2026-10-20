[SYSTEM GUARD]: The model server reported an error in the middle of this response and stopped generating. Anything above is a partial answer — the turn did not finish, and no tool call from this turn ran.

The engine said: {detail}

This is an upstream failure, not a fault in the task or in the last tool result. Do not re-read the last result looking for a mistake that is not there; everything done earlier in this session is intact.

If the message mentions memory, the request was too large for the server to handle in one piece. Retry it in smaller stages — fewer files, a narrower part of a file, or fewer images — rather than sending the same one again unchanged. Otherwise, say plainly that the turn failed upstream and ask how to proceed.
