[GATEWAY ESCALATION]
The same failing action has now been repeated {depth} times in a row. A
different model is answering from here, and repeating that action will fail
again for the same reason it failed before.

Before your next tool call:
- State in one sentence what you were trying to do and what error came back.
- Choose a DIFFERENT approach. If `edit` kept failing on a string match, write
  the full file instead. If a command kept failing, read the file or the error
  output before running it again.
- Do not re-issue a call that has already failed with the same arguments.

If you cannot find a different approach, say so plainly and ask the user,
rather than trying the same thing once more.
