Execution checks the current file hash against the reviewed proposal on resume.
The sandbox still opens the file by path, so a change between that check and
the interpreter opening the file can race with execution. Imported files are
not covered by the script's hash.

A similar shortcoming is that if you ask the agent to make changes to a file it will read it and save the hash.
If you then make chnages before it makes its own changes, then you will get an error on clicking on accept due to an old hash.

Multiple approval-requiring tool calls in a single model response are not supported. The models tested so far have only requested one such action per response.
