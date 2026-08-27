The run command has shortcomings in that it doesnt reload files so the shown file content can be stale.
so an accept should compare this presumably, I consider it sufficiently safe for the moment.

A similar shortcoming is that if you ask the agent to make changes to a file it will read it and save the hash.
If you then make chnages before it makes its own changes, then you will get an error on clicking on accept due to an old hash.
