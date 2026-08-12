import os
from datetime import datetime, timezone


def get_time():
    return {"utc": datetime.now(timezone.utc).isoformat()}


def ls(include_hidden):
    entries = os.listdir(".")
    if not include_hidden:
        entries = [entry for entry in entries if not entry.startswith(".")]
    return {"entries": sorted(entries)}


tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Return the current time in UTC.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List the files in the current directory, optionally including hidden files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Whether to include entries whose names start with a dot.",
                    },
                },
                "required": ["include_hidden"],
            },
        },
    },
]

tool_registry = {
    "get_time": get_time,
    "ls": ls,
}
