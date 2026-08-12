from langchain_core.tools import tool

from tools import get_time, ls


@tool
def get_time_tool():
    """Return the current time in UTC."""
    return get_time()


@tool
def ls_tool(include_hidden: bool):
    """List current-directory entries, optionally including hidden files."""
    return ls(include_hidden)


langgraph_tools = [get_time_tool, ls_tool]
