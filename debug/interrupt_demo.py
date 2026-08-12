"""Minimal, LLM-free demo of LangGraph's interrupt() / Command(resume=...).

Run it:  python interrupt_demo.py

Nothing here talks to Ollama. The point is to watch, in isolation, what
interrupt() does to graph execution.

Three things to watch in the output:
  1. interrupt() does NOT wait. The input() lives in the CALLING code below,
     outside the graph. interrupt() just unwinds out of invoke().
  2. ask_human() re-runs FROM THE TOP on every resume. Watch the entry
     counter and the "SIDE EFFECT" line repeat.
  3. Two interrupts in one node = two separate pauses. The first one is
     replayed from memory on the second pass; the second one unwinds.
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    log: list[str]


entry_count = 0


def prepare(state: State) -> State:
    print("    [prepare] running")
    return {"log": state["log"] + ["prepare"]}


def ask_human(state: State) -> State:
    global entry_count
    entry_count += 1
    print(f"    [ask_human] top of function  (entry #{entry_count})")

    # Anything before an interrupt() runs again on every resume. Imagine this
    # was `open(path, "w").write(...)` -- it would fire multiple times.
    print("    [ask_human] SIDE EFFECT HERE would repeat!")

    first = interrupt({"question": "Approve the change? (yes/no)"})
    print(f"    [ask_human] past interrupt #1, it returned: {first!r}")

    second = interrupt({"question": "Leave a comment:"})
    print(f"    [ask_human] past interrupt #2, it returned: {second!r}")

    return {"log": state["log"] + [f"decision={first}", f"comment={second}"]}


def finish(state: State) -> State:
    print("    [finish] running")
    return {"log": state["log"] + ["finish"]}


builder = StateGraph(State)
builder.add_node("prepare", prepare)
builder.add_node("ask_human", ask_human)
builder.add_node("finish", finish)
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "ask_human")
builder.add_edge("ask_human", "finish")
builder.add_edge("finish", END)

# A checkpointer is REQUIRED for interrupt() -- the paused state must live
# somewhere. InMemorySaver keeps it in RAM (fine here); agent.py uses
# SqliteSaver, which would survive a process restart.
with SqliteSaver.from_conn_string("checkpoints/checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

    # The resume calls MUST use the same thread_id, or there is no paused run
    # to resume.
    config = {"configurable": {"thread_id": "demo-thread"}}


    print("=== invoke #1: start the graph ===")
    result = graph.invoke({"log": []}, config)

    # THIS is the loop that matters. invoke() returned instead of blocking, so we
    # are free to go get a value from anywhere -- input(), an HTTP request, a
    # message queue -- and hand it back via Command(resume=...).
    pass_number = 1
    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        print(f"\n  graph is PAUSED. it is asking: {payload}")
        print(f"  'finish' has not run yet. next node = {graph.get_state(config).next}")

        answer = input(f"\n>>> {payload['question']} ")

        pass_number += 1
        print(f"\n=== invoke #{pass_number}: resume with {answer!r} ===")
        result = graph.invoke(Command(resume=answer), config)

    print("\n--- done ---")
    print("state log        :", result["log"])
    print("__interrupt__    :", result.get("__interrupt__"))
    print("next node to run :", graph.get_state(config).next, "(empty = finished)")
    print(f"ask_human() was entered {entry_count} times for ONE logical visit.")