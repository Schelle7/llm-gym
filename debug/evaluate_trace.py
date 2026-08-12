import os

from langsmith import Client


def messages_from(value):
    if isinstance(value, dict):
        return value.get("messages", [])
    return []


def message_label(message):
    if isinstance(message, dict):
        return message.get("type", message.get("role", "message"))
    return message.__class__.__name__


def message_content(message):
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", str(message))


def evaluate_agent_transition(run):
    inputs = messages_from(run.inputs)
    outputs = messages_from(run.outputs)
    checks = {
        "has_input_messages": bool(inputs),
        "has_output_messages": bool(outputs),
        "one_output_message": len(outputs) == 1,
    }
    return checks


def print_message_list(label, messages):
    print(f"{label} ({len(messages)} messages):")
    for index, message in enumerate(messages, start=1):
        print(f"  {index}. {message_label(message)}: {message_content(message)}")


def main():
    project_name = os.getenv("LANGSMITH_PROJECT", "llm-gym")
    trace_id = input("LangSmith trace ID: ").strip()
    client = Client()
    runs = sorted(
        client.list_runs(project_name=project_name, trace_id=trace_id),
        key=lambda run: run.start_time,
    )

    if not runs:
        print(f"No runs found in project {project_name!r} for trace {trace_id!r}.")
        return

    print(f"Project: {project_name}")
    print(f"Trace: {trace_id}")
    print(f"Runs: {len(runs)}")

    agent_runs = [run for run in runs if run.name == "agent"]
    if not agent_runs:
        print("No run named 'agent' was found in this trace.")
        print("Available runs:")
        for run in runs:
            print(f"  {run.name} ({run.run_type})")
        return

    for index, agent_run in enumerate(agent_runs, start=1):
        print(f"\nAgent node {index}: {agent_run.id}")
        print(f"Started: {agent_run.start_time}")
        print(f"Ended: {agent_run.end_time}")
        print_message_list("State entering agent", messages_from(agent_run.inputs))
        print_message_list("State leaving agent", messages_from(agent_run.outputs))
        print(f"Checks: {evaluate_agent_transition(agent_run)}")

        child_llm_runs = [
            run
            for run in runs
            if run.parent_run_id == agent_run.id and run.run_type == "llm"
        ]
        for llm_run in child_llm_runs:
            print(f"  LLM run: {llm_run.name} ({llm_run.id})")
            print(f"  LLM input: {llm_run.inputs}")
            print(f"  LLM output: {llm_run.outputs}")


if __name__ == "__main__":
    main()
