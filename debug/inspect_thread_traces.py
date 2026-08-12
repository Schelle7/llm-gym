import os
import asyncio

from langsmith import Client, RateLimitError


async def list_trace_runs(client, trace_id, project_id):
    for attempt in range(4):
        try:
            return await client.traces.list_runs(
                trace_id,
                project_id=project_id,
                selects=[
                    "ID",
                    "NAME",
                    "RUN_TYPE",
                    "PARENT_RUN_IDS",
                    "START_TIME",
                    "END_TIME",
                ],
            )
        except RateLimitError:
            if attempt == 3:
                raise
            await asyncio.sleep(2**attempt)


async def main():
    project_name = os.getenv("LANGSMITH_PROJECT", "llm-gym")
    thread_id = input("LangSmith thread ID: ").strip()
    client = Client()
    project = await client.aread_project(project_name=project_name)

    traces = []
    async for trace in client.threads.list_traces(
        thread_id,
        project_id=str(project.id),
        selects=["TRACE_ID", "NAME", "START_TIME", "END_TIME", "INPUTS", "OUTPUTS"],
    ):
        traces.append(trace)

    print(f"Project: {project_name}")
    print(f"Thread: {thread_id}")
    print(f"Traces: {len(traces)}")

    for index, trace in enumerate(traces, start=1):
        trace_id = str(trace.trace_id)
        trace_runs_response = await list_trace_runs(
            client,
            trace_id,
            str(project.id),
        )
        trace_runs = trace_runs_response.items

        print(f"\nTrace {index}: {trace_id}")
        print(f"Name: {trace.name}")
        print(f"Started: {trace.start_time}")
        print(f"Ended: {trace.end_time}")
        print(f"Input: {trace.inputs}")
        print(f"Output: {trace.outputs}")
        print(f"Runs: {len(trace_runs)}")

        for run in trace_runs:
            print(
                f"  {run.name} ({run.run_type}): {run.id} "
                f"parent_run_ids={run.parent_run_ids}"
            )


if __name__ == "__main__":
    asyncio.run(main())
