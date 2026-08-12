import sys

from langgraph.checkpoint.sqlite import SqliteSaver

from graph import build_graph


CHECKPOINT_DB = "checkpoints/checkpoints.db"


def main():
    thread_id = sys.argv[1]
    config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer)

        print("Current state:")
        print(graph.get_state(config))

        print("\nCheckpoint history:")
        for snapshot in graph.get_state_history(config):
            print(snapshot)


if __name__ == "__main__":
    main()
