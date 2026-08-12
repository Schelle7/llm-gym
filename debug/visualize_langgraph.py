from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from graph import build_graph


OUTPUT_PATH = Path("langgraph.png")


def main():
    graph = build_graph(MemorySaver())
    image = graph.get_graph().draw_mermaid_png()
    OUTPUT_PATH.write_bytes(image)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
