# Import necessary libraries
from langgraph.checkpoint.sqlite import SqliteSaver

# Import project modules
from qwen_base_assistant import BaseQwenAssistant


class Qwen3VLAssistant(BaseQwenAssistant):
    """
    Clean Qwen 3 VL assistant using a local OpenAI-compatible runtime.
    """

    default_model_name = "Qwen/Qwen3-VL-4B-Instruct"
    display_name = "Qwen3-VL Assistant"


def main():
    print("=" * 50)
    print("Qwen3-VL Assistant with Tools")
    print("=" * 50)
    print()
    print("Expected local runtime:")
    print("  vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000")
    print()

    thread_id = "qwen3vl-chat-1"
    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string("qwen3vl_checkpoints.db") as checkpointer:
        agent = Qwen3VLAssistant()

        print("Creating LangGraph...")
        graph = agent.create_graph(checkpointer=checkpointer)
        print("Graph created successfully!")

        state = graph.get_state(config)
        if state.next:
            print("\nResuming from checkpoint...")
            graph.invoke(None, config=config)
        else:
            print("\nStarting new chat...")
            graph.invoke(agent.initial_state, config=config)


if __name__ == "__main__":
    main()
