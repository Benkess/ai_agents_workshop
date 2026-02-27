# Import necessary libraries
from langgraph.checkpoint.sqlite import SqliteSaver

# Import project modules
from qwen_base_assistant import BaseQwenAssistant


class Qwen25VLAssistant(BaseQwenAssistant):
    """
    Clean Qwen 2.5 VL assistant using a local OpenAI-compatible runtime.
    """

    default_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    display_name = "Qwen2.5-VL Assistant"


def main():
    print("=" * 50)
    print("Qwen2.5-VL Assistant with Tools")
    print("=" * 50)
    print()
    print("Expected local runtime:")
    print("  vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8000")
    print()

    thread_id = "qwen25vl-chat-1"
    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string("qwen25vl_checkpoints.db") as checkpointer:
        agent = Qwen25VLAssistant()

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
