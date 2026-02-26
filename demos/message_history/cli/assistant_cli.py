# Imports
from langgraph.checkpoint.sqlite import SqliteSaver # pip install langgraph-checkpoint-sqlite 

# Import Project Modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from assistant import Assistant

def run_lang_graph_agent(agent: Assistant, thread_id: str = "chat-1"):
    print("=" * 50)
    print("LangGraph Agent CLI")
    print("=" * 50)
    print()

    config = {"configurable": {"thread_id": thread_id}}
    print(f"Using thread ID: {thread_id}")
    print()
    with SqliteSaver.from_conn_string("assistant_checkpoints.db") as checkpointer:
        print("Creating LangGraph...")
        graph = agent.create_graph(checkpointer=checkpointer)
        print("Graph created successfully!")

        print("Saving graph visualization...")
        # save_graph_image(graph)

        state = graph.get_state(config)
        if state.next:
            print("\n🔄 Resuming from checkpoint...")
            graph.invoke(None, config=config)
        else:
            print("\n▶️ Starting new chat...")
            graph.invoke(agent.initial_state, config=config)

def save_graph_image(graph, filename="lg_graph.png"):
    """
    Generate a Mermaid diagram of the graph and save it as a PNG image.
    Uses the graph's built-in Mermaid export functionality.
    """
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")

if __name__ == "__main__":
    from cli.open_ai_assistant import OpenAIAssistant
    agent = OpenAIAssistant()

    # Generate a unique thread ID or use a fixed one for testing
    # Fixed thread ID for testing
    # thread_id = "chat-1"
    # Unique thread ID based on timestamp
    import time
    thread_id = f"chat-{int(time.time())}"


    run_lang_graph_agent(agent, thread_id=thread_id)