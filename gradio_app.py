import gradio as gr
from main import initialize_vectorstore, create_rag_chain

def launch_app():
    # Initialize Core Logic
    vectorstore = initialize_vectorstore()
    rag_chain = create_rag_chain(vectorstore)

    if not rag_chain:
        print("RAG/VectorStore init failed.")
        return

    # Chat wrapper function
    def ask_papagan(message, history):
        try:
            partial_message = ""
            for chunk in rag_chain.stream(message):
                partial_message += chunk
                yield partial_message
        except Exception as e:
            yield f"Error: {str(e)}"

    # Gradio UI
    print("Starting Gradio Server...")
    demo = gr.ChatInterface(
        fn=ask_papagan,
        title="PAPAGAN CHATBOT",
        description="Dokümanlarınızla sohbet edin.",
        examples=["Temel tasarım ilkeleri nelerdir?", "Scrum nedir?"],
        cache_examples=False
    )
    
    demo.launch(share=False)

if __name__ == "__main__":
    launch_app()
