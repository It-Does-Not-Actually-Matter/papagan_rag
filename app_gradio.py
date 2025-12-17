import gradio as gr
from main import process_query  # main.py ile ayni klasorde oldugu icin

def chatbot(message, history=None):
    response = process_query(message)
    return response

demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, label="Sorunu yaz"),
    outputs=gr.Textbox(label="Cevap"),
    title="Papagan RAG",
    description="Datalar uzerinden RAG ile cevap veren basit chatbot",
    theme="compact"
)

if __name__ == "__main__":
    demo.launch(share=True)