import torch
import gradio as gr
from transformers import pipeline

# Define the path to the pre-trained model
model_path = "../Model/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"

# Initialize the summarization pipeline
text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

# Function to summarize input text
def summary(input_text):
    output = text_summary(input_text)
    return output[0]['summary_text']

# Close any existing Gradio interfaces
gr.close_all()

# Create the Gradio interface
demo = gr.Interface(
    fn=summary,
    inputs=gr.Textbox(
        label="Enter Text to Summarize",
        placeholder="Type or paste your text here...",
        lines=6
    ),
    outputs=gr.Textbox(
        label="Summarized Text",
        placeholder="Your summarized text will appear here...",
        lines=4
    ),
    title="📄 sithija's Text Summarizer",
    description="This application allows you to summarize long texts into concise and meaningful summaries. Ideal for students, professionals, and anyone looking for quick insights!"
)

# Launch the interface
demo.launch(share=True)
