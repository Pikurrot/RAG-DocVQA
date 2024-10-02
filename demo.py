import gradio as gr
from torch.utils.data import DataLoader
from src.build_utils import build_model, build_dataset
from src.utils import load_config
from src.MP_DocVQA import mpdocvqa_collate_fn
import argparse

# Function to create a dataloader generator
def get_dataloader_generator(dataloader):
    for batch in dataloader:
        yield batch

# Function to process the current batch
def process_next_batch():
    try:
        # Fetch the next batch from the dataloader generator
        batch = next(batch_generator)
        
        # Inference using the model
        outputs, pred_answers, pred_answer_pages, pred_answers_conf, retrieval = model.inference(batch, return_retrieval=True)
        
        # Prepare original information
        original_images = batch["images"][0]  # List of PIL images
        original_text_list = batch["contexts"][0]  # List of strings (text per page)
        original_text = "\n\n".join([f"Page {i+1}:\n{text}" for i, text in enumerate(original_text_list)])
        question = batch["questions"][0]
        actual_answers_list = batch["answers"][0]  # List of strings
        actual_answers = ", ".join(actual_answers_list)
        actual_answer_page_idx = batch["answer_page_idx"][0]

        # Prepare retrieved information
        retrieved_patches = retrieval["patches"][0]  # List of PIL images
        retrieved_text_list = retrieval["text"][0]  # List of strings
        retrieved_text = "\n".join(retrieved_text_list)
        retrieved_page_indices_list = retrieval["page_indices"][0]  # List of integers
        retrieved_page_indices = ", ".join(map(str, retrieved_page_indices_list))

        # Model outputs
        predicted_answer = pred_answers[0]
        predicted_confidence = pred_answers_conf[0]

        return (original_images, original_text, retrieved_patches, retrieved_text, question, actual_answers, actual_answer_page_idx, predicted_answer, predicted_confidence, retrieved_page_indices)
    
    except StopIteration:
        return ([], "No more batches available.", [], "No more retrieved text.", "", "", 0, "", 0.0, "")

with gr.Blocks() as demo:
    gr.Markdown("# RAG Visual Question Answering Demo")

    with gr.Row():
        next_button = gr.Button("Load Next Batch")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Original Information")
            original_images_output = gr.Gallery(label="Original Page Images", elem_id="original_gallery", columns=2, height=300)
            original_text_output = gr.Textbox(label="Original OCR Text", lines=10)

        with gr.Column():
            gr.Markdown("## Retrieved Information")
            retrieved_patches_output = gr.Gallery(label="Retrieved Patches", elem_id="retrieved_gallery", columns=2, height=300)
            retrieved_text_output = gr.Textbox(label="Retrieved Text Chunks", lines=10)

    gr.Markdown("## Question and Answers")

    with gr.Row():
        question_output = gr.Textbox(label="Question")
        actual_answers_output = gr.Textbox(label="Actual Answers")
        actual_answer_page_idx_output = gr.Number(label="Actual Answer Page Index")
        predicted_answer_output = gr.Textbox(label="Predicted Answer")
        predicted_confidence_output = gr.Number(label="Predicted Answer Confidence")
        retrieved_page_indices_output = gr.Textbox(label="Retrieved Page Indices")

    next_button.click(fn=process_next_batch, inputs=None, outputs=[
        original_images_output,
        original_text_output,
        retrieved_patches_output,
        retrieved_text_output,
        question_output,
        actual_answers_output,
        actual_answer_page_idx_output,
        predicted_answer_output,
        predicted_confidence_output,
        retrieved_page_indices_output
    ])

if __name__ == "__main__":
    print("Starting...")
    args = {
        "model": "VT5",
        "dataset": "MP-DocVQA",
        "embed_model": "BGE"
    }
    args = argparse.Namespace(**args)
    config = load_config(args)
    print("Building model...")
    model = build_model(config)
    print("Building dataset...")
    mpdocvqa = build_dataset(config=config, split="val")
    
    # Initialize the dataloader
    dataloader = DataLoader(mpdocvqa, batch_size=1, shuffle=False, collate_fn=mpdocvqa_collate_fn)
    
    # Create a generator from the dataloader
    batch_generator = get_dataloader_generator(dataloader)
    
    # Launch the Gradio demo
    demo.launch()
