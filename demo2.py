import gradio as gr
from torch.utils.data import DataLoader
from src.build_utils import build_model
from src.utils import load_config
from src.process_pdf import load_pdf
import argparse

# Function to create a dataloader generator
def get_dataloader_generator(dataloader):
    for batch in dataloader:
        yield batch

# Function to process the current batch with a given question
def process_next_batch(question):
    try:
        global batch
        batch["questions"] = [question]  # Update the batch with the input question

        # Inference using the model
        outputs, pred_answers, pred_answer_pages, pred_answers_conf, retrieval = \
            model.inference(batch, return_retrieval=True, include_surroundings=10, k=5)

        # Prepare original information
        original_images = batch["images"][0]  # List of PIL images
        original_text_list = batch["contexts"][0]  # List of strings (text per page)
        original_text = "\n\n".join([f"Page {i+1}:\n{text}" for i, text in enumerate(original_text_list)])

        # Prepare retrieved information
        retrieved_patches = retrieval.get("patches", [[]])[0]  # List of PIL images
        retrieved_text_list = retrieval.get("input_words", [[]])[0]  # List of strings
        retrieved_text = " ".join(retrieved_text_list)

        retrieved_page_indices_list = retrieval.get("page_indices", [[]])[0]  # List of integers
        if model.page_retrieval == "concat":
            retrieved_page_indices = ", ".join([str(idx) for idx in retrieved_page_indices_list])
        elif model.page_retrieval == "maxconf":
            retrieved_page_indices = ", ".join([
                f"[{str(idx)}]" if i == retrieval.get("max_confidence_indices", [0])[0] else str(idx)
                for i, idx in enumerate(retrieved_page_indices_list)
            ])

        # Model outputs
        predicted_answer = pred_answers[0]
        predicted_confidence = pred_answers_conf[0]

        return (original_images, original_text, retrieved_patches, retrieved_text, predicted_answer, predicted_confidence, retrieved_page_indices)

    except StopIteration:
        return ([], "No more batches available.", [], "No more retrieved text.", "", "", 0, "", 0.0, "")

# Function to process a new PDF document
def process_pdf_document(pdf, page):
	global batch
	record = load_pdf(pdf.name)
	
	question = "What is the title of this document?"
	answers = [""]
	answer_page_idx = 0
	num_pages = len(record["ocr_tokens"])

	context = []
	for page_ix in range(num_pages):
		context.append(" ".join([word.lower() for word in record["ocr_tokens"][page_ix]]))
	context_page_corresp = None
	images = record["images"]
	words = []
	boxes = record["ocr_boxes"]
	for p in range(num_pages):
		words.append([word.lower() for word in record["ocr_tokens"][p]])
	start_idxs, end_idxs = None, None
	
	batch = {
		"question_id": 0,
		"questions": [question],
		"answers": [answers],
		"answer_page_idx": [answer_page_idx],
		"contexts": [context],
		"context_page_corresp": context_page_corresp,
		"images": [images],
		"words": [words],
		"boxes": [boxes],
	}
	
	gr.Info("Document processed successfully. Ready for inference.")

with gr.Blocks() as demo:
    gr.Markdown("# RAG Visual Question Answering Demo")

    # Upload Button for PDF
    upload_pdf_button = gr.UploadButton(label="Upload Document in PDF", file_types=[".pdf"])
    question_input = gr.Textbox(label="Enter your question", lines=1)
    ask_button = gr.Button("Submit Question")

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
        predicted_answer_output = gr.Textbox(label="Predicted Answer")
        predicted_confidence_output = gr.Number(label="Predicted Answer Confidence")
        retrieved_page_indices_output = gr.Textbox(label="Retrieved Page Indices")

    # Handling PDF Upload
    upload_pdf_button.upload(fn=process_pdf_document, inputs=upload_pdf_button, outputs=None)

    # Handling Question Submission
    ask_button.click(fn=process_next_batch, inputs=question_input, outputs=[
        original_images_output,
        original_text_output,
        retrieved_patches_output,
        retrieved_text_output,
        predicted_answer_output,
        predicted_confidence_output,
        retrieved_page_indices_output
    ])

if __name__ == "__main__":
    print("Starting...")
    args = {
        "model": "RAGVT5",
        "dataset": "MP-DocVQA",
        "embed_model": "BGE"  # VT5 or BGE
    }
    args = argparse.Namespace(**args)
    config = load_config(args)
    print("Building model...")
    model = build_model(config)
    
    # Launch the Gradio demo
    demo.launch()
