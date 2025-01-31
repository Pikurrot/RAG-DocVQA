import gradio as gr
from src.build_utils import build_model
from src.utils import load_config
from src.process_pdf import load_pdf
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import os

layout_map = {
	0: 'Background',
	1: 'Caption',
	2: 'Footnote',
	3: 'Formula',
	4:'List-item',
	5: 'Page-footer',
	6: 'Page-header',
	7:'Picture',
	8: 'Section-header',
	9: 'Table',
	10: 'Text',
	11: 'Title'
}

color_map = {
	0: 'red',
	1: 'green',
	2: 'blue',
	3: 'yellow',
	4: 'purple',
	5: 'orange',
	6: 'cyan',
	7: 'magenta',
	8: 'lime',
	9: 'pink',
	10: 'brown',
	11: 'gray'
}

color2rgb = {
	"red": (255, 0, 0),
	"green": (0, 255, 0),
	"blue": (0, 0, 255),
	"yellow": (255, 255, 0),
	"purple": (128, 0, 128),
	"orange": (255, 165, 0),
	"cyan": (0, 255, 255),
	"magenta": (255, 0, 255),
	"lime": (0, 255, 0),
	"pink": (255, 192, 203),
	"brown": (165, 42, 42),
	"gray": (128, 128, 128)
}

# Function to create a dataloader generator
def get_dataloader_generator(dataloader):
	for batch in dataloader:
		yield batch

# Function to process the current batch with a given question
def process_next_batch(question: str):
	try:
		global batch, config
		batch["questions"] = [question]  # Update the batch with the input question

		# Inference using the model
		outputs, pred_answers, pred_answer_pages, pred_answers_conf, retrieval = model.inference(
			batch,
			return_retrieval=True,
			chunk_num=config.get("chunk_num", 10),
			chunk_size=config.get("chunk_size", 60),
			chunk_size_tol=config.get("chunk_size_tol", 15),
			overlap=config.get("overlap", 10),
			include_surroundings=config.get("include_surroundings", 0)
		)

		# Prepare original information
		original_images = batch["images"][0]  # List of PIL images
		original_text_list = batch["contexts"][0]  # List of strings (text per page)
		original_text = "\n\n".join([f"Page {i+1}:\n{text}" for i, text in enumerate(original_text_list)])

		# Prepare retrieved information
		retrieved_patches = retrieval["patches"][0]  # List of PIL images
		retrieved_chunks_list = retrieval["text"][0]  # List of strings
		retrieved_top_k_layout_labels = retrieval["top_k_layout_labels"][0]  # List of integers
		retrieved_chunks = "\n\n".join([f"Chunk {i+1} ({layout_map[label]}):\n{text}" for i, (text, label) in enumerate(zip(retrieved_chunks_list, retrieved_top_k_layout_labels))])

		retrieved_page_indices_list = retrieval["page_indices"][0]  # List of integers
		if model.page_retrieval == "concat":
			retrieved_page_indices = ", ".join([str(idx) for idx in retrieved_page_indices_list])
		elif model.page_retrieval == "maxconf":
			retrieved_page_indices = ", ".join([
				f"[{str(idx)}]" if i == retrieval["max_confidence_indices"][0] else str(idx)
				for i, idx in enumerate(retrieved_page_indices_list)
			])

		# Other info
		all_chunks_list = retrieval["steps"]["text_chunks"][0]  # List of strings
		all_chunks = "\n\n".join([f"Chunk {i+1}:\n{text}" for i, text in enumerate(all_chunks_list)])

		# Draw layout boxes on original images
		retrieved_layout_info = retrieval["steps"]["layout_info"][0]  # List of dictionaries
		images_with_boxes = [img.copy() for img in original_images]
		for i, layout_info in enumerate(retrieved_layout_info):
			boxes = layout_info.get("boxes", [])
			labels = layout_info.get("labels", [])
			for j, box in enumerate(boxes):
				resized_box = [
					box[0] * original_images[i].width,
					box[1] * original_images[i].height,
					box[2] * original_images[i].width,
					box[3] * original_images[i].height
				]
				img = images_with_boxes[i]
				draw = ImageDraw.Draw(img)
				draw.rectangle(resized_box, outline=color_map[labels[j]], width=3)
				font = ImageFont.truetype("arial.ttf", 20)  # Specify the font and size
				draw.text(resized_box[:2], layout_map[labels[j]], fill=color_map[labels[j]], font=font)
		
		# Draw layout segments and raw boxes on original images
		retrieved_layout_segments = retrieval["steps"]["layout_segments"][0]  # List of 2d arrays
		retrieved_layout_info_raw = retrieval["steps"]["layout_info_raw"][0]  # List of dictionaries
		images_with_segments_and_boxes = [img.copy() for img in original_images]
		for i, (layout_segment, layout_info_raw) in enumerate(zip(retrieved_layout_segments, retrieved_layout_info_raw)):
			img = images_with_segments_and_boxes[i].convert("RGBA")
			# draw segments transparently
			seg_array = layout_segment.numpy().astype(np.uint8)
			seg_colored = np.zeros((seg_array.shape[0], seg_array.shape[1], 4), dtype=np.uint8)
			for label, color_name in color_map.items():
				r, g, b = color2rgb[color_name]
				seg_colored[seg_array == label] = [r, g, b, 128]
			overlay = Image.fromarray(seg_colored, mode="RGBA")
			blended = Image.alpha_composite(img, overlay)
			# draw boxes
			draw = ImageDraw.Draw(blended)
			boxes = layout_info_raw.get("boxes", [])
			labels = layout_info_raw.get("labels", [])
			for j, box in enumerate(boxes):
				draw.rectangle(box, outline=color_map[labels[j]], width=3)
				font = ImageFont.truetype("arial.ttf", 20)
				draw.text(box[:2], layout_map[labels[j]], fill=color_map[labels[j]], font=font)
			images_with_segments_and_boxes[i] = blended

		# Model outputs
		predicted_answer = pred_answers[0]
		predicted_confidence = pred_answers_conf[0]

		return (
			images_with_boxes,
			original_text,
			images_with_segments_and_boxes,
			all_chunks,
			retrieved_patches,
			retrieved_chunks,
			predicted_answer,
			predicted_confidence,
			retrieved_page_indices
		)

	except StopIteration:
		return ([], "No more batches available.", [], "No more retrieved text.", "", "", 0, "", 0.0, "")

# Function to process a new PDF document
def process_pdf_document(pdf, page):
	gr.Info("Wait for the document to be processed...")
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

	with gr.Column():
		gr.Markdown("## Original Information")
		original_images_output = gr.Gallery(label="Original Page Images", elem_id="original_gallery", columns=2, height=300)
		original_text_output = gr.Textbox(label="Original OCR Text", lines=10)
		with gr.Row():
			with gr.Column():
				gr.Markdown("## Layout Information")
				layout_segments_output = gr.Gallery(label="Layout Segments", elem_id="layout_segments_gallery", columns=2, height=300)
				all_chunks_output = gr.Textbox(label="All Text Chunks", lines=10)
			with gr.Column():
				gr.Markdown("## Retrieved Information")
				retrieved_patches_output = gr.Gallery(label="Retrieved Patches", elem_id="retrieved_gallery", columns=2, height=300)
				retrieved_chunks_output = gr.Textbox(label="Retrieved Text Chunks", lines=10)

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
		layout_segments_output,
		all_chunks_output,
		retrieved_patches_output,
		retrieved_chunks_output,
		predicted_answer_output,
		predicted_confidence_output,
		retrieved_page_indices_output
	])

if __name__ == "__main__":
	print("Starting...")
	args = {
		"model": "RAGVT5",
		"dataset": "MP-DocVQA",
		"embed_model": "BGE",
		"page_retrieval": "Concat",
		"add_sep_token": False,
		#"batch_size": 32,
		"layout_batch_size": 4,
		"chunk_num": 10,
		"chunk_size": 60,
		"chunk_size_tol": 0.2,
		"overlap": 10,
		"include_surroundings": 0,
		"visible_devices": "0",
		"embed_weights": "/home/elopezc/data/models/bge-finetuned-2/checkpoint-820",
		"layout_model_weights": "cmarkea/dit-base-layout-detection",
		"use_layout_labels": True,
	}
	os.environ["CUDA_VISIBLE_DEVICES"] = args["visible_devices"]
	args = argparse.Namespace(**args)
	config = load_config(args)
	print("Building model...")
	model = build_model(config)
	
	# Launch the Gradio demo
	demo.launch()
