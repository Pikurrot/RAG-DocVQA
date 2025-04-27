import gradio as gr
from torch.utils.data import DataLoader
from src.build_utils import build_model, build_dataset
from src.utils import load_config
from src.MP_DocVQA import mpdocvqa_collate_fn
from src._modules import get_layout_model_map, get_raw_layout_model_map
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import os

color_map = {
	-1: 'black',
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
	"black": (0, 0, 0),
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

# Function to process the current batch
def process_next_batch():
	try:
		batch = next(batch_generator)
	except StopIteration:
		return ([], "No more batches available.", [], "", [], "", "", "", "", "", "")
	return process(batch)

def process_specific_item(question_id):
	try:
		item = dataset.sample(question_id=int(question_id))
	except ValueError:
		return ([], "Item not found.", [], "", [], "", "", "", "", "", "", "")
	batch = mpdocvqa_collate_fn([item])
	return process(batch)

def process(batch):
	# Inference using the model
	outputs, pred_answers, pred_answer_pages, pred_answers_conf, retrieval = model.inference(
		batch,
		return_retrieval=True,
		return_steps=True
	)

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
	retrieved_chunks_list = retrieval["text"][0] if "text" in retrieval else [""]*len(retrieved_patches)  # List of strings
	retrieved_top_k_layout_labels = retrieval["top_k_layout_labels"][0]  # List of integers
	retrieved_chunks = "\n\n".join([f"Chunk {i+1} ({layout_map[label]}):\n{text}" for i, (text, label) in enumerate(zip(retrieved_chunks_list, retrieved_top_k_layout_labels))])

	retrieved_page_indices_list = retrieval["page_indices"][0] if "page_indices" in retrieval else [0]*len(retrieved_patches)  # List of integers
	page_retrieval = model.page_retrieval if hasattr(model, "page_retrieval") else "concat"
	if page_retrieval == "oracle":
		retrieved_page_indices = str(retrieved_page_indices_list[0])
	elif page_retrieval == "concat":
		retrieved_page_indices = ", ".join([str(idx) for idx in retrieved_page_indices_list])
	elif page_retrieval == "maxconf":
		retrieved_page_indices = ", ".join([
			f"[{str(idx)}]" if i == retrieval["max_confidence_indices"][0] else str(idx)
			for i, idx in enumerate(retrieved_page_indices_list)
		])

	# Other info
	all_chunks_list = retrieval["steps"]["text_chunks"][0] if "text_chunks" in retrieval["steps"] else [""]*len(retrieved_patches)  # List of strings
	all_chunks = "\n\n".join([f"Chunk {i+1}:\n{text}" for i, text in enumerate(all_chunks_list)])

	# Draw layout boxes on original images
	retrieved_layout_info = retrieval["steps"]["layout_info"][0]  # List of dictionaries
	images_with_boxes = [img.copy() for img in original_images]
	for i, layout_info in enumerate(retrieved_layout_info):
		boxes = layout_info.get("boxes", [])
		labels = layout_info.get("labels", [])
		clusters = layout_info.get("clusters", [-1]*len(boxes))
		for j, box in enumerate(boxes):
			resized_box = [
				box[0] * original_images[i].width,
				box[1] * original_images[i].height,
				box[2] * original_images[i].width,
				box[3] * original_images[i].height
			]
			img = images_with_boxes[i]
			draw = ImageDraw.Draw(img)
			draw.rectangle(resized_box, outline=color_map[clusters[j]], width=3)
			font = ImageFont.truetype("arial.ttf", 40)  # Specify the font and size
			draw.text((resized_box[0], resized_box[1]-40), f"{layout_map[labels[j]]}, {clusters[j]}", fill=color_map[clusters[j]], font=font)
	
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
			if box[0] <= 1:
				box = [
					box[0] * original_images[i].width,
					box[1] * original_images[i].height,
					box[2] * original_images[i].width,
					box[3] * original_images[i].height
				]
			draw.rectangle(box, outline=color_map[labels[j]], width=3)
			font = ImageFont.truetype("arial.ttf", 40)
			draw.text((box[0], box[1]-40), layout_map[labels[j]], fill=color_map[labels[j]], font=font)
		images_with_segments_and_boxes[i] = blended

	# Model outputs
	predicted_answer = pred_answers[0]
	predicted_confidence = pred_answers_conf[0] if pred_answers_conf is not None else 0

	return (
		images_with_boxes,
		original_text,
		images_with_segments_and_boxes,
		all_chunks,
		retrieved_patches,
		retrieved_chunks,
		question,
		actual_answers,
		actual_answer_page_idx,
		predicted_answer,
		predicted_confidence,
		retrieved_page_indices
	)

with gr.Blocks() as demo:
	gr.Markdown("# RAG Visual Question Answering Demo")

	next_button = gr.Button("Load Next Batch")
	with gr.Row():
		question_id = gr.Textbox(label="Question ID", placeholder="Enter Question ID")
		load_button = gr.Button("Load Sample")

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
		question_output = gr.Textbox(label="Question")
		actual_answers_output = gr.Textbox(label="Actual Answers")
		actual_answer_page_idx_output = gr.Number(label="Actual Answer Page Index")
		predicted_answer_output = gr.Textbox(label="Predicted Answer")
		predicted_confidence_output = gr.Number(label="Predicted Answer Confidence")
		retrieved_page_indices_output = gr.Textbox(label="Retrieved Page Indices")

	next_button.click(fn=process_next_batch, inputs=None, outputs=[
		original_images_output,
		original_text_output,
		layout_segments_output,
		all_chunks_output,
		retrieved_patches_output,
		retrieved_chunks_output,
		question_output,
		actual_answers_output,
		actual_answer_page_idx_output,
		predicted_answer_output,
		predicted_confidence_output,
		retrieved_page_indices_output
	])

	load_button.click(fn=process_specific_item, inputs=question_id, outputs=[
		original_images_output,
		original_text_output,
		layout_segments_output,
		all_chunks_output,
		retrieved_patches_output,
		retrieved_chunks_output,
		question_output,
		actual_answers_output,
		actual_answer_page_idx_output,
		predicted_answer_output,
		predicted_confidence_output,
		retrieved_page_indices_output
	])

if __name__ == "__main__":
	print("Starting...")
	# args = {
	# 	"use_RAG": True,
	# 	"model": "RAGVT5",
	# 	"dataset": "MP-DocVQA",
	# 	"embed_model": "BGE",
	# 	"reranker_model": "BGE",
	# 	"page_retrieval": "Concat",
	# 	"add_sep_token": False,
	# 	"batch_size": 1,
	# 	"layout_batch_size": 4,
	# 	"chunk_num": 20,
	# 	"chunk_size": 60,
	# 	"chunk_size_tol": 0.2,
	# 	"overlap": 10,
	# 	"include_surroundings": 0,
	# 	"model_weights": "Qwen/Qwen2.5-VL-7B-Instruct",
	# 	"embed_weights": "/data/users/elopez/models/bge-finetuned/checkpoint-820",
	# 	"reorder_chunks": False,
	# 	"reranker_weights": "BAAI/bge-reranker-v2-m3",
	# 	"rerank_filter_tresh": 0,
	# 	"rerank_max_chunk_num": 10,
	# 	"rerank_min_chunk_num": 1
	# }
	args = {
		"use_RAG": True,
		"model": "RAGPix2Struct",
		"layout_model": "DIT",
		"dataset": "MP-DocVQA", # MP-DocVQA / Infographics / DUDE
		"batch_size": 1,
		"layout_batch_size": 4,
		"embedder_batch_size": 16,
		"use_precomputed_layouts": False,
		"use_layout_labels": True,
		"chunk_mode": "horizontal",
		"chunk_num": 5,
		"include_surroundings": (0,0),
		"model_weights": "google/pix2struct-docvqa-base",
		"layout_model_weights": "cmarkea/dit-base-layout-detection",
		"use_precomputed_layouts": True,
		"precomputed_layouts_path": "/data/users/elopez/data/images_layouts_dit_s2_spa.npz",
		"cluster_layouts": True,
		"cluster_mode": "spatial",
		"calculate_n_clusters": "best"
	}
	extra_args = {
		"visible_devices": "0,1,2,3,4",
		"device": "cuda:1",
		"save_folder": "9-train_generator_with_layout",
		"save_name_append": "train_generator",
		"val_size": 1.0,
		"log_wandb": False,
		"log_media_interval": 10,
		"return_scores_by_sample": True,
		"return_answers": True,
		"save_results": False,
		"save_continuously": True,
		"compute_stats": False,
		"compute_stats_examples": True,
		"n_stats_examples": 5,
	}
	args.update(extra_args)
	os.environ["CUDA_VISIBLE_DEVICES"] = args["visible_devices"]
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
	args = argparse.Namespace(**args)
	config = load_config(args)
	config["page_retrieval"] = config["page_retrieval"].lower()
	layout_map = get_layout_model_map(config)
	raw_layout_map = get_raw_layout_model_map(config)
	print("Building model...")
	model = build_model(config)
	print("Building dataset...")
	dataset = build_dataset(config=config, split="val")
	
	# Initialize the dataloader
	dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=mpdocvqa_collate_fn, num_workers=0)
	
	# Create a generator from the dataloader
	batch_generator = get_dataloader_generator(dataloader)
	
	# Launch the Gradio demo
	demo.launch(server_name="0.0.0.0", server_port=7860)
