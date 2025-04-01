from torch.utils.data import Dataset
from src.MP_DocVQA import mpdocvqa_collate_fn
from src.build_utils import build_dataset, build_model
from src.utils import load_config, flatten
from src.QwenVLInstruct import QwenVLForConditionalGeneration, Qwen2_5_VLConfig
from src.RAGVT5 import RAGVT5
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from datetime import datetime
from PIL import Image
from transformers import (
	Qwen2_5_VLForConditionalGeneration,
	Qwen2_5_VLConfig,
	AutoProcessor,
	TrainerCallback
)
import torch
import argparse
import os
import gc

class QwenDatasetWrapper(Dataset):
	"""
	Wrapper around an existing dataset that preprocesses batches for Qwen model.
	"""
	def __init__(self, original_dataset, use_layout_labels="Default", add_sep_token=False):
		self.dataset = original_dataset
		self.use_layout_labels = use_layout_labels
		self.add_sep_token = add_sep_token

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		# Get the original item from dataset
		item = self.dataset[idx]
		
		# Process the batch similar to how it's done in forward method
		if self.use_layout_labels == "Text":
			add_sep_token = "."
		else:
			add_sep_token = "<sep>" if self.add_sep_token else None
		
		# Create a single item (not batched) for the trainer
		# We'll handle batching in the collate function
		new_item = {}
		
		# Handle question
		question = item["questions"] if "questions" in item else item["question"]
		new_item["question"] = question
		
		# Handle words
		if isinstance(item["words"][0], list):  # Nested lists
			words = item["words"]
			new_item["words"] = words
		else:  # Single list
			words = [item["words"]]
			new_item["words"] = words
		
		# Handle boxes if available
		if "boxes" in item:
			if isinstance(item["boxes"][0], list):  # Nested lists
				boxes = item["boxes"]
			else:  # Single list
				boxes = [item["boxes"]]
			new_item["boxes"] = boxes
		
		# Handle images
		images = item["images"]
		if not isinstance(images, list):
			images = [images]
		new_item["images"] = images
		
		# Handle answers if available
		if "answers" in item:
			answers = item["answers"]
			if not isinstance(answers, list):
				answers = [answers]
			new_item["answers"] = answers
		
		return new_item

def qwen_collate_fn(batch, processor):
	"""
	Custom collate function for Qwen2.5 VL that handles MP-DocVQA dataset structure
	"""
	questions = []
	all_words = []
	all_images = []
	answers = []
	
	# Extract data from batch - each item is already a dict with lists
	for item in batch:
		# Get question
		question = item["question"]
		if isinstance(question, list):
			questions.extend(question)
		else:
			questions.append(question)
		
		# Get words - already list of lists
		words = item["words"]
		for word_list in words:
			all_words.append(word_list)
		
		# Get images - handle different formats
		images = item["images"]
		# Ensure images is a list of lists
		if not isinstance(images[0], list):
			# If it's a flat list of images, make it a list with one item
			all_images.append(images)
		else:
			# If it's already a list of lists, extend
			all_images.extend(images)
		
		# Get answers if available
		if "answers" in item:
			answers.extend(item["answers"])
	
	# Resize small images
	resized_images = []
	for image_list in all_images:
		batch_resized = []
		# Make sure we're working with a list
		if hasattr(image_list, 'mode'):  # It's a single PIL image
			image_list = [image_list]
			
		for img in image_list:
			# Resize image if too small
			if img.width < 28 or img.height < 28:
				new_width = max(img.width, 28)
				new_height = max(img.height, 28)
				batch_resized.append(img.resize((new_width, new_height)))
			# Downscale if too large (add this part)
			elif img.width > 448 or img.height > 448:
				# Maintain aspect ratio when resizing
				aspect = img.width / img.height
				if aspect > 1:
					new_width = 448
					new_height = int(448 / aspect)
				else:
					new_height = 448
					new_width = int(448 * aspect)
				batch_resized.append(img.resize((new_width, new_height), Image.LANCZOS))
			else:
				batch_resized.append(img)
		resized_images.append(batch_resized)
	
	# Create message format expected by Qwen2.5-VL
	messages = []
	for i in range(len(questions)):
		# Flatten the words for this example
		if i < len(all_words):
			words_text = " ".join(all_words[i])
		else:
			words_text = ""
			
		# Create user content with text
		user_content = [{
			"type": "text",
			"text": "question: " + questions[i] +
					" Directly provide only a short answer to the question. " +
					"Context: " + words_text
		}]
		
		# Add images to content if available
		if i < len(resized_images):
			for img in resized_images[i]:
				user_content.append({
					"type": "image",
					"image": img,
				})
		
		# Create the message
		message = [{
			"role": "user",
			"content": user_content
		}]
		messages.append(message)
	
	# Apply chat template
	texts = [
		processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
		for msg in messages
	]
	
	# Process vision info
	from qwen_vl_utils import process_vision_info
	image_inputs, _ = process_vision_info(messages)
	
	# Process inputs
	inputs = processor(
		text=texts,
		images=image_inputs,
		videos=None,
		padding=True,
		return_tensors="pt",
		padding_side="left",
	)
	
	# If answers are provided, prepare labels for training
	if answers and len(answers) > 0:
		answer_texts = [f"assistant: {ans}" for ans in answers]
		
		# Create combined input-output texts
		combined_texts = []
		for i in range(len(questions)):
			if i < len(answers):
				combined_text = texts[i] + " " + answer_texts[i]
				combined_texts.append(combined_text)
			else:
				# Handle cases where we might have more questions than answers
				combined_texts.append(texts[i])
		
		# Process the combined texts
		combined_inputs = processor(
			text=combined_texts,
			return_tensors="pt",
			padding=True,
			truncation=True
		)
		
		# Create labels with -100 for input tokens
		input_ids = combined_inputs.input_ids
		labels = input_ids.clone()
		
		# Find where the assistant part begins for each example
		for i in range(len(combined_texts)):
			# Find the position where assistant response begins
			assistant_token = processor.tokenizer.convert_tokens_to_ids(
				processor.tokenizer.tokenize("assistant:")[0]
			)
			assistant_pos = (input_ids[i] == assistant_token).nonzero(as_tuple=True)[0]
			
			if len(assistant_pos) > 0:
				# Mask input tokens with -100
				labels[i, :assistant_pos[0]] = -100
			else:
				# If no assistant token is found, mask all tokens
				labels[i, :] = -100
		
		# Return the combined inputs with labels
		return {
			"input_ids": combined_inputs.input_ids,
			"attention_mask": combined_inputs.attention_mask,
			"labels": labels
		}
	
	# For inference, just return the inputs without labels
	return inputs

class GarbageCollectionCallback(TrainerCallback):
	def on_step_end(self, args, state, control, **kwargs):
		gc.collect()
		torch.cuda.empty_cache()

def train_lora(
		model: QwenVLForConditionalGeneration,
		processor: AutoProcessor,
		dataset: Dataset,
		config: dict,
		filename: str,
		args: argparse.Namespace,
	):
	save_dir = config["checkpoint_dir"]

	peft_config = LoraConfig(
		lora_alpha=16,
		lora_dropout=0.05,
		r=8,
		bias="none",
		target_modules=["q_proj", "v_proj"],
		task_type="CAUSAL_LM",
		modules_to_save=[]
	)
	model = get_peft_model(model, peft_config)

	dataset = QwenDatasetWrapper(
		dataset,
		use_layout_labels=config.get("use_layout_labels", "Default"),
		add_sep_token=config.get("add_sep_token", False)
	)

	training_args = SFTConfig(
		output_dir=os.path.join(save_dir, filename),
		max_steps=args.max_steps,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.eval_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		gradient_checkpointing=True if args.gradient_checkpointing else False,
		# Optimizer and scheduler settings
		# optim="adamw_torch_fused",
		optim="adamw_8bit",
		learning_rate=args.learning_rate,
		lr_scheduler_type="constant",
		# Logging and evaluation
		logging_steps=10,
		eval_steps=args.eval_steps,
		eval_strategy="no",
		save_strategy="steps",
		save_steps=args.eval_steps,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		load_best_model_at_end=False,
		# Mixed precision and gradient settings
		bf16=True,
		bf16_full_eval=True,
		tf32=True,
		max_grad_norm=0.3, 
		warmup_ratio=0.05,
		# Hub and reporting
		push_to_hub=False,
		report_to="wandb" if args.wandb else 'none',
		# Gradient checkpointing settings
		gradient_checkpointing_kwargs={"use_reentrant": False},
		# Dataset configuration
		dataset_kwargs={"skip_prepare_dataset": True},
		remove_unused_columns=False,
		dataloader_num_workers=8,
		dataloader_persistent_workers=True,
		dataloader_drop_last=True,
		# Seed
		seed=args.seed
	)

	# Create a data collator that uses the model to prepare inputs
	data_collator = lambda batch: qwen_collate_fn(batch, processor)

	trainer = SFTTrainer(
		model=model,
		args=training_args,
		train_dataset=dataset,
		eval_dataset=None,
		data_collator=data_collator,
		peft_config=peft_config,
		callbacks=[GarbageCollectionCallback()]
	)
	trainer.train()
	trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
	args = {
		"use_RAG": False,
		"model": "RAGVT5",
		"dataset": "MP-DocVQA",
		"embed_model": "BGE",
		"reranker_model": "BGE",
		"page_retrieval": "Concat",
		"add_sep_token": False,
		"batch_size": 1,
		"layout_batch_size": 1,
		"chunk_num": 20,
		"chunk_size": 60,
		"chunk_size_tol": 0.2,
		"overlap": 10,
		"include_surroundings": 0,
		"model_weights": "Qwen/Qwen2.5-VL-7B-Instruct",
		"embed_weights": "/data/users/elopez/models/bge-finetuned/checkpoint-820",
		"reorder_chunks": False,
		"reranker_weights": "BAAI/bge-reranker-v2-m3",
		"rerank_filter_tresh": 0,
		"rerank_max_chunk_num": 10,
		"rerank_min_chunk_num": 1
	}
	extra_args = {
		"visible_devices": "0",
		"device": "cuda:0",
		"save_folder": "9-train_generator_with_layout",
		"save_name_append": "train_generator",
		"val_size": 1.0,
		"log_wandb": True,
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
	filename = f"{config['model_name']}_lora_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

	config["page_retrieval"] = config["page_retrieval"].lower()
	print("Building model...")
	qwen_config = Qwen2_5_VLConfig.from_pretrained(
		config["model_weights"],
		cache_dir=config["cache_dir"],
		torch_dtype=torch.bfloat16,
		attn_implementation="flash_attention_2",
	)
	model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
		config["model_weights"],
		config=qwen_config,
		torch_dtype=torch.bfloat16,
		device_map=config["device"],
		offload_folder="offload",
		offload_state_dict=True,
		cache_dir=config["cache_dir"],
	)
	processor = AutoProcessor.from_pretrained(config["model_weights"], cache_dir=config["cache_dir"])

	print("Building dataset...")
	dataset = build_dataset(config=config, split="val")

	args = {
		"max_steps": 1000,
		"gradient_checkpointing": True,
		"gradient_accumulation_steps": 32,
		"learning_rate": 2e-4,
		"eval_steps": 100,
		"wandb": config["log_wandb"],
		"seed": 42,
		"batch_size": config["batch_size"],
		"eval_batch_size": config["batch_size"],
	}
	args = argparse.Namespace(**args)

	train_lora(
		model=model,
		processor=processor,
		dataset=dataset,
		config=config,
		filename=filename,
		args=args
	)
