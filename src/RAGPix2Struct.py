import torch
import gc
import numpy as np
import os
from src._modules import (
	ImageChunker,
	ImageEncoder,
	VisualRetriever,
	LayoutModel,
	get_layout_model_map
)
from typing import Any
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from transformers.models.pix2struct.image_processing_pix2struct import render_text
from transformers.image_processing_base import BatchFeature
from src.custom_pix2struct_processor import CustomPix2StructProcessor, CustomPix2StructImageProcessor
from src._model_utils import get_generative_confidence
from PIL import Image

class RAGPix2Struct(torch.nn.Module):
	def __init__(self, config: dict):
		super(RAGPix2Struct, self).__init__()
		# Load config
		self.use_RAG = config.get("use_RAG", True)
		self.model_path = config.get("model_weights", "google/pix2struct-docvqa-base")
		self.layout_bs = config.get("layout_batch_size", 1)
		self.use_precomputed_layouts = config.get("use_precomputed_layouts", False)
		self.use_layout_labels = config.get("use_layout_labels", "Default")
		self.chunk_mode = config.get("chunk_mode", "horizontal")
		self.layout_map = get_layout_model_map(config)
		print(f"Loading model from {self.model_path}")
		if self.use_RAG:
			if self.use_precomputed_layouts or not self.use_RAG:
				self.layout_model_weights = None
			else:
				self.layout_model_weights = config.get("layout_model_weights", None)
			if self.layout_model_weights:
				print(f"Loading layout model from {self.layout_model_weights}")
			elif self.use_precomputed_layouts:
				print("Using precomputed layouts")
			else:
				print("Not using layout information")
		else:
			self.layout_model_weights = None
			print("Not using RAG, only model by default")
		self.device = config.get("device", "cuda")
		self.cache_dir = config.get("cache_dir", None)
		print(f"Using {self.cache_dir} as cache folder")

		self.generator = Pix2StructForConditionalGeneration.from_pretrained(
			self.model_path,
			cache_dir=self.cache_dir
		)
		if self.model_path.endswith(".ckpt") and "preprocessor_config.json" in os.listdir(self.model_path):
			preprocessor_path = self.model_path
		else:
			preprocessor_path = "google/pix2struct-docvqa-base"
		self.default_processor = Pix2StructProcessor.from_pretrained(
			preprocessor_path,
			cache_dir=self.cache_dir
		)
		self.image_processor = CustomPix2StructImageProcessor(
			do_convert_rgb=True,
			do_normalize=True,
			patch_size={"height": 16, "width": 16},
			max_total_patches=2048,
			is_vqa=True
		)
		self.processor = CustomPix2StructProcessor(
			self.image_processor,
			self.default_processor.tokenizer
		)

		# Load components
		if self.use_RAG:
			if self.layout_model_weights:
				self.layout_model = LayoutModel(config)
			else:
				self.layout_model = None
			self.chunker = ImageChunker(config)
			self.embedder = ImageEncoder(config, self.generator.get_encoder())
			self.retriever = VisualRetriever(config)

	
	def to(self, device: Any):
		self.device = device
		try:
			self.generator.to(device)
		except ValueError:
			pass
		if self.use_RAG:
			if self.layout_model is not None:
				self.layout_model.to(device)
	
	def eval(self):
		self.generator.eval()
		self.train_mode = False

	def train(self):
		self.generator.train()
		self.train_mode = True

	def online_retrieve(
			self,
			batch: dict,
			return_steps: bool = False
	) -> tuple:
		questions = batch["questions"] # (bs, )
		images = batch["images"] # (bs, n_pages) PIL images
		steps = {}

		# Get layout boxes and labels
		if self.layout_model:
			with torch.no_grad():
				layout_info, layout_steps = self.layout_model.batch_forward(
					images,
					return_steps=True,
					question_id=batch["question_id"]
				)
		elif self.use_precomputed_layouts:
			layout_info = batch["layouts"] # (bs, n_pages)
			layout_steps = {
				"layout_segments": [[]],
				"layout_info_raw": [[]]
			}
		else:
			layout_info = [[]]
			layout_steps = {
				"layout_segments": [[]],
				"layout_info_raw": [[]]
			}
		
		# Get chunks
		(
			patches_flatten, # (bs, n_pages*n_boxes*n_patches)
			patches_flatten_indices, # (bs, n_pages*n_boxes*n_patches)
			patches_matrix_list # (bs, n_pages*n_boxes, n_patches)
		) =\
			self.chunker.get_chunks(
				images,
				layout_info
		)

		# Get text and question embeddings
		question_images = [render_text(q) for q in questions]
		with torch.no_grad():
			patch_embeddings = self.embedder.batch_forward(patches_flatten) # (bs, n_pages*n_boxes*n_patches, seq_len, dim)
			question_embeddings = self.embedder.forward(question_images) # (bs, seq_len, dim)

		# Get top k chunks and boxes
		(
			top_k_patches # (bs, k)
		) =\
			self.retriever.retrieve(
				patch_embeddings,
				question_embeddings,
				patches_flatten_indices,
				patches_matrix_list
			)
		
		# Prepare output
		if return_steps:
			steps = {
				"layout_info": layout_info, # (bs, n_pages)
				"layout_segments": layout_steps["layout_segments"], # (bs, n_pages)
				"layout_info_raw": layout_steps["layout_info_raw"] # (bs, n_pages)
			}
			steps.update(layout_steps)
		else:
			steps = {}

		return (
			top_k_patches, # (bs, k)
			steps
		)
	
	def forward(
			self,
			batch: dict,
			return_pred_answer: bool = True,
			return_retrieval: bool = True,
			return_steps: bool = False,
			**kwargs
	) -> dict:
		# Retrieve top k patches
		if self.use_RAG:
			with torch.no_grad():
				(
					top_k_patches,
					steps
				) = self.online_retrieve(batch, return_steps=return_steps)
			gc.collect()
			torch.cuda.empty_cache()
		else:
			top_k_patches = batch["images"]
			steps = {"layout_info": [[]], "layout_segments": [[]], "layout_info_raw": [[]]}
		top_k_layout_labels = [[1] * len(patches) for patches in top_k_patches]

		# Generate
		bs = len(top_k_patches)
		if self.use_RAG:
			inputs = []
			for b in range(bs):
				question = batch["questions"][b]
				prompt = question# + " Provide the answer in few words"
				input_patches = top_k_patches[b]
				if len(input_patches) == 0:
					# input_patches = [Image.fromarray(np.random.randint(0,255,size=(2,2,3), dtype=np.uint8))]
					input_patches = batch["images"][b]
				batch_inputs = self.processor(images=input_patches, text=prompt)
				batch_inputs = {key:torch.from_numpy(value).to(self.device).unsqueeze(0) for key, value in batch_inputs.items() if key in ("flattened_patches", "attention_mask")}
				inputs.append(batch_inputs)

			inputs_flat = {}
			for inputs_dict in inputs:
				for key, value in inputs_dict.items():
					if key not in inputs_flat:
						inputs_flat[key] = value
					else:
						inputs_flat[key] = torch.cat((inputs_flat[key], value), dim=0)
			inputs_flat = BatchFeature(data=inputs_flat, tensor_type="pt").to(self.device)

			if "answers" in batch and self.train_mode:
				# Take first answer as target
				target_texts = [answers[0] for answers in batch["answers"]]
				# Tokenize targets
				# ----- Equivalent targets ------
				# processor = AutoProcessor.from_pretrained("google/pix2struct-base")
				# targets = processor(text=target_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
				# -------------------------------
				with self.processor.tokenizer.as_target_tokenizer():
					labels = self.processor.tokenizer(
						target_texts,
						padding=True,
						truncation=True,
						add_special_tokens=True,
						return_tensors="pt"
					)["input_ids"].to(self.device)
					labels[labels == self.processor.tokenizer.pad_token_id] = -100
				# Add labels to inputs
				inputs_flat["labels"] = labels

				# Forward pass through the model
				outputs = self.generator(**inputs_flat)

				if return_pred_answer:
					# logits = outputs.logits
					# pred_token_ids = torch.argmax(logits, dim=-1)
					# pred_answer = self.processor.batch_decode(pred_token_ids, skip_special_tokens=True)
					with torch.no_grad():
						output_ids = self.generator.generate(**inputs_flat)
						pred_answer = self.processor.batch_decode(output_ids, skip_special_tokens=True)
				else:
					pred_answer = None
			else:
				outputs = None
				with torch.no_grad():
					output_ids = self.generator.generate(**inputs_flat)
					pred_answer = self.processor.batch_decode(output_ids, skip_special_tokens=True)
		else:
			# Generate for each page, then take the one with the highest confidence
			output_ids = []
			for b in range(bs):
				question = batch["questions"][b]
				prompt = question
				input_patches = batch["images"][b]
				batch_inputs = self.default_processor(images=input_patches, text=prompt, return_tensors="pt").to(self.device)
				outputs = self.generator.generate(**batch_inputs, output_scores=True, return_dict_in_generate=True)
				confidences = torch.tensor(get_generative_confidence(outputs))
				best_page = torch.argmax(confidences)
				output_ids.append(outputs.sequences[best_page])
			# pad the output ids
			max_length = max([len(ids) for ids in output_ids])
			for i in range(len(output_ids)):
				if len(output_ids[i]) < max_length:
					output_ids[i] = torch.cat((output_ids[i], torch.full((max_length - len(output_ids[i]),), self.default_processor.tokenizer.pad_token_id).to(self.device)))
			output_ids = torch.stack(output_ids).to(self.device)
			outputs = None
			pred_answer = self.processor.batch_decode(output_ids, skip_special_tokens=True)

		result = (outputs, pred_answer, None, None)

		if return_retrieval:
			retrieval = {
				"patches": top_k_patches,
				"steps": steps,
				"top_k_layout_labels": top_k_layout_labels
			}
		else:
			retrieval = {}

		return *result, retrieval

	@torch.inference_mode()
	def inference(
			self,
			batch: dict,
			**kwargs
	):
		self.eval()
		return self.forward(
			batch,
			**kwargs
		)