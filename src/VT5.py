import numpy as np
import random
import torch
import os
from transformers import PreTrainedModel, T5Tokenizer
from src.LayoutT5 import LayoutT5ForConditionalGeneration
from src._modules import SpatialEmbeddings, VisualEmbeddings, CustomT5Config
from src._model_utils import shift_tokens_right, get_generative_confidence
from typing import Any, Tuple, Optional
from safetensors.torch import load_file

class VT5ForConditionalGeneration(PreTrainedModel):
	config_class = CustomT5Config

	def __init__(self, config: CustomT5Config):
		super().__init__(config)
		self.tokenizer = T5Tokenizer.from_pretrained(config._name_or_path, ignore_mismatched_sizes=True)
		self.language_backbone = LayoutT5ForConditionalGeneration(config)
		self.spatial_embedding = SpatialEmbeddings(config)
		self.visual_embedding = VisualEmbeddings(config)
		self.layout_embedding = torch.nn.Embedding(12, self.language_backbone.model_dim)
		self.layout_embedding_scale = torch.nn.Parameter(torch.tensor(1.0))

	def post_init(self):
		"""Initialize weights after model is loaded"""
		super().post_init()
		with torch.no_grad():
			torch.nn.init.xavier_normal_(self.language_backbone.layout_classifier.weight, gain=1.0)
			torch.nn.init.zeros_(self.language_backbone.layout_classifier.bias)

	@classmethod
	def from_pretrained(cls, model_path: str, **kwargs):
		print(f"Loading model from {model_path}")
		if model_path == "rubentito/vt5-base-spdocvqa":
			model = super(VT5ForConditionalGeneration, cls).from_pretrained(model_path, **kwargs)
		else:
			safetensors_path = os.path.join(model_path, "model.safetensors")
			config = kwargs.get("config", None)
			model = cls(config)
			model.load_state_dict(load_file(safetensors_path), strict=False)
		# Initialize weights and apply final processing
		model.post_init()
		return model

	def load_config(self, config: dict):
		# Load extra config
		self.save_dir = config.get("save_dir", "save/")
		self.batch_size = config.get("batch_size", 16)
		self.model_path = config.get("model_weights", "rubentito/vt5-base-spdocvqa")
		self.page_retrieval = config.get("page_retrieval", None)
		self.max_source_length = config.get("max_source_length", 512)
		self.use_layout_labels = config.get("use_layout_labels", False) and self.page_retrieval != "oracle"
		if not config["train_language_backbone"]:
			# set requires_grad to False for all parameters
			for param in self.language_backbone.parameters():
				param.requires_grad = False
		if not config["train_spatial_embedding"]:
			for param in self.spatial_embedding.parameters():
				param.requires_grad = False
		if not config["train_visual_embedding"]:
			for param in self.visual_embedding.parameters():
				param.requires_grad = False
		if not config["train_layout_embedding"]:
			for param in self.layout_embedding.parameters():
				param.requires_grad = False
		self.layout_embedding_scale.data.fill_(float(config.get("layout_embedding_scale", 1.0)))

	def to(self, device: Any):
		self.language_backbone.to(device)
		self.spatial_embedding.to(device)
		self.visual_embedding.to(device)
		self.layout_embedding.to(device)

	def prepare_inputs_for_vqa(
			self,
			question: list, # (bs,)
			words: list, # (bs, n_words)
			boxes: list, # (bs, n_words, 4)
			layout_labels: list, # (bs, n_words)
			images: list, # (bs,) PIL images
			answers: Optional[list]=None, # (bs, n_answers)
			return_ids: bool=False # 
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Prepare inputs for the model
			:param question: list of questions
			:param words: list of lists of words
			:param boxes: list of lists of boxes
			:param layout_labels: list of lists of layout labels
			:param images: list of images
			:param answers: list of lists of answers
			:param return_ids: if True, return the input_ids instead of the embeddings
			:return: input_embeds, attention_mask, labels
		"""
		bs = len(words)
		prompt_text = ["question: {:s}  context: ".format(q) for q in question]
		prompt_box = [0, 0, 1000, 1000]
		eos_box = [0, 0, 0, 0]
		padding_box_value = 0  # To become [0, 0, 0, 0] array.
		prompt_layout_value = 0
		eos_layout_value = 0
		padding_layout_value = 0

		# Get input_ids, boxes, layout_labels and attention_mask
		longest_seq = 0
		batch_input_ids = []
		batch_input_boxes = []
		batch_input_layout_labels = []
		for batch_idx in range(bs):
			tokenized_prompt = self.tokenizer(prompt_text[batch_idx])
			prompt_token_ids = tokenized_prompt.input_ids[:-1] # Exclude EOS
			prompt_layout = [prompt_layout_value] * len(prompt_token_ids)
			
			input_ids = prompt_token_ids.copy() # (n_tokens,)
			input_boxes = [prompt_box] * len(prompt_token_ids) # (n_tokens, 4)
			input_layout_labels = prompt_layout.copy()

			for i in range(len(words[batch_idx])):
				word = words[batch_idx][i]
				box = boxes[batch_idx][i]
				if layout_labels:
					layout_label = layout_labels[batch_idx][i]
				tokenized_word = self.tokenizer(word).input_ids[:-1] # Tokenize the word and ignore eos_token
				input_ids.extend(tokenized_word)
				input_boxes.extend((np.array([box]*len(tokenized_word))*1000).tolist())  # Repeat the box for each token corresponding to the word.
				if layout_labels:
					input_layout_labels.extend([layout_label] * len(tokenized_word))

			batch_input_ids.append(input_ids[:self.max_source_length-1] + [self.tokenizer.eos_token_id])  # Append the eos_token at the end.
			batch_input_boxes.append(np.concatenate([input_boxes[:self.max_source_length-1],  np.array([eos_box])]))  # Append a bounding box corresponding to the eos_token.
			if layout_labels:
				batch_input_layout_labels.append(input_layout_labels[:self.max_source_length-1] + [eos_layout_value])  # Append the layout label for the eos_token.
			longest_seq = min(max(longest_seq, len(input_ids) + 1), self.max_source_length)

		# Convert to tensors and pad. Actually, a pad tensor is created and it"s filled with corresponding values.
		tensor_input_ids = torch.full([bs, longest_seq], fill_value=self.tokenizer.pad_token_id, dtype=torch.long)
		tensor_boxes = torch.full([bs, longest_seq, 4],  fill_value=padding_box_value, dtype=torch.long)
		if layout_labels:
			tensor_layout_labels = torch.full([bs, longest_seq], fill_value=padding_layout_value, dtype=torch.long)
		tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

		for batch_idx in range(bs):
			seq_len = len(batch_input_ids[batch_idx])
			tensor_input_ids[batch_idx, :seq_len] = torch.LongTensor(batch_input_ids[batch_idx])
			tensor_boxes[batch_idx, :seq_len] = torch.from_numpy(batch_input_boxes[batch_idx][:seq_len])
			if layout_labels:
				tensor_layout_labels[batch_idx, :seq_len] = torch.LongTensor(batch_input_layout_labels[batch_idx])
			tensor_attention_mask[batch_idx, :seq_len] = 1

		# Send everything to GPU
		tensor_input_ids = tensor_input_ids.to(self.language_backbone.device) # (bs, longest_seq)
		tensor_boxes = tensor_boxes.to(self.language_backbone.device) # (bs, longest_seq, 4)
		if layout_labels:
			tensor_layout_labels = tensor_layout_labels.to(self.language_backbone.device) # (bs, longest_seq)
		tensor_attention_mask = tensor_attention_mask.to(self.language_backbone.device) # (bs, longest_seq)

		# Get semantic, spatial and layout embeddings
		semantic_embedding = self.language_backbone.shared(tensor_input_ids) # (bs, longest_seq, dim)
		spatial_embedding = self.spatial_embedding(tensor_boxes) # (bs, longest_seq, dim)
		if layout_labels:
			layout_embedding = self.layout_embedding(tensor_layout_labels) # (bs, longest_seq, dim)
		visual_embedding, visual_emb_mask = self.visual_embedding(images) # (bs, n_visual_tokens, dim), (bs, n_visual_tokens) n_visual_tokens = 14x14+CLS = 197

		# Sum and concatenate embeddings
		input_embeds = semantic_embedding + spatial_embedding # (bs, longest_seq, dim)
		if layout_labels:
			input_embeds = input_embeds + layout_embedding * self.layout_embedding_scale
		input_embeds = torch.cat([input_embeds, visual_embedding], dim=1) # (bs, longest_seq + n_visual_tokens, dim)
		tensor_attention_mask = torch.cat([tensor_attention_mask, visual_emb_mask], dim=1) # (bs, longest_seq + n_visual_tokens)

		# Tokenize answers
		if answers is not None:
			answers = [random.choice(answer) for answer in answers]
			labels = self.tokenizer(answers, return_tensors="pt", padding=True)
			labels.input_ids[labels.input_ids[:] == self.tokenizer.pad_token_id] = -100
			labels = labels.input_ids.to(self.language_backbone.device)
		else:
			labels = None

		to_return = (tensor_attention_mask, labels)
		if return_ids:
			to_return = (tensor_input_ids, *to_return)
		else:
			to_return = (input_embeds, *to_return)
		if layout_labels:
			to_return = (*to_return, tensor_layout_labels)
		return to_return

	def forward(self, batch: dict, return_pred_answer: bool=False):
		question = batch["questions"]
		words = batch["words"]
		boxes = batch["boxes"]
		if self.use_layout_labels:
			layout_labels = batch["layout_labels"]
		else:
			layout_labels = None
		images = batch["images"]
		answers = batch.get("answers", None)

		input_embeds, attention_mask, labels, tensor_layout_labels = self.prepare_inputs_for_vqa(
			question, words, boxes, layout_labels, images, answers
		)

		if labels is not None:
			decoder_input_ids = shift_tokens_right(
				labels,
				pad_token_id=self.tokenizer.pad_token_id,
				decoder_start_token_id=self.language_backbone.config.decoder_start_token_id
			)
			decoder_inputs_embeds = self.language_backbone.shared(decoder_input_ids)
			
			outputs = self.language_backbone(
				inputs_embeds=input_embeds,
				decoder_inputs_embeds=decoder_inputs_embeds,
				attention_mask=attention_mask,
				labels=labels,
				layout_labels=tensor_layout_labels
			)
			if return_pred_answer:
				pred_answers, pred_answers_conf = self.get_answer_from_model_output(input_embeds, attention_mask)
			else:
				pred_answers, pred_answers_conf = None, None
			pred_answer_pages = None
		else:
			outputs = None
			pred_answers, pred_answers_conf = self.get_answer_from_model_output(input_embeds, attention_mask)

			pred_answer_pages = None

		return outputs, pred_answers, pred_answer_pages, pred_answers_conf

	def get_answer_from_model_output(
			self,
			input_embeds: torch.Tensor,
			attention_mask: torch.Tensor
	) -> Tuple[list, list]:
		with torch.no_grad():
			output = self.language_backbone.generate(
				inputs_embeds=input_embeds,
				attention_mask=attention_mask,
				output_scores=True,
				return_dict_in_generate=True,
				output_attentions=False,
				max_new_tokens=100,
			)
		pred_answers = self.tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
		pred_answers_conf = get_generative_confidence(output)

		return pred_answers, pred_answers_conf
