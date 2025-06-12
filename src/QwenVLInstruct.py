import torch
import os
from transformers import (
	Qwen2_5_VLForConditionalGeneration,
	Qwen2_5_VLConfig,
	AutoProcessor
)
from qwen_vl_utils import process_vision_info
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Tuple, Optional
from peft import PeftModel

def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    """Shift token ids to the right and prepend decoder_start_token_id."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    
    if pad_token_id is not None:
        # Replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    
    return shifted_input_ids

def get_generative_confidence(output):
    """Extract confidence scores from the model output"""
    # Simplified implementation - you might need to adjust based on your needs
    probs = torch.exp(output.scores[0])
    top_probs = torch.max(probs, dim=-1).values
    confidences = [float(prob.mean()) for prob in top_probs]
    return confidences

class QwenVLForConditionalGeneration(torch.nn.Module):
	def __init__(self, model_path: str, config: Qwen2_5_VLConfig):
		super(QwenVLForConditionalGeneration, self).__init__()
		self.lora_weights = config.lora_weights
		self.processor = AutoProcessor.from_pretrained(model_path)
		self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
			model_path,
			config=config,
			cache_dir=config.cache_dir,
			torch_dtype=config.torch_dtype,
			attn_implementation=config._attn_implementation,
   			device_map=config.device,
		)
		if self.lora_weights:
			print(f"Loading LoRA weights from {self.lora_weights}")
			self.model = PeftModel.from_pretrained(
				self.model,
				self.lora_weights,
				torch_dtype=config.torch_dtype,
				device_map=config.device,
			)
			print("LoRA weights loaded successfully")

		self.device = config.device
		self.max_seq_lenght = 131072
		self.train_mode = False

	def to(self, device):
		self.model.to(device)
		self.device = device

	def eval(self):
		self.model.eval()
		self.train_mode = False

	def train(self):
		self.model.train()
		self.train_mode = True

	def prepare_inputs_for_vqa(
			self,
			question: list,  # (bs,)
			words: list,     # (bs, n_words)
			images: list,    # (bs, k) PIL images
			answers: Optional[list] = None  # (bs,) Optional ground truth answers
	) -> dict:
		resized_images = []
		for batch_imgs in images:
			batch_resized = []
			for img in batch_imgs:
				if img.width < 28 or img.height < 28:
					new_width = max(img.width, 28)
					new_height = max(img.height, 28)
					batch_resized.append(img.resize((new_width, new_height)))
				else:
					batch_resized.append(img)
			resized_images.append(batch_resized)

		# Construct messages for each sample in the batch
		messages = []
		for i in range(len(question)):
			message = {
				"role": "user",
				"content": [
					{
						"type": "text",
						"text": "question: " + question[i] +
								"\nDirectly provide only a short answer to the question. " +
								# "If the question cannot be answered with the provided data, respond with 'not answerable'. " +
								"Context: " + " ".join(words[i])
					}
				]
			}
			# Add images for this sample
			for img in resized_images[i]:
				message["content"].append({
					"type": "image",
					"image": img,
				})
			messages.append([message])

		texts = [
			self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
			for msg in messages
		]
		image_inputs, _ = process_vision_info(messages)

		# Process the input
		inputs = self.processor(
			text=texts,
			images=image_inputs,
			videos=None,
			padding=True,
			return_tensors="pt",
			padding_side="left",
		)
		
		# Move to device
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		
		# If answers are provided, prepare labels for training
		labels = None
		if answers:
			answer_texts = [f"assistant: {ans}" for ans in answers]
			
			# For training, we need to handle the labels differently
			# Create combined input-output texts for proper labeling
			combined_texts = []
			for i in range(len(question)):
				combined_text = texts[i] + " " + answer_texts[i]
				combined_texts.append(combined_text)
				
			# Process the combined texts
			combined_inputs = self.processor(
				text=combined_texts,
				return_tensors="pt",
				padding=True,
				truncation=True
			)
			
			# Create labels with -100 for input tokens (to mask them from loss calculation)
			input_ids = combined_inputs.input_ids.to(self.device)
			labels = input_ids.clone()
			
			# Find where the assistant part begins for each example
			for i in range(len(question)):
				# Find the position where assistant response begins
				assistant_pos = (input_ids[i] == self.processor.tokenizer.convert_tokens_to_ids(
					self.processor.tokenizer.tokenize("assistant:")[0]
				)).nonzero(as_tuple=True)[0]
				
				if len(assistant_pos) > 0:
					# Mask input tokens with -100
					labels[i, :assistant_pos[0]] = -100
			
			# Return both the original inputs for generation and the combined inputs+labels for training
			return inputs, {"input_ids": combined_inputs.input_ids.to(self.device), 
							"attention_mask": combined_inputs.attention_mask.to(self.device)}, labels
		
		return inputs, None, None

	def forward(self, batch: dict, return_pred_answer: bool = False):
		question = batch["questions"]
		words = batch["words"]
		images = batch["images"]
		answers = batch.get("answers", None) if self.train_mode else None

		inputs, combined_inputs, labels = self.prepare_inputs_for_vqa(question, words, images, answers)

		if labels is not None:
			# Training mode with labels
			outputs = self.model(
				**combined_inputs,
				labels=labels
			)

			if return_pred_answer:
				pred_answers, pred_answers_conf = self.get_answer_from_model_output(inputs)
			else:
				pred_answers, pred_answers_conf = None, None
			pred_answer_pages = None
		else:
			# Inference mode
			outputs = None
			pred_answers, pred_answers_conf = self.get_answer_from_model_output(inputs)
			pred_answer_pages = [0] * len(pred_answers)

		return outputs, pred_answers, pred_answer_pages, pred_answers_conf

	def get_answer_from_model_output(
			self,
			inputs: dict
	) -> Tuple[list, list]:
		with torch.no_grad():
			output = self.model.generate(
				**inputs,
				output_scores=True,
				return_dict_in_generate=True,
				output_attentions=False,
				max_new_tokens=16,
			)

		generated_ids_trimmed = [
			out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output.sequences)
		]

		pred_answers = self.processor.batch_decode(
			generated_ids_trimmed, 
			skip_special_tokens=True, 
			clean_up_tokenization_spaces=False
		)

		# Extract confidence scores
		pred_answers_conf = get_generative_confidence(output)

		return pred_answers, pred_answers_conf
