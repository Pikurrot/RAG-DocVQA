import torch
import os
from transformers import (
	Qwen2_5_VLForConditionalGeneration,
	Qwen2_5_VLConfig,
	AutoProcessor
)

class QwenVLForConditionalGeneration(torch.nn.Module):
	def __init__(self, model_path: str, config: Qwen2_5_VLConfig):
		super(QwenVLForConditionalGeneration, self).__init__()
		self.processor = AutoProcessor.from_pretrained(model_path)
		self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
			model_path,
			config=config,
			cache_dir=config.cache_dir,
			torch_dtype=config.torch_dtype,
			attn_implementation=config._attn_implementation,
   			device_map=config.device,
		)
		self.device = config.device
		self.max_seq_lenght = 131072

	def to(self, device):
		self.model.to(device)
		self.device = device

	def eval(self):
		self.model.eval()

	def train(self):
		self.model.train()

	def prepare_inputs_for_vqa(
			self,
			question: list, # (bs,)
			words: list, # (bs, n_words)
			images: list # (bs, k) PIL images
	) -> dict:
		messages = [[
			{
				"role": "user",
				"content": [
					{
						"type": "image",
						"image": img,
					} for img in images[i]
				] + [
					{
						"type": "text",
						"text": "question: " + question[i] +
								" context: " + " ".join(words[i]) + 
								". Directly provide only a short answer to the question."
					}
				]
			}
			for i in range(len(question))
		]]
		texts = [
			self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
			for msg in messages
		]

		inputs = self.processor(
			text=texts,
			images=images,
			videos=None,
			padding=True,
			return_tensors="pt",
			padding_side="left",
			max_lenght=self.max_seq_lenght,
			truncation=True
		)
		inputs = inputs.to(self.device)
		return inputs

	def forward(self, batch: dict, return_pred_answer: bool = False):
		question = batch["questions"]
		words = batch["words"]
		images = batch["images"]

		inputs = self.prepare_inputs_for_vqa(question, words, images)

		generated_ids = self.model.generate(**inputs, max_new_tokens=16)
		generated_ids_trimmed = [
			out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
		]
		pred_answers = self.processor.batch_decode(
			generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
		)
		pred_answer_pages = [0] * len(pred_answers)
		pred_answers_conf = [1.0] * len(pred_answers)
		return None, pred_answers, pred_answer_pages, pred_answers_conf
