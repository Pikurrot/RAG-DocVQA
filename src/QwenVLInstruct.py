import torch
from transformers import (
	Qwen2_5_VLForConditionalGeneration,
	AutoProcessor
)
from qwen_vl_utils import process_vision_info
from typing import Optional, Tuple

class QwenVLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
	def __init__(self, config):
		super().__init__(config)
		config_dict = config.to_dict()
		self.processor = AutoProcessor.from_pretrained(config_dict["model_weights"])

	def prepare_inputs_for_vqa(
			self,
			question: list, # (bs,)
			words: list, # (bs, n_words)
			images: list # (bs,) PIL images
	) -> dict:
		messages = [
			{
				"role": "user",
				"content": [
					{
						"type": "image",
						"image": "" # for special token
					},
					{
						"type": "text",
						"text": "question: " + question[i] +
								" context: " + " ".join(words[i]) + 
								". Directly provide only a short answer to the question."
					}
				]
			}
			for i in range(len(question))
		]
		texts = [
			self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
			for msg in messages
		]
		inputs = self.processor(
			text=texts,
			images=images,videos=None,
			padding=True,
			return_tensors="pt",
			padding_side="left"
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
