import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from torch.nn import CrossEntropyLoss
from transformers.utils import (
	add_start_docstrings_to_model_forward,
	replace_return_docstrings,
)
from typing import Optional, Tuple, Union
from transformers.models.t5.modeling_t5 import auto_docstring
import warnings
from dataclasses import dataclass

@dataclass
class LayoutSeq2SeqLMOutput(Seq2SeqLMOutput):
	lm_loss: Optional[torch.FloatTensor] = None
	layout_loss: Optional[torch.FloatTensor] = None

class LayoutT5ForConditionalGeneration(T5ForConditionalGeneration):
	def __init__(self, config: T5Config):
		super().__init__(config)
		self.layout_classifier = nn.Linear(config.d_model, 12)
		self.layout_norm = nn.LayerNorm(config.d_model)
		self.layout_loss_weight = getattr(config, "layout_loss_weight", 1.0)
		# print("layout classifier weight:", self.layout_classifier.weight.shape, self.layout_classifier.weight)
		# print("layout classifier bias:", self.layout_classifier.bias.shape, self.layout_classifier.bias)

	@auto_docstring
	def forward(
		self,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.FloatTensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.BoolTensor] = None,
		head_mask: Optional[torch.FloatTensor] = None,
		decoder_head_mask: Optional[torch.FloatTensor] = None,
		cross_attn_head_mask: Optional[torch.Tensor] = None,
		encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
		past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		layout_labels: Optional[torch.LongTensor] = None, # layout labels
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
		r"""
		labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
			config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
			labels in `[0, ..., config.vocab_size]`

		Returns:

		Examples:

		```python
		>>> from transformers import AutoTokenizer, T5ForConditionalGeneration

		>>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
		>>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

		>>> # training
		>>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
		>>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
		>>> outputs = model(input_ids=input_ids, labels=labels)
		>>> loss = outputs.loss
		>>> logits = outputs.logits

		>>> # inference
		>>> input_ids = tokenizer(
		...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
		... ).input_ids  # Batch size 1
		>>> outputs = model.generate(input_ids)
		>>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
		>>> # studies have shown that owning a dog is good for you.
		```"""
		use_cache = use_cache if use_cache is not None else self.config.use_cache
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
		if head_mask is not None and decoder_head_mask is None:
			if self.config.num_layers == self.config.num_decoder_layers:
				decoder_head_mask = head_mask

		# Encode if needed (training, first prediction pass)
		if encoder_outputs is None:
			# Convert encoder inputs in embeddings if needed
			encoder_outputs = self.encoder(
				input_ids=input_ids,
				attention_mask=attention_mask,
				inputs_embeds=inputs_embeds,
				head_mask=head_mask,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)
		elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
			encoder_outputs = BaseModelOutput(
				last_hidden_state=encoder_outputs[0],
				hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
				attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
			)

		hidden_states = encoder_outputs[0]

		layout_loss = None
		if layout_labels is not None:
			text_length = layout_labels.size(1)
			# Slice the encoder hidden states to only the text tokens (exclude visual tokens)
			text_hidden_states = hidden_states[:, :text_length, :]
			normalized_states = self.layout_norm(text_hidden_states)
			# print("Normalized states:", normalized_states.shape, normalized_states[0, 0, :])
			# print("layout classifier weight:", self.layout_classifier.weight.shape, self.layout_classifier.weight)
			# print("layout classifier bias:", self.layout_classifier.bias.shape, self.layout_classifier.bias)
			layout_logits = self.layout_classifier(normalized_states)  # shape: (bs, text_length, 12)
			# layout_logits = torch.clamp(layout_logits, -100, 100)
			loss_fct_layout = torch.nn.CrossEntropyLoss(ignore_index=-100)
			layout_loss = loss_fct_layout(
				layout_logits.view(-1, layout_logits.size(-1)),
				layout_labels.view(-1)
			)
			# print("Text length:", text_length)
			# print(text_hidden_states.shape, text_hidden_states[0, 0, :])
			# print("Layout logits:", layout_logits.shape, layout_logits[0, 0, :])
			# print("Layout loss:", layout_loss)

		if self.model_parallel:
			torch.cuda.set_device(self.decoder.first_device)

		if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
			# get decoder inputs from shifting lm labels to the right
			decoder_input_ids = self._shift_right(labels)

		# Set device for model parallelism
		if self.model_parallel:
			torch.cuda.set_device(self.decoder.first_device)
			hidden_states = hidden_states.to(self.decoder.first_device)
			if decoder_input_ids is not None:
				decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
			if attention_mask is not None:
				attention_mask = attention_mask.to(self.decoder.first_device)
			if decoder_attention_mask is not None:
				decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

		# Decode
		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			inputs_embeds=decoder_inputs_embeds,
			past_key_values=past_key_values,
			encoder_hidden_states=hidden_states,
			encoder_attention_mask=attention_mask,
			head_mask=decoder_head_mask,
			cross_attn_head_mask=cross_attn_head_mask,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		sequence_output = decoder_outputs[0]

		# Set device for model parallelism
		if self.model_parallel:
			torch.cuda.set_device(self.encoder.first_device)
			self.lm_head = self.lm_head.to(self.encoder.first_device)
			sequence_output = sequence_output.to(self.lm_head.weight.device)

		if self.config.tie_word_embeddings:
			# Rescale output before projecting on vocab
			# See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
			sequence_output = sequence_output * (self.model_dim**-0.5)

		lm_logits = self.lm_head(sequence_output)

		loss = None
		lm_loss = None
		if labels is not None:
			loss_fct = CrossEntropyLoss(ignore_index=-100)
			# move labels to correct device to enable PP
			labels = labels.to(lm_logits.device)
			lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
			loss = lm_loss
			# TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
			if layout_loss is not None:
				loss = loss + self.config.layout_loss_weight * layout_loss

		if not return_dict:
			output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
			return ((loss,) + output) if loss is not None else output

		return LayoutSeq2SeqLMOutput(
			loss=loss,
			lm_loss=lm_loss,
			layout_loss=layout_loss,
			logits=lm_logits,
			past_key_values=decoder_outputs.past_key_values,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			encoder_hidden_states=encoder_outputs.hidden_states,
			encoder_attentions=encoder_outputs.attentions,
		)
