import torch
import numpy as np

def get_extractive_confidence(outputs: torch.Tensor) -> list:
	bs = len(outputs['start_logits'])
	start_idxs = torch.argmax(outputs.start_logits, axis=1)
	end_idxs = torch.argmax(outputs.end_logits, axis=1)

	answ_confidence = []
	for batch_idx in range(bs):
		conf_mat = np.matmul(np.expand_dims(outputs.start_logits.softmax(dim=1)[batch_idx].unsqueeze(dim=0).detach().cpu(), -1),
							 np.expand_dims(outputs.end_logits.softmax(dim=1)[batch_idx].unsqueeze(dim=0).detach().cpu(), 1)).squeeze(axis=0)

		answ_confidence.append(
			conf_mat[start_idxs[batch_idx], end_idxs[batch_idx]].item()
		)

	return answ_confidence

def get_generative_confidence(output: torch.Tensor) -> list:
	batch_logits = torch.stack(output.scores, dim=1)[:, :-1, :]  # b x s x V and dropping EOS token
	decoder_output_confs = torch.amax(batch_logits.softmax(-1), 2)
	confidences = decoder_output_confs.prod(1)  # b
	return confidences.tolist()

def shift_tokens_right(
		input_ids: torch.Tensor,
		pad_token_id: int,
		decoder_start_token_id: int
):
	"""
	Shift input ids one token to the right.
	"""
	shifted_input_ids = input_ids.new_zeros(input_ids.size())
	shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
	shifted_input_ids[:, 0] = decoder_start_token_id

	# Replace possible -100 values in labels by `pad_token_id`
	shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

	return shifted_input_ids

def torch_no_grad(func):
	def wrapper(*args, **kwargs):
		with torch.no_grad():
			return func(*args, **kwargs)
	return wrapper

def mean_pooling(
		embs: torch.Tensor,
		attention_mask: torch.Tensor
) -> torch.Tensor:
	"""
	Obtain sentence embeddings by mean pooling the embeddings of the tokens.
	"""
	attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embs.size())
	masked_embs = embs * attention_mask_expanded
	sum_embeddings = masked_embs.sum(dim=1)
	non_padding_counts = attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9)
	mean_embedding = sum_embeddings / non_padding_counts
	return mean_embedding
