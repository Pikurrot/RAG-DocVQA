import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Config
from transformers import AutoFeatureExtractor, AutoModel
from torch.nn import CrossEntropyLoss
from torch.nn import LayerNorm as BertLayerNorm


class CustomT5Config(T5Config):
	def __init__(self, max_2d_position_embeddings=1024,  **kwargs):
		super().__init__(**kwargs)
		self.max_2d_position_embeddings = max_2d_position_embeddings
		self.hidden_dropout_prob = 0.1
		self.layer_norm_eps = 1e-12


class SpatialEmbeddings(nn.Module):
	"""
	Spatial embedding by summing x, y, w, h projected by nn.Embedding to hidden size.
	"""

	def __init__(self, config):
		super(SpatialEmbeddings, self).__init__()

		self.x_position_embeddings = nn.Embedding(
			config.max_2d_position_embeddings, config.hidden_size
		)
		self.y_position_embeddings = nn.Embedding(
			config.max_2d_position_embeddings, config.hidden_size
		)

		self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

		self.spatial_emb_matcher = MLP(config.hidden_size, 0, config.hidden_size, 1)

		self.config = config

	def forward(self, bbox):
		left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
		upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
		right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
		lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

		embeddings = (
				left_position_embeddings
				+ upper_position_embeddings
				+ right_position_embeddings
				+ lower_position_embeddings
		)

		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		embeddings = self.spatial_emb_matcher(embeddings)
		return embeddings


class MLP(nn.Module):
	""" Very simple multi-layer perceptron (also called FFN)"""

	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		self.num_layers = num_layers
		h = [hidden_dim] * (num_layers - 1)
		self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
		return x


class VisualEmbeddings(nn.Module):

	def __init__(self, config):
		super(VisualEmbeddings, self).__init__()

		self.feature_extractor = AutoFeatureExtractor.from_pretrained(
			config.visual_module_config['model_weights'],
			ignore_mismatched_sizes=True
		)
		self.image_model = AutoModel.from_pretrained(
			config.visual_module_config['model_weights'],
			ignore_mismatched_sizes=True
		)
		self.visual_emb_matcher = MLP(self.image_model.config.hidden_size, 0, self.image_model.config.hidden_size, 1)

		if not config.visual_module_config.get('finetune', False):
			self.freeze()

	def freeze(self):
		for p in self.image_model.parameters():
			p.requires_grad = False

	def get_visual_boxes(self, num_pages=1, scale=1):
		boxes = torch.tensor([[0, 0, 1, 1]] + [[x / 14, y / 14, (x + 1) / 14, (y + 1) / 14] for y in range(0, 14) for x in range(0, 14)], dtype=torch.float32)
		boxes = boxes.unsqueeze(dim=0).expand([num_pages, -1, -1])
		boxes = boxes * scale
		return boxes

	def forward(self, images, page_idx_mask=None):
		inputs = self.feature_extractor(images=images, return_tensors="pt")
		vis_embeddings = self.image_model(inputs.pixel_values.to(self.image_model.device))
		vis_embeddings = vis_embeddings.last_hidden_state  # BS; 14x14+CLS (197); 768 (hidden size)
		vis_embeddings = self.visual_emb_matcher(vis_embeddings)

		if page_idx_mask is not None:
			vis_attention_mask = torch.zeros(vis_embeddings.shape[:2], dtype=torch.long).to(self.image_model.device)
			vis_attention_mask[page_idx_mask] = 1
		else:
			vis_attention_mask = torch.ones(vis_embeddings.shape[:2], dtype=torch.long).to(self.image_model.device)

		return vis_embeddings, vis_attention_mask


class Chunker:
	pass

class ChunkEmbedder:
	pass


class Retriever:
	pass
