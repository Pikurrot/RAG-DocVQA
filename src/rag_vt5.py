import torch
from src.VT5 import VT5ForConditionalGeneration
from src ._modules import CustomT5Config

class RAGVT5(torch.nn.Module):
	def __init__(
			self,
			config: dict,
	):
		super(RAGVT5, self).__init__()

		# Load config
		self.save_dir = config.get("save_dir", "save/")
		self.batch_size = config.get("batch_size", 16)
		self.model_path = config.get("model_weights", "rubentito/vt5-base-spdocvqa")
		self.page_retrieval = config["page_retrieval"].lower() if "page_retrieval" in config else None
		self.max_source_length = config.get("max_source_length", 512)
		self.device = config.get("device", "cuda")

		t5_config = CustomT5Config.from_pretrained(self.model_path, ignore_mismatched_sizes=True)
		t5_config.visual_module_config = config.get("visual_module", {})

		# Load models
		self.generator = VT5ForConditionalGeneration.from_pretrained(
			self.model_path, config=t5_config, ignore_mismatched_sizes=True
		)
		self.generator.load_config(config)

	def to(self, device):
		self.device = device
		self.generator.to(device)

	def eval(self):
		self.generator.eval()

	def train(self):
		self.generator.train()

	def forward(self, batch: dict, return_pred_answer: bool=True):
		return self.generator(batch, return_pred_answer=return_pred_answer)
		