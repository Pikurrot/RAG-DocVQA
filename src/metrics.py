import editdistance
import numpy as np
from typing import List, Union
from src.utils import get_similarity_score
from src._modules import get_layout_model_map

class Evaluator:
	def __init__(
			self,
			config: dict,
			case_sensitive: bool=False
	):
		self.layout_map = get_layout_model_map(config)
		self.case_sensitive = case_sensitive
		self.get_edit_distance = editdistance.eval
		self.anls_threshold = 0.5

		self.best_accuracy = 0
		# self.best_anls = 0
		self.best_epoch = 0

	def get_metrics(
			self,
			gt_answers: List[List[str]],
			preds: Union[List[str], List[List[str]]],
			answer_types: List[str]=None,
			top_k_layout_labels: List[List[int]]=None
	) -> dict:
		if preds is None:
			return {"accuracy": [0]*len(gt_answers), "anls": [0]*len(gt_answers)}
		answer_types = answer_types if answer_types is not None else ["string" for batch_idx in range(len(gt_answers))]
		batch_accuracy = []
		batch_anls = []
		layout_labels_accuracy = {value: [] for value in self.layout_map.values()}
		layout_labels_anls = {value: [] for value in self.layout_map.values()}

		# Compute metrics for each batch element
		for b in range(len(preds)):
			gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[b]]
			if isinstance(preds[b], list): # in case of Anyconf
				any_pred = [self._preprocess_str(pred) for pred in preds[b]]
				batch_accuracy_max = 0
				batch_anls_max = 0
				for i, pred in enumerate(any_pred):
					accuracy = self._calculate_accuracy(gt, pred, answer_types[b])
					anls = self._calculate_anls(gt, pred, answer_types[b])
					batch_accuracy_max = max(batch_accuracy_max, accuracy)
					batch_anls_max = max(batch_anls_max, anls)
					if top_k_layout_labels is not None:
						label = self.layout_map[top_k_layout_labels[b][i]]
						layout_labels_accuracy[label].append(accuracy)
						layout_labels_anls[label].append(anls)
				batch_accuracy.append(batch_accuracy_max)
				batch_anls.append(batch_anls_max)
			else: # others
				pred = self._preprocess_str(preds[b])
				accurracy = self._calculate_accuracy(gt, pred, answer_types[b])
				anls = self._calculate_anls(gt, pred, answer_types[b])
				if top_k_layout_labels is not None:
					layout_labels = top_k_layout_labels[b]
					for label in layout_labels:
						label = self.layout_map[label]
						layout_labels_accuracy[label].append(accurracy)
						layout_labels_anls[label].append(anls)
				batch_accuracy.append(accurracy)
				batch_anls.append(anls)

		return {"accuracy": batch_accuracy, "anls": batch_anls, "layout_labels_accuracy": layout_labels_accuracy, "layout_labels_anls": layout_labels_anls}

	def get_retrieval_metric(
			self,
			gt_answer_page: List[int],
			pred_answer_pages: Union[List[int], List[List[int]]]
	) -> list:
		if isinstance(pred_answer_pages[0], int):
			retrieval_precision = [
				1 if gt == pred else 0
				for gt, pred in zip(gt_answer_page, pred_answer_pages)
			]
		else:
			retrieval_precision = [
				1 if gt in preds else 0
				for gt, preds in zip(gt_answer_page, pred_answer_pages)
			]
		return retrieval_precision

	def eval_retrieval(
			self,
			batch: dict,
			retrieval: dict
	) -> dict:
		if not retrieval:
			return {"chunk_score": [0]*len(batch["answers"])}
		# Check if the answer is in the chunks
		chunks = retrieval["text"] # (bs, k)
		answers = batch["answers"] # (bs, n)
		scores = [] # (bs,)
		for b in range(len(answers)):
			top_chunks = chunks[b]
			possible_answers = answers[b]
			best_score = 0
			for ans in possible_answers:
				ans_scores = [get_similarity_score(chunk, ans) for chunk in top_chunks]
				best_score = max(best_score, max(ans_scores+[0]))
			scores.append(np.log(best_score+1) / np.log(2))
		return {
			"chunk_score": scores
		}

	def update_global_metrics(
			self,
			accuracy: float,
			anls: float,
			current_epoch: int
	):
		if accuracy > self.best_accuracy:
			self.best_accuracy = accuracy
			self.best_epoch = current_epoch
			return True
		else:
			return False

	def _preprocess_str(self, string: str) -> str:
		if string is None:
			return ""
		if not self.case_sensitive:
			string = string.lower()
		return string.strip()

	def _calculate_accuracy(
			self,
			gt: List[str],
			pred: str,
			answer_type: str
	) -> int:
		if answer_type == "not-answerable":
			return 1 if pred in ["", "none", "NA", None, []] else 0

		if pred == "none" and answer_type != "not-answerable":
			return 0

		for gt_elm in gt:
			if gt_elm == pred:
				return 1

		return 0

	def _calculate_anls(
			self,
			gt: List[str],
			pred: str,
			answer_type: str
	) -> float:
		if len(pred) == 0:
			return 0

		if answer_type == "not-answerable":
			return 1 if pred in ["", "none", "NA", None, []] else 0

		if pred == "none" and answer_type != "not-answerable":
			return 0

		answers_similarity = [1 - self.get_edit_distance(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
		max_similarity = max(answers_similarity)

		anls = max_similarity if max_similarity >= self.anls_threshold else 0
		return anls

if __name__ == "__main__":
	m = Evaluator()
	print(m.get_metrics([["aa", "ab"], ["xx", "xz"]], ["bb", "xz"]))
