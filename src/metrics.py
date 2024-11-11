import editdistance
import numpy as np
from typing import List, Union
from src.utils import get_similarity_score

class Evaluator:
	def __init__(
			self,
			case_sensitive: bool=False
	):
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
			answer_types: List[str]=None
	) -> dict:
		answer_types = answer_types if answer_types is not None else ["string" for batch_idx in range(len(gt_answers))]
		batch_accuracy = []
		batch_anls = []

		# Compute metrics for each batch element
		for b in range(len(preds)):
			gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[b]]
			if isinstance(preds[b], list): # in case of Anyconf
				pred = [self._preprocess_str(pred_elm) for pred_elm in preds[b]]
				batch_accuracy_max = 0
				batch_anls_max = 0
				for pred_i in pred:
					i_accuracy = self._calculate_accuracy(gt, pred_i, answer_types[b])
					i_anls = self._calculate_anls(gt, pred_i, answer_types[b])
					batch_accuracy_max = max(batch_accuracy_max, i_accuracy)
					batch_anls_max = max(batch_anls_max, i_anls)
				batch_accuracy.append(batch_accuracy_max)
				batch_anls.append(batch_anls_max)
			else: # others
				pred = self._preprocess_str(preds[b])
				batch_accuracy.append(self._calculate_accuracy(gt, pred, answer_types[b]))
				batch_anls.append(self._calculate_anls(gt, pred, answer_types[b]))

		return {"accuracy": batch_accuracy, "anls": batch_anls}

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
