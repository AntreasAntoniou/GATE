from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional
from gate.metrics.glossary import (
    contractions,
    manual_map,
    articles,
    period_strip,
    comma_strip,
    punct,
)
import torch
from tqdm.auto import tqdm
from rich import print


@dataclass
class AnswerData:
    answer: str
    answer_confidence: Optional[str] = None
    answer_id: Optional[int] = None


@dataclass
class VQAItem:
    answers: List[AnswerData]
    image_id: int
    question: str
    question_id: int
    question_type: str
    answer_type: str
    multiple_choice_answer: Optional[str] = None


class VQAEval:
    """A class to evaluate VQA predictions."""

    def __init__(
        self,
        vqa_data: Dict[int, VQAItem],
        vqa_predictions: Dict[int, AnswerData],
        precision: int = 2,
    ):
        self.precision = precision
        self.accuracy = {}
        self.eval_question_answer = {}
        self.eval_question_type = {}
        self.eval_answer_type = {}
        self.vqa_data = vqa_data
        self.vqa_predictions = vqa_predictions
        self.contractions = contractions
        self.manual_map = manual_map
        self.articles = articles
        self.period_strip = period_strip
        self.comma_strip = comma_strip
        self.punctuation = punct

    def evaluate(
        self, question_ids: Optional[List[int]] = None
    ) -> Dict[str, List[float]]:
        """Evaluate the VQA predictions against the ground truth.

        Args:
            question_ids: A list of question IDs to evaluate. If None, all the IDs in vqa_data are used.

        Returns:
            A dictionary with accuracy results for overall and question type categories.
        """
        if question_ids is None:
            question_ids = list(self.vqa_data.keys())
        target_qa_dict = {qid: self.vqa_data[qid] for qid in question_ids}
        predicted_answer_dict = {
            qid: self.vqa_predictions[qid] for qid in question_ids
        }

        results_dict = defaultdict(list)
        for question_id in tqdm(question_ids, desc="Processing questions"):
            vqa_item = target_qa_dict[question_id]
            for answer in vqa_item.answers:
                answer.answer = (
                    answer.answer.replace("\n", " ").replace("\t", " ").strip()
                )

            predicted_answer = predicted_answer_dict[question_id].answer
            predicted_answer = (
                predicted_answer.replace("\n", " ").replace("\t", " ").strip()
            )

            if len(set([ans.answer for ans in vqa_item.answers])) > 1:
                vqa_item.answers = [
                    AnswerData(
                        answer=self.process_digit_article(
                            self.process_punctuation(ans.answer)
                        ),
                        answer_confidence=ans.answer_confidence,
                        answer_id=ans.answer_id,
                    )
                    for ans in vqa_item.answers
                ]
                predicted_answer = self.process_digit_article(
                    self.process_punctuation(predicted_answer)
                )

            temp_accuracy = []
            for target_answer in vqa_item.answers:
                matching_answers = [
                    ans.answer
                    for ans in vqa_item.answers
                    if ans.answer == predicted_answer
                    and ans != target_answer.answer
                ]
                print(
                    f"Matching answers: {matching_answers}, Target answers: {target_answer.answer}, Predicted: {predicted_answer}"
                )
                accuracy = min(1.0, float(len(matching_answers)) / 3.0)
                temp_accuracy.append(accuracy)

            avg_accuracy = torch.mean(
                torch.tensor(temp_accuracy, dtype=torch.float32)
            )
            results_dict["overall"].append(avg_accuracy.item())
            results_dict[vqa_item.question_type].append(avg_accuracy.item())

        return results_dict

    def process_punctuation(self, input_text: str) -> str:
        """Process punctuation in the input text.

        Args:
            input_text: A string containing the input text.

        Returns:
            A string with processed punctuation.
        """
        output_text = input_text
        for punct_char in self.punctuation:
            if (
                punct_char + " " in input_text
                or " " + punct_char in input_text
            ) or (re.search(self.comma_strip, input_text) is not None):
                output_text = output_text.replace(punct_char, "")
            else:
                output_text = output_text.replace(punct_char, " ")
        output_text = self.period_strip.sub("", output_text, re.UNICODE)
        return output_text

    def process_digit_article(self, input_text: str) -> str:
        """Process digits and articles in the input text.

        Args:
            input_text: A string containing the input text.

        Returns:
            A string with processed digits and articles.
        """
        temp_text = input_text.lower().split()
        output_text = [
            self.manual_map.get(word, word)
            for word in temp_text
            if word not in self.articles
        ]
        for word_id, word in enumerate(output_text):
            if word in self.contractions:
                output_text[word_id] = self.contractions[word]
        return " ".join(output_text)
