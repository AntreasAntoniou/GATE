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


def process_punctuation(input_text: str) -> str:
    """Process punctuation in the input text.

    Args:
        input_text: A string containing the input text.

    Returns:
        A string with processed punctuation.
    """
    output_text = input_text
    for punct_char in punct:
        if (
            punct_char + " " in input_text or " " + punct_char in input_text
        ) or (re.search(comma_strip, input_text) is not None):
            output_text = output_text.replace(punct_char, "")
        else:
            output_text = output_text.replace(punct_char, " ")
    output_text = period_strip.sub("", output_text, re.UNICODE)
    return output_text


def process_digit_article(input_text: str) -> str:
    """Process digits and articles in the input text.

    Args:
        input_text: A string containing the input text.

    Returns:
        A string with processed digits and articles.
    """
    temp_text = input_text.lower().split()
    output_text = [
        manual_map.get(word, word)
        for word in temp_text
        if word not in articles
    ]
    for word_id, word in enumerate(output_text):
        if word in contractions:
            output_text[word_id] = contractions[word]
    return " ".join(output_text)


def vqa_metric(
    vqa_data: Dict[int, VQAItem],
    vqa_predictions: Dict[int, AnswerData],
):
    """A class to evaluate VQA predictions."""

    question_ids = list(vqa_data.keys())
    target_qa_dict = {qid: vqa_data[qid] for qid in question_ids}
    predicted_answer_dict = {qid: vqa_predictions[qid] for qid in question_ids}

    results_dict = defaultdict(list)
    for question_id in tqdm(question_ids, desc="Processing questions"):
        vqa_item = target_qa_dict[question_id]
        for answer in vqa_item.answers:
            print(answer)
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
                    answer=process_digit_article(
                        process_punctuation(ans.answer)
                    ),
                    answer_confidence=ans.answer_confidence,
                    answer_id=ans.answer_id,
                )
                for ans in vqa_item.answers
            ]
            predicted_answer = process_digit_article(
                process_punctuation(predicted_answer)
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
