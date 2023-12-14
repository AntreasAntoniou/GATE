import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional

import torch

from gate.metrics.glossary import (articles, comma_strip, contractions,
                                   manual_map, period_strip, punct)


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
    answers: List[List[str]],
    predicted_answers: List[str],
):
    results_dict = defaultdict(list)

    for question_id, (answer_list, predicted_answer) in enumerate(
        zip(answers, predicted_answers)
    ):
        for idx, answer in enumerate(answer_list):
            answer_list[idx] = (
                answer.replace("\n", " ").replace("\t", " ").strip()
            )

        predicted_answer = (
            predicted_answer.replace("\n", " ").replace("\t", " ").strip()
        )

        if len(set(answer_list)) > 1:
            answer_list = [
                process_digit_article(process_punctuation(ans))
                for ans in answer_list
            ]
            predicted_answer = process_digit_article(
                process_punctuation(predicted_answer)
            )

        temp_accuracy = []
        for target_answer in list(combinations(answer_list, 9)):
            matches = torch.tensor(
                [
                    1.0
                    for answer in target_answer
                    if (
                        answer in predicted_answer
                        or predicted_answer in answer
                    )
                ]
            )
            accuracy = min(1.0, torch.sum(matches) / 3.0)
            temp_accuracy.append(accuracy)

        avg_accuracy = torch.mean(torch.tensor(temp_accuracy))
        results_dict["overall"].append(avg_accuracy)

    return results_dict
