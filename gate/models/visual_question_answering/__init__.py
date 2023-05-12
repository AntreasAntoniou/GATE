from collections import defaultdict
from copy import deepcopy as copy
import random
from typing import Union, Any, Dict


def transform_wrapper(inputs: Union[Dict, Any], transform_dict: Dict):
    output_dict = defaultdict(dict)
    if "image" in inputs:
        output_dict["image"]["image_encoder_tokens"] = transform_dict[
            "image_encoder"
        ](inputs["image"])
        # output_dict["image"]["image_original"] = [inputs["image"]]

    if "text" in inputs and "question" in inputs["text"]:
        output_dict["text"]["question_encoder_tokens"] = transform_dict[
            "text_encoder"
        ](copy(inputs["text"]["question"]))

        output_dict["text"]["question_decoder_tokens"] = transform_dict[
            "text_decoder"
        ](copy(inputs["text"]["question"]))

        output_dict["text"]["question_original"] = [
            copy(inputs["text"]["question"])
        ]

    if "text" in inputs and "answers" in inputs["text"]:
        random_idx = random.randint(0, len(inputs["text"]["answers"]) - 1)
        output_dict["text"]["answer_decoder_tokens"] = transform_dict[
            "text_decoder"
        ](copy(inputs["text"]["answers"])[random_idx])

        output_dict["text"]["answer_original"] = copy(
            inputs["text"]["answers"]
        )

    return output_dict
