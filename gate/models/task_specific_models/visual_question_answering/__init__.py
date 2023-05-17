import random
from collections import defaultdict
from copy import deepcopy as copy
from typing import Any, Dict, Union

import torch


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
        ](copy(inputs["text"]["question"])).view(-1)

        output_dict["text"]["question_decoder_tokens"] = transform_dict[
            "text_decoder"
        ](copy(inputs["text"]["question"])).view(-1)

        text_decoder_tokenizer = transform_dict["tokenizers"][
            "text_decoder_tokenizer"
        ]

        # print(
        #     f"post tokenizer pre: {output_dict['text']['question_decoder_tokens'].shape}"
        # )

        input_ids = output_dict["text"]["question_decoder_tokens"][:-1]

        output_dict["text"]["question_decoder_tokens"] = input_ids

        output_dict["text"]["question_original"] = copy(
            inputs["text"]["question"]
        )

        # print(
        #     f"post tokenizer post: {output_dict['text']['question_decoder_tokens'].shape}"
        # )

        # print(f"eos_token_id: {text_decoder_tokenizer.eos_token_id}")
        # print(f"bos_token_id: {text_decoder_tokenizer.bos_token_id}")
        # print(
        #     f"answer_start_token_id: {text_decoder_tokenizer.answer_start_token_id}"
        # )
        # print(
        #     f"answer_end_token_id: {text_decoder_tokenizer.answer_end_token_id}"
        # )

    if "text" in inputs and "answers" in inputs["text"]:
        # print(inputs["text"]["answers"])
        random_idx = random.randint(0, len(inputs["text"]["answers"]) - 1)
        output_dict["text"]["answer_decoder_tokens"] = transform_dict[
            "text_decoder"
        ](copy(inputs["text"]["answers"])[random_idx]).view(-1)

        text_decoder_tokenizer = transform_dict["tokenizers"][
            "text_decoder_tokenizer"
        ]

        input_ids = output_dict["text"]["answer_decoder_tokens"]

        output_dict["text"]["answer_decoder_tokens"] = input_ids[1:]

        output_dict["text"]["answer_original"] = copy(
            inputs["text"]["answers"]
        )

        # print(
        #     f"post answer tokenizer post: {output_dict['text']['answer_decoder_tokens'].shape}"
        # )

        # print(f"eos_token_id: {text_decoder_tokenizer.eos_token_id}")
        # print(f"bos_token_id: {text_decoder_tokenizer.bos_token_id}")
        # print(
        #     f"answer_start_token_id: {text_decoder_tokenizer.answer_start_token_id}"
        # )
        # print(
        #     f"answer_end_token_id: {text_decoder_tokenizer.answer_end_token_id}"
        # )

        # print(inputs["text"]["answers"])

    return output_dict
