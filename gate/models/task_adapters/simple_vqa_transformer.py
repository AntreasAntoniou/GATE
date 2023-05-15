# Standard library imports
from typing import Dict, Optional

# Third-party imports
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    BartForQuestionAnswering,
    BartTokenizer,
    BartTokenizerFast,
)


def tokenize_with_start_end(text, tokenizer):
    output_dict = tokenizer(text, truncation=True, return_tensors="pt")
    start_token_id = tokenizer.bos_token_id  # get the id of the start token
    end_token_id = tokenizer.eos_token_id  # get the id of the end token

    # add start and end tokens

    output_dict = torch.cat(
        [
            torch.tensor([[start_token_id]]),
            output_dict["input_ids"],
            torch.tensor([[end_token_id]]),
        ],
        dim=1,
    )

    return output_dict

class BartWithCustomEncoder(BartForQuestionAnswering):



class SimpleVQATransformer(nn.Module):
    """
    This class represents a simple Visual Question Answering (VQA) transformer model.
    It incorporates image and text encoders, and a text decoder for generating answers to visual questions.
    """

    def __init__(
        self,
        image_encoder: nn.Module,  # Encoder for the image input
        image_encoder_transforms: nn.Module,  # Transformations to apply on the image input
        image_encoder_num_features: int,  # Number of features in the image encoder's output
        text_encoder: nn.Module,  # Encoder for the text input
        text_encoder_num_features: int,  # Number of features in the text encoder's output
        text_encoder_transforms: nn.Module,  # Transformations to apply on the text input
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_encoder_tokenizer = text_encoder.tokenizer.tokenizer

        self.image_encoder_num_features = image_encoder_num_features
        self.text_encoder_num_features = text_encoder_num_features

        self.image_encoder_transforms = image_encoder_transforms
        self.text_encoder_transforms = text_encoder_transforms

        # Existing tokenizer
        self.text_decoder_tokenizer: BartTokenizer | BartTokenizerFast = (
            AutoTokenizer.from_pretrained(
                "valhalla/bart-large-finetuned-squadv1", use_fast=True
            )
        )

        self.text_decoder = BartForQuestionAnswering.from_pretrained(
            "valhalla/bart-large-finetuned-squadv1"
        )

        self._setup_special_tokens()

        self.image_embedding_projection = nn.Linear(
            image_encoder_num_features, 512
        )

        # Linear layer to combine image and text embeddings
        self.combine_embeddings_linear = nn.Linear(
            512,
            768,  # The combined embeddings size is set to match the hidden size of the 'distilgpt2' model
        )

    def _setup_special_tokens(self):
        # New tokens
        new_tokens = [
            "<pad>",
            "<q>",
            "<q/>",
            "<a>",
            "<a/>",
        ]

        # Add new tokens
        num_added_tokens = self.text_decoder_tokenizer.add_tokens(
            new_tokens, special_tokens=True
        )
        # print(f"Added {num_added_tokens} tokens")

        pad_token = self.text_decoder_tokenizer.encode("<pad>")[0]
        question_start_token = self.text_decoder_tokenizer.encode("<q>")[0]
        question_end_token = self.text_decoder_tokenizer.encode("<q/>")[0]
        answer_start_token = self.text_decoder_tokenizer.encode("<a>")[0]
        answer_end_token = self.text_decoder_tokenizer.encode("<a/>")[0]

        print(
            f"pad_token: {pad_token}, question_start_token: {question_start_token}, question_end_token: {question_end_token}, answer_start_token: {answer_start_token}, answer_end_token: {answer_end_token}"
        )

        setattr(
            self.text_decoder_tokenizer,
            "pad_token_id",
            pad_token,
        )
        setattr(
            self.text_decoder_tokenizer,
            "question_start_token_id",
            question_start_token,
        )
        setattr(
            self.text_decoder_tokenizer,
            "question_end_token_id",
            question_end_token,
        )
        setattr(
            self.text_decoder_tokenizer,
            "answer_start_token_id",
            answer_start_token,
        )
        setattr(
            self.text_decoder_tokenizer,
            "answer_end_token_id",
            answer_end_token,
        )

        # print(self.text_decoder_tokenizer.additional_special_tokens_ids)

        self.text_decoder.resize_token_embeddings(
            len(self.text_decoder_tokenizer)
        )  # Notice resize_token_embeddings method

    def _setup_padding(
        self,
        question_encoder_tokens,
        question_decoder_tokens,
        answer_decoder_tokens,
    ):
        question_encoder_tokens[
            question_encoder_tokens == -1
        ] = self.text_encoder_tokenizer.pad_token_id
        ###########################################################
        question_decoder_tokens_attention_mask = torch.ones(
            question_decoder_tokens.shape
        ).to(question_decoder_tokens.device)
        question_decoder_tokens_attention_mask[
            question_decoder_tokens == -1
        ] = 0
        question_decoder_tokens[
            question_decoder_tokens == -1
        ] = self.text_decoder_tokenizer.pad_token_id
        # print(f"pad_token_id: {self.text_decoder_tokenizer.pad_token_id}")

        ###########################################################
        answer_decoder_tokens_attention_mask = torch.ones(
            answer_decoder_tokens.shape
        ).to(answer_decoder_tokens.device)
        answer_decoder_tokens_attention_mask[answer_decoder_tokens == -1] = 0
        answer_decoder_tokens[
            answer_decoder_tokens == -1
        ] = self.text_decoder_tokenizer.pad_token_id

        return (
            question_encoder_tokens,
            question_decoder_tokens,
            question_decoder_tokens_attention_mask,
            answer_decoder_tokens,
            answer_decoder_tokens_attention_mask,
        )

    def forward(
        self,
        input_dict: Optional[Dict] = None,
        image_encoder_tokens: Optional[torch.Tensor] = None,
        question_encoder_tokens: Optional[torch.Tensor] = None,
        question_decoder_tokens: Optional[torch.Tensor] = None,
        answer_decoder_tokens: Optional[torch.Tensor] = None,
        image: Optional[Dict[str, torch.Tensor]] = None,
        text: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        The forward method for the SimpleVQATransformer model.

        Parameters:
        - input_dict (Optional[Dict]): A dictionary containing image and question tokens.
        - image_encoder_tokens (Optional[torch.Tensor]): Tokens for the image encoder.
        - question_encoder_tokens (Optional[torch.Tensor]): Tokens for the question encoder.
        - question_decoder_tokens (Optional[torch.Tensor]): Tokens for the question decoder.
        - answer_decoder_tokens (Optional[torch.Tensor]): Tokens for the answer decoder.

        Returns:
        - A dictionary with tensor outputs of the text decoder.
        """
        if input_dict is not None:
            image_encoder_tokens = input_dict["image_encoder_tokens"]
            question_encoder_tokens = input_dict["question_encoder_tokens"]
            question_decoder_tokens = input_dict["question_decoder_tokens"]
            answer_decoder_tokens = input_dict["answer_decoder_tokens"]

        if image is not None:
            image_encoder_tokens = image["image_encoder_tokens"]

        if text is not None:
            question_encoder_tokens = text["question_encoder_tokens"]
            question_decoder_tokens = text["question_decoder_tokens"]
            answer_decoder_tokens = text["answer_decoder_tokens"]

        (
            question_encoder_tokens,
            question_decoder_tokens,
            question_decoder_tokens_attention_mask,
            answer_decoder_tokens,
            answer_decoder_tokens_attention_mask,
        ) = self._setup_padding(
            question_encoder_tokens=question_encoder_tokens,
            question_decoder_tokens=question_decoder_tokens,
            answer_decoder_tokens=answer_decoder_tokens,
        )

        # Obtain the image embeddings from the image encoder
        image_embeddings = self.image_encoder(image=image_encoder_tokens)[
            "image"
        ]["raw_features"][:, 0:8, :]
        image_embeddings = self.image_embedding_projection(image_embeddings)

        # Obtain the question text embeddings from the text encoder
        question_text_embeddings = self.text_encoder(
            text=question_encoder_tokens
        )["text"]["raw_features"]

        # Concatenate image and text embeddings along dimension 2
        concat_embeddings = torch.cat(
            [image_embeddings, question_text_embeddings], dim=1
        )

        # Combine image and text embeddings using a linear layer
        combine_embeddings = self.combine_embeddings_linear(
            concat_embeddings.view(
                -1,
                512,
            )
        ).view(concat_embeddings.shape[0], -1, 768)

        if answer_decoder_tokens is not None:
            # If answer tokens are provided, concatenate question and answer tokens
            question_answer_tokens = {
                "input_ids": torch.cat(
                    [question_decoder_tokens, answer_decoder_tokens],
                    dim=1,
                ),
                "attention_mask": torch.cat(
                    [
                        question_decoder_tokens_attention_mask,
                        answer_decoder_tokens_attention_mask,
                    ],
                    dim=1,
                ),
            }
            input_dict = {
                "input_ids": question_answer_tokens["input_ids"][:, :-1],
                "attention_mask": question_answer_tokens["attention_mask"][
                    :, :-1
                ],
            }
            label_dict = {
                "input_ids": question_answer_tokens["input_ids"][:, 1:],
                "attention_mask": question_answer_tokens["attention_mask"][
                    :, 1:
                ],
            }
            # torch.set_printoptions(
            #     edgeitems=1000
            # )  # Adjust the number as needed

            # print(
            #     f"question_answer_tokens: {question_answer_tokens['input_ids'][0]}"
            # )

            # print(
            #     f"input: {input_dict['input_ids']}, input_shape: {input_dict['input_ids'].shape},\n"
            #     f"attention_mask: {input_dict['attention_mask']}, attention_mask_shape: {input_dict['attention_mask'].shape},\n"
            #     f"label: {label_dict['input_ids']}, label_shape: {label_dict['input_ids'].shape}"
            # )

            # Return the output of the text decoder, using combined embeddings as encoder hidden states
            # and question tokens as labels
            output = self.text_decoder(
                **input_dict,
                encoder_hidden_states=combine_embeddings,
                labels=label_dict["input_ids"],
            )
        else:
            model_input = {
                "input_ids": question_decoder_tokens,
                "attention_mask": torch.ones(answer_decoder_tokens.shape).to(
                    answer_decoder_tokens.device
                ),
            }
            # If answer tokens are not provided, simply return the output of the text decoder
            output = self.text_decoder(
                **model_input,
                encoder_hidden_states=combine_embeddings,
            )
        return output.__dict__

    def generate_tokens(
        self,
        input_dict: Optional[Dict] = None,
        image_encoder_tokens: Optional[torch.Tensor] = None,
        question_encoder_tokens: Optional[torch.Tensor] = None,
        question_decoder_tokens: Optional[torch.Tensor] = None,
        image: Optional[Dict[str, torch.Tensor]] = None,
        text: Optional[Dict[str, torch.Tensor]] = None,
        max_length: int = 100,
        do_sample: bool = True,
    ) -> str:
        """
        This method generates a textual answer given the same inputs
        as the forward pass.

        Parameters:
        - input_dict (Optional[Dict]): A dictionary containing image
        and question tokens.
        - image_encoder_tokens (Optional[torch.Tensor]): Tokens for the
        image encoder.
        - question_encoder_tokens (Optional[torch.Tensor]): Tokens for the
        question encoder.
        - max_length (int): Maximum length of the generated answer.

        Returns:
        - A string containing the generated answer.
        """

        # Get the model output using the forward method with input_dict or the
        # supplied encoder tokens
        if input_dict is not None:
            image_encoder_tokens = input_dict["image_encoder_tokens"]
            question_encoder_tokens = input_dict["question_encoder_tokens"]
            question_decoder_tokens = input_dict["question_decoder_tokens"]

        if image is not None:
            image_encoder_tokens = image["image_encoder_tokens"]

        if text is not None:
            question_encoder_tokens = text["question_encoder_tokens"]
            question_decoder_tokens = text["question_decoder_tokens"]

        # Obtain the image embeddings from the image encoder
        image_embeddings = self.image_encoder(image=image_encoder_tokens)[
            "image"
        ]["raw_features"][:, 0:8, :]
        image_embeddings = self.image_embedding_projection(image_embeddings)
        # Obtain the question text embeddings from the text encoder
        question_text_embeddings = self.text_encoder(
            text=question_encoder_tokens
        )["text"]["raw_features"]

        # Concatenate image and text embeddings along dimension 2
        concat_embeddings = torch.cat(
            [image_embeddings, question_text_embeddings], dim=1
        )

        # Combine image and text embeddings using a linear layer
        combined_embeddings = self.combine_embeddings_linear(
            concat_embeddings.view(
                -1,
                512,
            )
        ).view(concat_embeddings.shape[0], -1, 768)
        # Use the Transformers 'generate' method to generate an answer

        # max_length: The maximum length of the generated sequence.

        # do_sample: If set to False (default), the model will use "greedy decoding" --
        # it will always choose the token with the highest probability as the next token.
        # If set to True, the model will sample the next token from the probability
        # distribution predicted by the model (potentially leading to more diverse, but
        # less deterministic, output).

        # temperature: Affects the randomness of the sampling process when do_sample=True.
        # Higher values (e.g., 1.0) make the output more random, while lower values
        # (e.g., 0.1) make it more deterministic.

        # top_k: Used for "Top-K sampling". The model's predicted probabilities are
        # sorted in descending order, and only the top k tokens are considered for sampling.

        # top_p: Used for "nucleus sampling" (also known as "Top-P sampling").
        # Instead of selecting the top k tokens, it selects the smallest set of
        # tokens whose cumulative probability exceeds p.

        # num_beams: The number of "beams" for beam search. Beam search keeps
        # track of the num_beams most probable sequences at each step.
        # If num_beams > 1, the model will generate num_return_sequences sequences.
        # If num_return_sequences is not provided, it will generate num_beams sequences.

        answer_tokens = self.text_decoder.generate(
            input_ids=question_decoder_tokens,
            attention_mask=torch.ones(question_decoder_tokens.shape).to(
                question_decoder_tokens.device
            ),
            encoder_hidden_states=combined_embeddings,
            max_length=max_length,
            do_sample=do_sample,
        )

        # print(f"answer_tokens: {answer_tokens[0]}")

        return answer_tokens

    def generate_text(
        self,
        input_dict: Optional[Dict] = None,
        image_encoder_tokens: Optional[torch.Tensor] = None,
        question_encoder_tokens: Optional[torch.Tensor] = None,
        question_decoder_tokens: Optional[torch.Tensor] = None,
        image: Optional[Dict[str, torch.Tensor]] = None,
        text: Optional[Dict[str, torch.Tensor]] = None,
        max_length: int = 100,
        do_sample: bool = True,
    ):
        decoded_tokens = self.generate_tokens(
            input_dict=input_dict,
            image_encoder_tokens=image_encoder_tokens,
            question_encoder_tokens=question_encoder_tokens,
            question_decoder_tokens=question_decoder_tokens,
            image=image,
            text=text,
            max_length=max_length,
            do_sample=do_sample,
        )
        # Decode the generated tokens into a string
        return [
            self.text_decoder_tokenizer.decode(
                decoded_token,
                skip_special_tokens=True,
            )
            for decoded_token in decoded_tokens
        ]

    def get_transforms(self):
        """
        This method returns a dictionary of transformations for the text decoder, image encoder, and text encoder.

        Returns:
        - A dictionary with keys 'text_decoder', 'image_encoder', 'text_encoder'.
          Each key corresponds to a function that applies the appropriate transformations.
        """
        return {
            "text_decoder": lambda x: tokenize_with_start_end(
                text=x, tokenizer=self.text_decoder_tokenizer
            ),
            "image_encoder": self.image_encoder_transforms,
            "text_encoder": self.text_encoder_transforms,
            "tokenizers": {
                "text_decoder_tokenizer": self.text_decoder_tokenizer
            },
        }
