# Standard library imports
from typing import Dict, Optional

# Third-party imports
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


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

        self.image_encoder_num_features = image_encoder_num_features
        self.text_encoder_num_features = text_encoder_num_features

        self.image_encoder_transforms = image_encoder_transforms
        self.text_encoder_transforms = text_encoder_transforms

        self.text_decoder_tokenizer = AutoTokenizer.from_pretrained(
            "distilgpt2"
        )
        self.text_decoder_tokenizer.pad_token = (
            self.text_decoder_tokenizer.eos_token
        )

        self.text_decoder = AutoModelForCausalLM.from_pretrained(
            "distilgpt2", add_cross_attention=True
        )

        # Linear layer to combine image and text embeddings
        self.combine_embeddings_linear = nn.Linear(
            image_encoder_num_features + text_encoder_num_features,
            768,  # The combined embeddings size is set to match the hidden size of the 'distilgpt2' model
        )

    def forward(
        self,
        input_dict: Optional[Dict] = None,
        image_encoder_tokens: Optional[torch.Tensor] = None,
        question_encoder_tokens: Optional[torch.Tensor] = None,
        question_decoder_tokens: Optional[torch.Tensor] = None,
        answer_decoder_tokens: Optional[torch.Tensor] = None,
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

        # Obtain the image embeddings from the image encoder
        image_embeddings = self.image_encoder(image_encoder_tokens)["image"][
            "raw_features"
        ][:, 0:8, :]

        # Obtain the question text embeddings from the text encoder
        question_text_embeddings = self.text_encoder(question_encoder_tokens)[
            "text"
        ]["raw_features"]

        # Concatenate image and text embeddings along dimension 2
        concat_embeddings = torch.cat(
            [image_embeddings, question_text_embeddings], dim=2
        )

        # Combine image and text embeddings using a linear layer
        combine_embeddings = self.combine_embeddings_linear(
            concat_embeddings.view(
                -1,
                self.image_encoder_num_features
                + self.text_encoder_num_features,
            )
        ).view(concat_embeddings.shape[0], -1, 768)

        if answer_decoder_tokens is not None:
            # If answer tokens are provided, concatenate question and answer tokens
            question_decoder_tokens["input_ids"] = torch.cat(
                [
                    question_decoder_tokens["input_ids"],
                    answer_decoder_tokens["input_ids"],
                ],
                dim=1,
            )
            question_decoder_tokens["attention_mask"] = torch.cat(
                [
                    question_decoder_tokens["attention_mask"],
                    answer_decoder_tokens["attention_mask"],
                ],
                dim=1,
            )

            # Return the output of the text decoder, using combined embeddings as encoder hidden states
            # and question tokens as labels
            return self.text_decoder(
                **question_decoder_tokens,
                encoder_hidden_states=combine_embeddings,
                labels=question_decoder_tokens["input_ids"],
            )
        else:
            # If answer tokens are not provided, simply return the output of the text decoder
            return self.text_decoder(
                **question_decoder_tokens,
                encoder_hidden_states=combine_embeddings,
            )

    def get_transforms(self):
        """
        This method returns a dictionary of transformations for the text decoder, image encoder, and text encoder.

        Returns:
        - A dictionary with keys 'text_decoder', 'image_encoder', 'text_encoder'.
          Each key corresponds to a function that applies the appropriate transformations.
        """
        return {
            "text_decoder": lambda x: self.text_decoder_tokenizer(
                x, truncation=True, padding=True, return_tensors="pt"
            ),
            "image_encoder": self.image_encoder_transforms,
            "text_encoder": self.text_encoder_transforms,
        }

    def generate_tokens(
        self,
        input_dict: Optional[Dict] = None,
        image_encoder_tokens: Optional[torch.Tensor] = None,
        question_encoder_tokens: Optional[torch.Tensor] = None,
        question_decoder_tokens: Optional[torch.Tensor] = None,
        max_length: int = 50,
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

        # Obtain the image embeddings from the image encoder
        image_embeddings = self.image_encoder(image_encoder_tokens)[
            "raw_features"
        ][:, 0:8, :]

        # Obtain the question text embeddings from the text encoder
        question_text_embeddings = self.text_encoder(question_encoder_tokens)[
            "raw_features"
        ]

        # Concatenate image and text embeddings along dimension 2
        concat_embeddings = torch.cat(
            [image_embeddings, question_text_embeddings], dim=2
        )

        # Combine image and text embeddings using a linear layer
        combined_embeddings = self.combine_embeddings_linear(
            concat_embeddings.view(
                -1,
                self.image_encoder_num_features
                + self.text_encoder_num_features,
            )
        ).view(concat_embeddings.shape[0], -1, 768)
        # Use the Transformers 'generate' method to generate an answer
        answer_tokens = self.text_decoder.generate(
            **question_decoder_tokens,
            encoder_hidden_states=combined_embeddings,
            max_length=max_length,
        )

        return answer_tokens

    def generate_text(
        self,
        input_dict: Optional[Dict] = None,
        image_encoder_tokens: Optional[torch.Tensor] = None,
        question_encoder_tokens: Optional[torch.Tensor] = None,
        question_decoder_tokens: Optional[torch.Tensor] = None,
        max_length: int = 50,
    ):
        decoded_tokens = self.generate_tokens(
            input_dict,
            image_encoder_tokens,
            question_encoder_tokens,
            question_decoder_tokens,
            max_length,
        )
        # Decode the generated tokens into a string
        return [
            self.text_decoder_tokenizer.decode(
                decoded_token,
                skip_special_tokens=True,
            )
            for decoded_token in decoded_tokens
        ]
