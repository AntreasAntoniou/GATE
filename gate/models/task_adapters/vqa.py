from typing import Dict, Optional

import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SimpleVQATransformer(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
    ):
        super().__init__()
        self.decoder_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_decoder = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.combine_embeddings_linear = nn.Linear(
            self.image_encoder.num_clip_features
            + self.text_encoder.num_clip_features,
            self.text_decoder.config.vocab_size,
        )

    def forward(
        self,
        input_dict: Optional[Dict] = None,
        image_tokens: Optional[torch.Tensor] = None,
        question_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if input_dict is not None:
            image_tokens = input_dict["image_tokens"]
            question_tokens = input_dict["question_tokens"]

        image_embeddings = self.image_encoder(image_tokens)["image_features"]

        question_text_embeddings = self.text_encoder(question_tokens)["text_features"]

        concat_embeddings = torch.cat(
            [image_embeddings, question_text_embeddings], dim=1
        )

        combine_embeddings = self.combine_embeddings_linear(concat_embeddings)

        # decode answer

        return x
    
    def training_step(self, input_dict: Optional[Dict] = None,
        image_tokens: Optional[torch.Tensor] = None,
        question_tokens: Optional[torch.Tensor] = None,
        answer_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Given question and image, predict answer and train auto regressively

    def get_transforms(self):
        return {"text": lambda x: self.decoder_tokenizer(x)}
