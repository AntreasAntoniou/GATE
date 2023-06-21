from collections import defaultdict
from typing import Optional
from urllib.request import urlopen

import PIL.Image as Image
import timm
import torch
import torch.nn as nn
from rich import print
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)
from transformers.models.clip.modeling_clip import CLIPOutput

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2", add_cross_attention=True
)
print(model.config)

# Assume we have the following input sequence
sample_sequence = [
    "I love deep learning and transformers.",
    "Wow, look at that beautiful sunset.",
    "I wonder what is going to happen next.",
]


def generate_qa_tokens(input_sequence_list):
    question_tokens = []
    answer_tokens = []
    for input_sequence in input_sequence_list:
        question_token = input_sequence[: len(input_sequence) // 2]
        answer_token = input_sequence[len(input_sequence) // 2 :]
        question_tokens.append(question_token)
        answer_tokens.append(answer_token)

    question_tokens = tokenizer(
        question_tokens, padding=True, truncation=True, return_tensors="pt"
    )
    answer_tokens = tokenizer(
        answer_tokens, padding=True, truncation=True, return_tensors="pt"
    )
    return question_tokens, answer_tokens


question_encoder_tokens, answer_encoder_tokens = generate_qa_tokens(
    sample_sequence
)
outputs = model(
    **question_encoder_tokens,
    encoder_hidden_states=torch.randn(3, 10, 768),
    labels=answer_encoder_tokens["input_ids"],
)
loss = outputs.loss
logits = outputs.logits

print(loss, logits.shape)

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class SimpleVQATransformer(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        image_encoder_transforms: nn.Module,
        image_encoder_num_features: int,
        text_encoder: nn.Module,
        text_encoder_num_features: int,
        text_encoder_transforms: nn.Module,
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
        self.text_decoder_tokenizer.pad_token = tokenizer.eos_token

        self.text_decoder = AutoModelForCausalLM.from_pretrained(
            "distilgpt2", add_cross_attention=True
        )

        self.combine_embeddings_linear = nn.Linear(
            image_encoder_num_features + text_encoder_num_features,
            768,
        )

    def forward(
        self,
        input_dict: Optional[Dict] = None,
        image_encoder_tokens: Optional[torch.Tensor] = None,
        question_encoder_tokens: Optional[torch.Tensor] = None,
        question_decoder_tokens: Optional[torch.Tensor] = None,
        answer_decoder_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if input_dict is not None:
            image_encoder_tokens = input_dict["image_tokens"]
            question_encoder_tokens = input_dict["question_tokens"]

        image_embeddings = self.image_encoder(
            image_encoder_tokens
        ).last_hidden_state[:, 0:8, :]

        question_text_embeddings = self.text_encoder(
            question_encoder_tokens
        ).last_hidden_state

        concat_embeddings = torch.cat(
            [image_embeddings, question_text_embeddings], dim=2
        )

        combine_embeddings = self.combine_embeddings_linear(
            concat_embeddings.view(
                -1,
                self.image_encoder_num_features
                + self.text_encoder_num_features,
            )
        ).view(concat_embeddings.shape[0], -1, 768)

        if answer_encoder_tokens is not None:
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

            return self.text_decoder(
                **question_decoder_tokens,
                encoder_hidden_states=combine_embeddings,
                labels=question_decoder_tokens["input_ids"],
            )
        else:
            return self.text_decoder(
                **question_decoder_tokens,
                encoder_hidden_states=combine_embeddings,
            )

    def get_transforms(self):
        return {
            "text_decoder": lambda x: self.text_decoder_tokenizer(
                x, truncation=True, padding=True, return_tensors="pt"
            ),
            "image_encoder": self.image_encoder_transforms,
            "text_encoder": self.text_encoder_transforms,
        }


import accelerate

from gate.models.backbones.clip import CLIPAdapter

accelerator = accelerate.Accelerator()

backbone_model = CLIPAdapter(
    model_name="openai/clip-vit-base-patch16", pretrained=True
)
clip_transforms = backbone_model.get_transforms()
vqa_model = SimpleVQATransformer(
    image_encoder=backbone_model.vision_model,
    image_encoder_transforms=clip_transforms["image"],
    image_encoder_num_features=768,
    text_encoder=backbone_model.text_model,
    text_encoder_transforms=clip_transforms["text"],
    text_encoder_num_features=512,
)
vqa_model = accelerator.prepare(vqa_model)
vqa_transforms = vqa_model.get_transforms()

img = Image.open(
    urlopen(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    )
)
images = [img, img, img]
questions = [
    "What is in the image?",
    "Who is in the image?",
    "What is the weather like?",
]
answers = ["beignets", "a cat", "sunny"]

encoder_images = [vqa_transforms["image_encoder"](image) for image in images]
encoder_questions = vqa_transforms["text_encoder"](questions)
decoder_questions = vqa_transforms["text_decoder"](questions)
decoder_answers = vqa_transforms["text_decoder"](answers)

encoder_images = torch.stack(encoder_images).to(accelerator.device)
encoder_questions = encoder_questions.to(accelerator.device)
decoder_questions = decoder_questions.to(accelerator.device)
decoder_answers = decoder_answers.to(accelerator.device)

print(
    f"encoder_images.shape: {encoder_images.shape}, "
    f"encoder_questions.shape: {encoder_questions.shape}, "
    f"decoder_questions.shape: {decoder_questions.input_ids.shape}, "
    f"decoder_answers.shape: {decoder_answers.input_ids.shape}"
)

optimizer = torch.optim.AdamW(vqa_model.parameters(), lr=1e-5)
optimizer = accelerator.prepare(optimizer)

for i in tqdm(range(100)):
    optimizer.zero_grad()
    output = vqa_model(
        image_encoder_tokens=encoder_images,
        question_encoder_tokens=encoder_questions,
        question_decoder_tokens=decoder_questions,
        answer_decoder_tokens=decoder_answers,
    )
    accelerator.backward(output.loss)
    optimizer.step()
    print(output.loss)
