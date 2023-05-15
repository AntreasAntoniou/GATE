from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from gate.models.core import reinit
from gate.models.task_adapters.few_shot_classification import (
    FewShotLearningClassificationEpisode,
)

from gate.models.task_adapters.few_shot_classification.utils import (
    get_accuracy,
    get_prototypes,
    prototypical_loss_and_logits,
)


class PrototypicalNetwork(nn.Module):
    """
    This is the Prototypical Network class.

    Args:
        model: The base model.
        num_clip_features: The number of clip features.
        modality: The modality of the data.
        num_output_features: The number of output features. Defaults to None.

    Attributes:
        model: The base model.
        modality: The modality of the data.
        num_output_features: The number of output features.
        linear: A linear layer, or an identity layer if num_output_features is None.
    """

    def __init__(
        self,
        model: nn.Module,
        num_clip_features: int,
        modality: str,
        num_output_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.modality = modality

        # If num_output_features is not provided, use num_clip_features and set linear layer to identity.
        if num_output_features is None:
            self.num_output_features = num_clip_features
            self.linear = nn.Identity()
        else:
            self.num_output_features = num_output_features
            self.linear = nn.Linear(num_clip_features, num_output_features)

    def init_weights(self):
        reinit(self)

    def _process_episode(self, image, audio, video, text):
        # check that only one of the inputs is not None
        assert (
            sum(
                [
                    image is not None,
                    audio is not None,
                    video is not None,
                    text is not None,
                ]
            )
            == 1
        ), "Only one input must be provided."

        # Get the inputs and labels
        if isinstance(image, Dict):
            image = FewShotLearningClassificationEpisode(**image)
        if isinstance(audio, Dict):
            audio = FewShotLearningClassificationEpisode(**audio)
        if isinstance(video, Dict):
            video = FewShotLearningClassificationEpisode(**video)
        if text is not None:
            text = FewShotLearningClassificationEpisode(**text)

        if image is not None:
            support_set_inputs = image.support_set_inputs
            support_set_labels = image.support_set_labels
            query_set_inputs = image.query_set_inputs
            query_set_labels = image.query_set_labels
        elif audio is not None:
            support_set_inputs = audio.support_set_inputs
            support_set_labels = audio.support_set_labels
            query_set_inputs = audio.query_set_inputs
            query_set_labels = audio.query_set_labels
        elif video is not None:
            support_set_inputs = video.support_set_inputs
            support_set_labels = video.support_set_labels
            query_set_inputs = video.query_set_inputs
            query_set_labels = video.query_set_labels
        elif text is not None:
            support_set_inputs = text.support_set_inputs
            support_set_labels = text.support_set_labels
            query_set_inputs = text.query_set_inputs
            query_set_labels = text.query_set_labels

        return (
            support_set_inputs,
            support_set_labels,
            query_set_inputs,
            query_set_labels,
        )

    def forward_features(
        self,
        input_dict: Optional[Dict[str, Union[torch.Tensor, Dict]]] = None,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        This method takes an input dictionary and applies the model to the input.

        Args:
            input_dict: A dictionary of input data.
            image: Optional image tensor.
            text: Optional text tensor.
            audio: Optional audio tensor.
            video: Optional video tensor.

        Returns:
            The output tensor after being processed by the model and the linear layer.
        """
        x = None
        if input_dict is not None:
            x = self.model(**input_dict)[self.modality]["features"]
        if image is not None:
            x = self.model(image=image)[self.modality]["features"]
        if text is not None:
            x = self.model(text=text)[self.modality]["features"]
        if audio is not None:
            x = self.model(audio=audio)[self.modality]["features"]
        if video is not None:
            x = self.model(video=video)[self.modality]["features"]

        assert x is not None, "At least one input must be provided."

        x = self.linear(x)
        return x

    def forward(
        self,
        image: Optional[
            Union[
                FewShotLearningClassificationEpisode, Dict[str, torch.Tensor]
            ]
        ] = None,
        audio: Optional[
            Union[
                FewShotLearningClassificationEpisode, Dict[str, torch.Tensor]
            ]
        ] = None,
        video: Optional[
            Union[
                FewShotLearningClassificationEpisode, Dict[str, torch.Tensor]
            ]
        ] = None,
        text: Optional[
            Union[
                FewShotLearningClassificationEpisode, Dict[str, torch.Tensor]
            ]
        ] = None,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        This method processes the support set and query set inputs and labels, and computes the loss and accuracy if the query set labels are provided.

        Args:
            support_set_inputs: The support set inputs.
            query_set_inputs: The query set inputs.
            support_set_labels: The support set labels.
            query_set_labels: The query set labels, if provided.

        Returns:
            A dictionary containing the prototypes, query set embeddings, support set embeddings, and optionally the loss and accuracy.
        """
        (
            support_set_inputs,
            support_set_labels,
            query_set_inputs,
            query_set_labels,
        ) = self._process_episode(image, audio, video, text)

        # Store outputs in this dictionary
        output_dict = {}

        # Get the number of tasks and examples
        num_tasks, num_examples = support_set_inputs.shape[:2]

        # Compute the support set features and embeddings
        support_set_features = self.forward_features(
            **{
                self.modality: support_set_inputs.view(
                    -1, *support_set_inputs.shape[2:]
                )
            }
        )

        support_set_embedding = support_set_features.view(
            num_tasks, num_examples, -1
        )

        # Compute the query set features and embeddings
        query_set_features = self.forward_features(
            **{
                self.modality: query_set_inputs.view(
                    -1, *query_set_inputs.shape[2:]
                )
            }
        )
        query_set_embedding = query_set_features.view(
            num_tasks, num_examples, -1
        )

        # Get the prototypes
        prototypes = get_prototypes(
            embeddings=support_set_embedding,
            targets=support_set_labels,
            num_classes=int(torch.max(support_set_labels)) + 1,
        )

        # Store the outputs
        output_dict["prototypes"] = prototypes
        output_dict["query_set_embedding"] = query_set_embedding
        output_dict["support_set_embedding"] = support_set_embedding

        # If query set labels are provided, calculate the loss and accuracy
        if query_set_labels is not None:
            prototype_loss_and_logits = prototypical_loss_and_logits(
                prototypes, query_set_embedding, query_set_labels
            )

            output_dict["loss"] = torch.mean(prototype_loss_and_logits["loss"])
            output_dict["logits"] = prototype_loss_and_logits[
                "logits"
            ].permute([0, 2, 1])

            accuracy = get_accuracy(
                prototypes, query_set_embedding, query_set_labels
            )
            output_dict["accuracy"] = accuracy

        return output_dict
