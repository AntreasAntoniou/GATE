from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from gate.models.core import reinit
from gate.models.task_adapters.few_shot_classification import (
    FewShotLearningClassificationEpisode,
)
from gate.models.task_adapters.few_shot_classification.utils import (
    get_accuracy,
    get_prototypes,
    prototypical_logits,
    prototypical_loss,
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

    def _process_episode(self, image, labels):
        # Get the inputs and labels

        return (
            image["support_set"],
            labels["support_set"],
            image["query_set"],
            labels["query_set"] if "query_set" in labels else None,
        )

    def forward_features(
        self,
        input_dict: Optional[Dict[str, Union[torch.Tensor, Dict]]] = None,
        image: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        This method takes an input dictionary and applies the model to the input.

        Args:
            input_dict: A dictionary of input data.
            image: Optional image tensor.

        Returns:
            The output tensor after being processed by the model and the linear layer.
        """
        x = None

        if input_dict is not None:
            x = self.model(**input_dict)[self.modality]["features"]
        if image is not None:
            x = self.model(image=image)[self.modality]["features"]

        assert x is not None, "At least one input must be provided."

        x = self.linear(x)
        return x

    def forward(
        self,
        image: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
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
        ) = self._process_episode(image, labels)

        output_dict = {}

        num_tasks, num_examples = support_set_inputs.shape[:2]
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

        query_set_features = self.forward_features(
            **{
                self.modality: query_set_inputs.view(
                    -1, *query_set_inputs.shape[2:]
                )
            }
        )
        query_set_embedding = query_set_features.view(
            num_tasks, -1, query_set_features.shape[-1]
        )

        prototypes = get_prototypes(
            embeddings=support_set_embedding,
            labels=support_set_labels,
            num_classes=int(torch.max(support_set_labels)) + 1,
        )

        output_dict["prototypes"] = prototypes
        output_dict["support_set_embedding"] = support_set_embedding
        output_dict["query_set_embedding"] = query_set_embedding
        output_dict["logits"] = prototypical_logits(
            prototypes, query_set_embedding
        )
        output_dict["labels"] = query_set_labels

        # If query set labels are provided, calculate the loss and accuracy
        if query_set_labels is not None:
            output_dict.update(
                self.compute_loss_and_metrics(
                    output_dict["logits"], output_dict["labels"]
                )
            )

        return output_dict

    def compute_loss_and_metrics(self, logits, labels):
        loss = prototypical_loss(logits, labels)
        loss = torch.mean(loss)

        accuracy = get_accuracy(logits=logits, labels=labels)
        return {"loss": loss, "accuracy_top_1": accuracy}
