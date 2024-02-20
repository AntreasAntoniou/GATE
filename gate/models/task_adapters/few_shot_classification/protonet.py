from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from gate.boilerplate.decorators import configurable, ensemble_marker
from gate.models.backbones import GATEncoder
from gate.models.core import SourceModalityConfig, TargetModalityConfig, reinit
from gate.models.task_adapters import BaseAdapterModule
from gate.models.task_adapters.few_shot_classification.utils import (
    compute_prototypes,
    compute_prototypical_accuracy,
    compute_prototypical_logits,
    compute_prototypical_loss,
)


class DataParallelWithDict(nn.DataParallel):
    def gather(self, outputs, output_device):
        return {
            key: nn.parallel.gather([d[key] for d in outputs], output_device)
            for key in outputs[0]
        }


@configurable(group="adapter", name="fs-protonet")
class PrototypicalNetwork(BaseAdapterModule):
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
        encoder: GATEncoder,
        num_output_features: Optional[int] = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__(encoder=encoder, freeze_encoder=freeze_encoder)
        # self.stem_instance_norm = nn.InstanceNorm2d(
        #     num_features=3, affine=True
        # )
        # If num_output_features is not provided, use num_clip_features and set linear layer to identity.
        if num_output_features is None:
            self.num_output_features = self.encoder.num_in_features_image
            self.linear = nn.Identity()
        else:
            self.num_output_features = num_output_features
            self.linear = nn.Linear(
                self.encoder.num_in_features_image, num_output_features
            )
        self.build()
        print(
            f"Available GPU devices and ids are: {torch.cuda.device_count()}"
        )

    def build(self):
        support_set_inputs = torch.rand(
            (2, 2, 3, self.encoder.image_shape[0], self.encoder.image_shape[1])
        )
        query_set_inputs = torch.rand(
            (2, 2, 3, self.encoder.image_shape[0], self.encoder.image_shape[1])
        )
        support_set_labels = torch.randint(0, 1, (2, 2))
        query_set_labels: torch.Tensor = torch.randint(0, 1, (2, 2))
        dummy_batch = {
            "image": {
                "support_set": support_set_inputs,
                "query_set": query_set_inputs,
            },
            "labels": {
                "support_set": support_set_labels,
                "query_set": query_set_labels,
            },
        }
        _ = self(**dummy_batch)
        if torch.cuda.device_count() > 1:
            self.encoder_transforms_copy = deepcopy(
                self.encoder.get_transforms
            )
            self.encoder = DataParallelWithDict(self.encoder)
            setattr(
                self.encoder, "get_transforms", self.encoder_transforms_copy
            )

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

    def forward_features_no_oom(
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

        oom_error = True
        while oom_error:
            try:
                x = self.forward_features(input_dict, image)
                oom_error = False
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()

                else:
                    raise e

    def forward_features(
        self,
        image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        This method takes an input dictionary and applies the model to the input.

        Args:
            input_dict: A dictionary of input data.
            image: Optional image tensor.

        Returns:
            The output tensor after being processed by the model and the linear layer.
        """

        output = self.encoder(image=image)

        # If output is a dictionary containing the "image" key
        if isinstance(output, dict) and "image" in output:
            x = output["image"]["features"]

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
                "image": support_set_inputs.view(
                    -1, *support_set_inputs.shape[2:]
                )
            }
        )
        support_set_embedding = support_set_features.view(
            num_tasks, num_examples, -1
        )

        query_set_features = self.forward_features(
            **{"image": query_set_inputs.view(-1, *query_set_inputs.shape[2:])}
        )
        query_set_embedding = query_set_features.view(
            num_tasks, -1, query_set_features.shape[-1]
        )

        prototypes = compute_prototypes(
            support=support_set_embedding,
            labels=support_set_labels,
            num_classes=torch.max(support_set_labels) + 1,
        )

        output_dict["prototypes"] = prototypes
        output_dict["support_set_embedding"] = support_set_embedding
        output_dict["query_set_embedding"] = query_set_embedding
        output_dict["logits"] = compute_prototypical_logits(
            prototypes=prototypes, queries=query_set_embedding
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

    @ensemble_marker
    def compute_loss_and_metrics(self, logits, labels):
        loss = compute_prototypical_loss(logits=logits, labels=labels)

        accuracy = compute_prototypical_accuracy(logits=logits, labels=labels)
        return {"loss": loss, "accuracy_top_1": accuracy}

    @property
    def modality_config(self):
        return TargetModalityConfig(image=[SourceModalityConfig(image=True)])

    @property
    def encoder_transforms(self) -> Dict[str, Callable]:
        return self.encoder.get_transforms()

    def adapter_transforms(self, inputs: Union[Dict, Any]):
        inputs["image"]["support_set"] = torch.stack(
            [
                self.encoder_transforms["image"](item)
                for item in inputs["image"]["support_set"]
            ]
        )

        inputs["image"]["query_set"] = torch.stack(
            [
                self.encoder_transforms["image"](item)
                for item in inputs["image"]["query_set"]
            ]
        )

        return inputs
