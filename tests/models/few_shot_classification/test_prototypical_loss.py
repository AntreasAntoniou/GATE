# Importing the necessary libraries
import torch
import torch.nn.functional as F

from gate.models.task_adapters.few_shot_classification.utils import (
    prototypical_logits,
    prototypical_loss,
)


# Define a test function to validate the prototypical_loss function
def test_prototypical_loss():
    # Create a batch of prototypes
    prototypes = torch.tensor(
        [[[0.5, 0.6], [0.1, 0.2]], [[0.7, 0.8], [0.3, 0.4]]],
        dtype=torch.float32,
    )  # shape (2, 2, 2)

    # Create a batch of embeddings
    embeddings = torch.tensor(
        [[[0.55, 0.65], [0.15, 0.25]], [[0.75, 0.85], [0.35, 0.45]]],
        dtype=torch.float32,
    )  # shape (2, 2, 2)

    # Create labels for the embeddings
    labels = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # shape (2, 2)

    # Compute the loss using the prototypical_loss function
    logits = prototypical_logits(prototypes, embeddings)
    loss = prototypical_loss(logits, labels)

    # The loss should be a single scalar value, i.e., a float
    assert isinstance(
        loss.item(), float
    ), f"Expected loss to be a float, got {type(loss.item())}"
