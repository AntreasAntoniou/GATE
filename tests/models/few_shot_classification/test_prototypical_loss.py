# Importing the necessary libraries
import torch

from gate.models.task_adapters.few_shot_classification.utils import (
    compute_prototypical_logits, compute_prototypical_loss)


# Define a test function to validate the prototypical_loss function
def test_prototypical_loss():
    # Create a batch of prototypes
    prototypes = torch.randn(10, 128)

    # Create a batch of embeddings
    embeddings = torch.randn(2, 128)

    # Create labels for the embeddings
    labels = torch.randint(0, 10, (2,))

    # Compute the loss using the prototypical_loss function
    logits = compute_prototypical_logits(prototypes, embeddings)
    loss = compute_prototypical_loss(logits, labels)

    # The loss should be a single scalar value, i.e., a float
    assert isinstance(
        loss.item(), float
    ), f"Expected loss to be a float, got {type(loss.item())}"
