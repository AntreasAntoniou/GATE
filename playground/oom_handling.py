import gc

import accelerate
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers.models.resnet.modeling_resnet import (
    ImageClassifierOutputWithNoAttention,
)


# Function to print GPU memory usage
def print_memory_usage():
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
    print(
        f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB"
    )
    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2} MB")
    print(
        f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024 ** 2} MB"
    )
    print("\n")


accelerator = accelerate.Accelerator(mixed_precision="bf16")
# Use a simple pre-trained model
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

model.train()
model = accelerator.prepare(model)
# Set a large initial batch size
batch_size = 1024 * 4
image_size = (3, 224, 224)

# Create a batch of images in CPU memory
images = torch.randn((batch_size, *image_size)).cpu()
print(f"Inital batch size: {batch_size}")

while batch_size > 128:
    try:
        # Print memory usage before processing the batch
        print("Before processing batch:")
        print_memory_usage()

        # Split the batch
        batches = torch.split(images, batch_size)
        outputs = []

        for i, batch in enumerate(batches):
            # Move batch to GPU memory
            batch = batch.to(device=accelerator.device)

            # Print memory usage before forward pass
            print("Before forward pass:")
            print_memory_usage()

            output: ImageClassifierOutputWithNoAttention = model(batch).logits
            outputs.append(output.cpu())  # Move output to CPU memory

            # Print memory usage after forward pass
            output = output.detach().cpu()
            del output
            batch = batch.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            print("After forward pass:")
            print_memory_usage()

        # Print memory usage after processing the batch
        print("After processing batch:")
        print_memory_usage()

        # Concatenate the outputs
        outputs = torch.cat(outputs, dim=0)

        # If forward pass is successful, break the loop
        print(f"Forward pass successful with batch size {batch_size}")
        break
    except torch.cuda.OutOfMemoryError:
        # Halve the batch size
        model.zero_grad()

        batch_size //= 2

        # Free up memory
        del batches, outputs
        torch.cuda.empty_cache()

        gc.collect()

        print(
            f"Reducing batch size. Trying again with batch size {batch_size}"
        )
        continue
