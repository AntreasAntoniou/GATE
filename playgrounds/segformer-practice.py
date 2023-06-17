import requests
from PIL import Image
from rich.table import Table
from rich.traceback import pretty
from torch import nn
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)


def pretty_print_parameters(model: nn.Module):
    from rich import print as rprint

    table = Table(title="Model Parameters")

    table.add_column("Name", justify="left")
    table.add_column("Shape", justify="center")
    table.add_column("Data Type", justify="center")
    table.add_column("Device", justify="center")

    for name, param in model.named_parameters():
        table.add_row(
            str(name),
            str(tuple(param.shape)),
            str(param.dtype),
            str(param.device),
        )

    rprint(table)
    return table


processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

pretty_print_parameters(model)
