import accelerate
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from gate.models import build_model


def test_clip_with_linear_forward():
    """
    Test the forward method of the CLIPWithLinear.
    """
    accelerator = accelerate.Accelerator()

    model_and_transform = build_model()
    x_dummy = torch.rand(16, 3, 224, 224).to(accelerator.device)
    y_dummy = torch.randint(0, 100, (16,)).to(accelerator.device)

    input_dict = {
        "pixel_values": torch.cat(
            [
                model_and_transform.transform({"image": x, "labels": y})[
                    "input_images"
                ].to(accelerator.device)
                for x, y in zip(x_dummy, y_dummy)
            ],
            dim=0,
        ),
    }

    model = model_and_transform.model.to(accelerator.device)
    output = model(input_dict)

    assert output.shape == (16, 100)
