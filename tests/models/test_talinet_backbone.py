import pytest
import torch
from gate.models.backbones.talinet import (
    TALINet,
)


def test_talinet_forward():
    # Instantiate a TALINet object
    model = TALINet()
    transforms = model.get_transforms()
    # Generate some fake input data
    image = torch.randint(low=0, high=255, size=(1, 3, 224, 224))
    text = [
        ["Hello my dude let's do a test"],
    ]
    audio = torch.randn(1, 44000)  # adjust dimensions as needed
    video = torch.randint(low=0, high=255, size=(1, 10, 3, 224, 224))

    input_dict = {
        "image": transforms["image"](image),
        "text": transforms["text"](text),
        "audio": transforms["audio"](audio),
        "video": transforms["video"](video),
    }

    # Call the forward method
    output = model.forward(**input_dict)

    # Check that the output is a dictionary
    assert isinstance(output, dict)

    # Check that the dictionary has keys for each modality
    assert "image_features" in output
    assert "image_projection_output" in output
    assert "text_features" in output
    assert "text_projection_output" in output
    assert "audio_features" in output
    assert "audio_projection_output" in output
    assert "video_features" in output
    assert "video_projection_output" in output

    # More detailed checks can be added here, depending on the expected properties of the output


# test case for no input modalities
def test_talinet_no_modalities():
    # Instantiate a TALINet object
    model = TALINet()

    # Call the forward method with no input modalities
    with pytest.raises(ValueError):
        model.forward()
