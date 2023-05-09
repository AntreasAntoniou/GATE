import pytest
import torch
from gate.models.backbones.talinet import (
    TALINet,
)


def test_talinet_forward():
    # Instantiate a TALINet object
    model = TALINet()

    # Generate some fake input data
    image = torch.randn(1, 3, 224, 224)
    text = torch.randn(1, 512)  # adjust dimensions as needed
    audio = torch.randn(1, 44100)  # adjust dimensions as needed
    video = torch.randn(1, 3, 224, 224)  # adjust dimensions as needed

    # Call the forward method
    output = model.forward(image=image, text=text, audio=audio, video=video)

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
