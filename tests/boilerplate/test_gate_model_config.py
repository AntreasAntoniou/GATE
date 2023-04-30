from gate.models.core import SourceModalityConfig, TargetModalityConfig


def test_modality_config_creation():
    source_config = SourceModalityConfig(text=True, image=True)
    target_config = TargetModalityConfig(image=[source_config])

    assert source_config.text is True
    assert source_config.image is True
    assert source_config.audio is False
    assert source_config.video is False

    assert target_config.image == [source_config]
    assert target_config.text is None
    assert target_config.audio is None
    assert target_config.video is None


if __name__ == "__main__":
    test_modality_config_creation()
    print("All tests passed!")
