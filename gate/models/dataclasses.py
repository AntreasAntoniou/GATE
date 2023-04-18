from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SourceModalityConfig:
    image: bool = False
    text: bool = False
    audio: bool = False
    video: bool = False


class TargetModalityConfig:
    image: Optional[List[SourceModalityConfig]] = None
    text: Optional[List[SourceModalityConfig]] = None
    audio: Optional[List[SourceModalityConfig]] = None
    video: Optional[List[SourceModalityConfig]] = None


@dataclass
class GATEModalityConfig:
    target_modality: str
    source_modalities: str
