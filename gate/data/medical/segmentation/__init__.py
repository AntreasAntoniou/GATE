from enum import Enum


class ACDCDatasetName(Enum):
    ACDC = "acdc"


class MD_DatasetName(Enum):
    BrainTumour: str = "medical_decathlon_brain_tumour"
    # Heart: str = "medical_decathlon_heart"
    # Liver: str = "medical_decathlon_liver"
    # Hippocampus: str = "medical_decathlon_hippocampus"
    # Prostate: str = "medical_decathlon_prostate"
    # Lung: str = "medical_decathlon_lung"
    # Pancreas: str = "medical_decathlon_pancreas"
    # HepaticVessel: str = "medical_decathlon_hepatic_vessel"
    # Spleen: str = "medical_decathlon_spleen"
    # Colon: str = "medical_decathlon_colon"
