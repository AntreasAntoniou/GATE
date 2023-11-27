from enum import Enum


class DatasetName(Enum):
    HMDB51_GULPRGB = "hmdb51-gulprgb"
    # UCF_101_GULPRGB = "ucf-101-gulprgb"
    # EPIC_KITCHENS_100_GULPRGB = "epic-kitchens-100-gulprgb"
    # KINETICS_400 = "kinetics-400"


class RegressionDatasetName(Enum):
    IWILDCAM_2022 = "iwildcam_2022"
