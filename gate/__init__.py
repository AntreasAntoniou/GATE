import warnings

# Ignore deprecation warnings from pkg_resources
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="pkg_resources"
)

# Ignore deprecation warnings from torchvision
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="torchvision"
)

# Ignore deprecation warnings from PIL (Pillow)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL")
