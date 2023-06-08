from transformers import OneFormerConfig, OneFormerModel

# Initializing a OneFormer shi-labs/oneformer_ade20k_swin_tiny configuration
configuration = OneFormerConfig()
# Initializing a model (with random weights) from the shi-labs/oneformer_ade20k_swin_tiny style configuration
model = OneFormerModel(configuration)
# Accessing the model configuration
configuration = model.config
