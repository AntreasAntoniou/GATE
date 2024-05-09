import timm

model = timm.create_model(
    model_name="efficientformerv2_s0",
    pretrained=True,
    features_only=True,
    img_size=1024,
)

print(model)
