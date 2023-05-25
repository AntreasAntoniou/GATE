import torch

# Define batch size, number of subjects, and feature size
b = 10
subject_id = 5
features = 100

# Create random image and text features
image_features = torch.rand(b, subject_id, features)
text_features = torch.rand(b, subject_id, features)

# Reshape features for dot product
image_features_reshaped = image_features.view(b * subject_id, features)
text_features_reshaped = text_features.view(b * subject_id, features)

# Compute dot product
dot_product = torch.mm(image_features_reshaped, text_features_reshaped.t())

# Reshape the dot product back to original dimensions
dot_product_reshaped = dot_product.view(b, subject_id, b, subject_id)

# True labels are 1 for same subject_id in same batch and 0 otherwise
true_labels = (
    torch.eye(subject_id)
    .unsqueeze(0)
    .unsqueeze(2)
    .expand(b, subject_id, b, subject_id)
)

# Predicted labels are 1 for higher dot product within batch than between batches
# Here we compare max dot product within batch (excluding self) with max dot product between batches

# Copy tensor before fill_diagonal_
dot_product_copy = dot_product_reshaped.clone()

# Fill diagonal of each b x b submatrix
for i in range(b):
    dot_product_copy[i, :, i, :].fill_diagonal_(float("-inf"))

between_batch_dot_product, _ = dot_product_copy.max(dim=[2, 3])

# Within batch dot product, ignoring diagonal
dot_product_reshaped.fill_diagonal_(float("-inf"))
within_batch_dot_product, _ = dot_product_reshaped.max(dim=-1)

pred_labels = (
    (within_batch_dot_product > between_batch_dot_product)
    .float()
    .unsqueeze(-1)
)

# Calculate accuracy
accuracy = (pred_labels == true_labels).float().mean()

print(f"Accuracy: {accuracy.item()}")
