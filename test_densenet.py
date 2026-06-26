import torch

from src.densenet import KneeOADenseNet


model = KneeOADenseNet()


print("\n" + "=" * 50)
print("TESTING DENSENET")
print("=" * 50)


dummy_input = torch.randn(
    16,
    1,
    224,
    224
)


print("\nInput shape:", dummy_input.shape)


output = model(dummy_input)


print("\nOutput shape:", output.shape)


# count trainable parameters
trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nTrainable parameters:", trainable)