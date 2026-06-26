import torch

from src.model import KneeOACNN


# create model
model = KneeOACNN()


print("\n" + "=" * 50)
print("TESTING MODEL")
print("=" * 50)


# fake batch
dummy_input = torch.randn(
    32,   # batch size
    1,    # grayscale channel
    224,  # height
    224   # width
)


print("\nInput shape:", dummy_input.shape)


# forward pass
output = model(dummy_input)


print("\nOutput shape:", output.shape)


print("\nOutput tensor:\n")
print(output)