

import torch
import unittest

# %load_ext autoreload
# %autoreload 2

test = unittest.TestCase()

import hw2.layers as layers
from hw2.grad_compare import compare_layer_to_torch


def test_block_grad(block: layers.Layer, x, y=None, delta=1e-3):
    diffs = compare_layer_to_torch(block, x, y)

    # Assert diff values
    for diff in diffs:
        test.assertLess(diff, delta)


# Show the compare function
# compare_layer_to_torch??

N = 100
in_features = 200
num_classes = 10
eps = 1e-6
# Test LeakyReLU
alpha = 0.1
lrelu = layers.LeakyReLU(alpha=alpha)
x_test = torch.randn(N, in_features)

# Test forward pass
z = lrelu(x_test)
test.assertSequenceEqual(z.shape, x_test.shape)
test.assertTrue(torch.allclose(z, torch.nn.LeakyReLU(alpha)(x_test), atol=eps))

# Test backward pass
test_block_grad(lrelu, x_test)


# Test Linear
out_features = 1000
fc = layers.Linear(in_features, out_features)
x_test = torch.randn(N, in_features)

# Test forward pass
z = fc(x_test)
test.assertSequenceEqual(z.shape, [N, out_features])
torch_fc = torch.nn.Linear(in_features, out_features,bias=True)
torch_fc.weight = torch.nn.Parameter(fc.w)
torch_fc.bias = torch.nn.Parameter(fc.b)
test.assertTrue(torch.allclose(torch_fc(x_test), z, atol=eps))

# Test backward pass
test_block_grad(fc, x_test)

# Test second backward pass
x_test = torch.randn(N, in_features)
z = fc(x_test)
z = fc(x_test)
test_block_grad(fc, x_test)


######## Cross Entropy

# Test CrossEntropy
cross_entropy = layers.CrossEntropyLoss()
scores = torch.randn(N, num_classes)
labels = torch.randint(low=0, high=num_classes, size=(N,), dtype=torch.long)

# Test forward pass
loss = cross_entropy(scores, labels)
expected_loss = torch.nn.functional.cross_entropy(scores, labels)
test.assertLess(torch.abs(expected_loss-loss).item(), 1e-5)
print('loss=', loss.item())

# Test backward pass
test_block_grad(cross_entropy, scores, y=labels)


# Test Sequential
# Let's create a long sequence of layers and see
# whether we can compute end-to-end gradients of the whole thing.

seq = layers.Sequential(
    layers.Linear(in_features, 100),
    layers.Linear(100, 200),
    layers.Linear(200, 100),
    layers.ReLU(),
    layers.Linear(100, 500),
    layers.LeakyReLU(alpha=0.01),
    layers.Linear(500, 200),
    layers.ReLU(),
    layers.Linear(200, 500),
    layers.LeakyReLU(alpha=0.1),
    layers.Linear(500, 1),
    layers.Sigmoid(),
)
x_test = torch.randn(N, in_features)

# Test forward pass
z = seq(x_test)
test.assertSequenceEqual(z.shape, [N, 1])

# Test backward pass
test_block_grad(seq, x_test)

# Create an MLP model
mlp = layers.MLP(in_features, num_classes, hidden_features=[100, 50, 100])
print(mlp)

# Test MLP architecture
N = 100
in_features = 10
num_classes = 10
for activation in ('relu', 'sigmoid'):
    mlp = layers.MLP(in_features, num_classes, hidden_features=[100, 50, 100], activation=activation)
    test.assertEqual(len(mlp.sequence), 7)

    num_linear = 0
    for b1, b2 in zip(mlp.sequence, mlp.sequence[1:]):
        if (str(b2).lower() == activation):
            test.assertTrue(str(b1).startswith('Linear'))
            num_linear += 1

    test.assertTrue(str(mlp.sequence[-1]).startswith('Linear'))
    test.assertEqual(num_linear, 3)

    # Test MLP gradients
    # Test forward pass
    x_test = torch.randn(N, in_features)
    labels = torch.randint(low=0, high=num_classes, size=(N,), dtype=torch.long)
    z = mlp(x_test)
    test.assertSequenceEqual(z.shape, [N, num_classes])

    # Create a sequence of MLPs and CE loss
    seq_mlp = layers.Sequential(mlp, layers.CrossEntropyLoss())
    loss = seq_mlp(x_test, y=labels)
    test.assertEqual(loss.dim(), 0)
    print(f'MLP loss={loss}, activation={activation}')

    # Test backward pass
    test_block_grad(seq_mlp, x_test, y=labels)