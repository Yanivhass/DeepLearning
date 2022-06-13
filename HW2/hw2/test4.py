import os
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

from hw2.cnn import CNN

test_params = [
    dict(
        in_size=(3,100,100), out_classes=10,
        channels=[32]*4, pool_every=2, hidden_dims=[100]*2,
        conv_params=dict(kernel_size=3, stride=1, padding=1),
        activation_type='relu', activation_params=dict(),
        pooling_type='max', pooling_params=dict(kernel_size=2),
    ),
    dict(
        in_size=(3,100,100), out_classes=10,
        channels=[32]*4, pool_every=2, hidden_dims=[100]*2,
        conv_params=dict(kernel_size=5, stride=2, padding=3),
        activation_type='lrelu', activation_params=dict(negative_slope=0.05),
        pooling_type='avg', pooling_params=dict(kernel_size=3),
    ),
    dict(
        in_size=(3,100,100), out_classes=3,
        channels=[16]*5, pool_every=3, hidden_dims=[100]*1,
        conv_params=dict(kernel_size=2, stride=2, padding=2),
        activation_type='lrelu', activation_params=dict(negative_slope=0.1),
        pooling_type='max', pooling_params=dict(kernel_size=2),
    ),
]
#
# for i, params in enumerate(test_params):
#     torch.manual_seed(seed)
#     net = CNN(**params)
#     print(f"\n=== test {i=} ===")
#     print(net)
#
#     torch.manual_seed(seed)
#     test_out = net(torch.ones(1, 3, 100, 100))
#     print(f'{test_out=}')
#
#     expected_out = torch.load(f'tests/assets/expected_conv_out_{i:02d}.pt')
#     print(f'max_diff={torch.max(torch.abs(test_out - expected_out)).item()}')
#     test.assertTrue(torch.allclose(test_out, expected_out, atol=1e-3))

from hw2.classifier import ArgMaxClassifier
model = ArgMaxClassifier(model=CNN(**test_params[0]))

test_image = torch.randint(low=0, high=256, size=(3, 100, 100), dtype=torch.float).unsqueeze(0)
test.assertEqual(model.classify(test_image).shape, (1,))
test.assertEqual(model.predict_proba(test_image).shape, (1, 10))
test.assertAlmostEqual(torch.sum(model.predict_proba(test_image)).item(), 1.0, delta=1e-3)

data_dir = os.path.expanduser('~/.pytorch-datasets')
ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())

print(f'Train: {len(ds_train)} samples')
print(f'Test: {len(ds_test)} samples')

x0,_ = ds_train[0]
in_size = x0.shape
num_classes = 10
print('input image size =', in_size)


###### Train/Test

from hw2.training import ClassifierTrainer
from hw2.answers import part4_optim_hp

torch.manual_seed(seed)

# Define a tiny part of the CIFAR-10 dataset to overfit it
batch_size = 2
max_batches = 25
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)

# Create model, loss and optimizer instances
model = ArgMaxClassifier(
    model=CNN(
        in_size, num_classes, channels=[32], pool_every=1, hidden_dims=[100],
        conv_params=dict(kernel_size=3, stride=1, padding=1),
        pooling_params=dict(kernel_size=2),
    )
)

hp_optim = part4_optim_hp()
loss_fn = hp_optim.pop('loss_fn')
optimizer = torch.optim.SGD(params=model.parameters(), **hp_optim)

# Use ClassifierTrainer to run only the training loop a few times.
trainer = ClassifierTrainer(model, loss_fn, optimizer, device)
best_acc = 0
for i in range(25):
    res = trainer.train_epoch(dl_train, max_batches=max_batches, verbose=(i % 5 == 0))
    best_acc = res.accuracy if res.accuracy > best_acc else best_acc

# Test overfitting
test.assertGreaterEqual(best_acc, 90)


from hw2.cnn import ResidualBottleneckBlock

torch.manual_seed(seed)
resblock_bn = ResidualBottleneckBlock(
    in_out_channels=256, inner_channels=[64, 32, 64], inner_kernel_sizes=[3, 5, 3],
    batchnorm=False, dropout=0.1, activation_type="lrelu"
)
print(resblock_bn)

# Test a forward pass
torch.manual_seed(seed)
test_in  = torch.ones(1, 256, 32, 32)
test_out = resblock_bn(test_in)
print(f'{test_out.shape=}')
assert test_out.shape == test_in.shape

expected_out = torch.load('tests/assets/expected_resblock_bn_out.pt')
print(f'max_diff={torch.max(torch.abs(test_out - expected_out)).item()}')
test.assertTrue(torch.allclose(test_out, expected_out, atol=1e-3))