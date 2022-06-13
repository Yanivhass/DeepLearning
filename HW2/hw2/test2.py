import os
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torchvision
import torchvision.transforms as tvtf



#%%
seed = 42
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

import hw2.optimizers as optimizers
help(optimizers.Optimizer)

# Test VanillaSGD
torch.manual_seed(42)
p = torch.randn(500, 10)
dp = torch.randn(*p.shape)*2
params = [(p, dp)]

vsgd = optimizers.VanillaSGD(params, learn_rate=0.5, reg=0.1)
vsgd.step()

expected_p = torch.load('../tests/assets/expected_vsgd.pt')
diff = torch.norm(p-expected_p).item()
print(f'diff={diff}')
test.assertLess(diff, 1e-3)

#### Train
import hw2.training as training

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
data_dir = os.path.expanduser('~/.pytorch-datasets')
ds_train = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=True, transform=tvtf.ToTensor())
ds_test = torchvision.datasets.CIFAR10(root=data_dir, download=True, train=False, transform=tvtf.ToTensor())

print(f'Train: {len(ds_train)} samples')
print(f'Test: {len(ds_test)} samples')

import hw2.layers as layers
import hw2.answers as answers
from torch.utils.data import DataLoader

# Overfit to a very small dataset of 20 samples
batch_size = 10
max_batches = 2
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)

# Get hyperparameters
hp = answers.part2_overfit_hp()

torch.manual_seed(seed)

# record = {"accuracy": 0, 'lr': 0, 'reg': 0}
# for i in range(100):
#     for j in range(15):
#         hp['reg'] = torch.rand(1)
#         hp['lr'] = torch.rand(1)
#         # Build a model and loss using our custom MLP and CE implementations
#         model = layers.MLP(3 * 32 * 32, num_classes=10, hidden_features=[128] * 3, wstd=hp['wstd'])
#         loss_fn = layers.CrossEntropyLoss()
#
#         # Use our custom optimizer
#         optimizer = optimizers.VanillaSGD(model.params(), learn_rate=hp['lr'], reg=hp['reg'])
#
#         # Run training over small dataset multiple times
#         trainer = training.LayerTrainer(model, loss_fn, optimizer)
#         best_acc = 0
#         for i in range(20):
#             res = trainer.train_epoch(dl_train, max_batches=max_batches,verbose=False)
#             best_acc = res.accuracy if res.accuracy > best_acc else best_acc
#         if best_acc > record['accuracy']:
#             record['accuracy'] = best_acc
#             record['lr'] = hp['lr']
#             record['reg'] = hp['reg']
#             print(record)
#
# test.assertGreaterEqual(best_acc, 98)

# Define a larger part of the CIFAR-10 dataset (still not the whole thing)
batch_size = 50
max_batches = 100
in_features = 3*32*32
num_classes = 10
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size//2, shuffle=False)


# Define a function to train a model with our Trainer and various optimizers
def train_with_optimizer(opt_name, opt_class, hp):
    torch.manual_seed(seed)

    # Get hyperparameters
    # hp = answers.part2_optim_hp()
    hidden_features = [128] * 5
    num_epochs = 10

    # Create model, loss and optimizer instances
    model = layers.MLP(in_features, num_classes, hidden_features, wstd=hp['wstd'])
    loss_fn = layers.CrossEntropyLoss()
    optimizer = opt_class(model.params(), learn_rate=hp[f'lr_{opt_name}'], reg=hp['reg'])

    # Train with the Trainer
    trainer = training.LayerTrainer(model, loss_fn, optimizer)
    fit_res = trainer.fit(dl_train, dl_test, num_epochs, max_batches=max_batches, print_every=100)

    # fig, axes = plot_fit(fit_res, fig=fig, legend=opt_name)
    return fit_res

# record = {"accuracy": 0, 'lr': 0, 'reg': 0}
# for i in range(100):
#     for j in range(15):
#         print(record)
#         hp = answers.part2_optim_hp()
#         hp['reg'] = torch.rand(1)*0.1
#         hp['lr_vanilla'] = torch.rand(1)*0.1
#         print('current lr: ' + str(hp['lr_vanilla'].item()) + ' reg: ' + str(hp['reg'].item()))
#         fig_optim = train_with_optimizer('vanilla', optimizers.VanillaSGD, hp)
#         if max(fig_optim.test_acc) > record['accuracy']:
#             record['accuracy'] = max(fig_optim.test_acc)
#             record['lr'] = hp['lr_vanilla']
#             record['reg'] = hp['reg']
#             print(record)
#
# print(record)
hp = answers.part2_optim_hp()
# fig_optim = train_with_optimizer('momentum', optimizers.MomentumSGD, hp)
# fig_optim = train_with_optimizer('rmsprop', optimizers.RMSProp, hp)

### RMSPROP testing

# record = {"accuracy": 0, 'lr': 0, 'reg': 0}
# for i in range(100):
#     for j in range(15):
#         print(record)
#         hp = answers.part2_optim_hp()
#         hp['reg'] = torch.rand(1)*0.005
#         hp['lr_rmsprop'] = torch.rand(1)*0.0002
#         print('current lr: ' + str(hp['lr_rmsprop'].item()) + ' reg: ' + str(hp['reg'].item()))
#         fig_optim = train_with_optimizer('rmsprop', optimizers.RMSProp, hp)
#         if max(fig_optim.test_acc) > record['accuracy']:
#             record['accuracy'] = max(fig_optim.test_acc)
#             record['lr'] = hp['lr_rmsprop']
#             record['reg'] = hp['reg']
#             print(record)
#
# print(record)
# {'accuracy': 19.04, 'lr': tensor([0.0005]), 'reg': tensor([0.0044])}
# {'accuracy': 23.64, 'lr': tensor([0.0001]), 'reg': tensor([0.0002])}

######### dropout

from hw2.grad_compare import compare_layer_to_torch

# Check architecture of MLP with dropout layers
mlp_dropout = layers.MLP(in_features, num_classes, [50]*3, dropout=0.6)
print(mlp_dropout)
test.assertEqual(len(mlp_dropout.sequence), 10)
for b1, b2 in zip(mlp_dropout.sequence, mlp_dropout.sequence[1:]):
    if str(b1).lower() == 'relu':
        test.assertTrue(str(b2).startswith('Dropout'))
test.assertTrue(str(mlp_dropout.sequence[-1]).startswith('Linear'))

# Test end-to-end gradient in train and test modes.
print('Dropout, train mode')
mlp_dropout.train(True)
for diff in compare_layer_to_torch(mlp_dropout, torch.randn(500, in_features)):
    test.assertLess(diff, 1e-3)

print('Dropout, test mode')
mlp_dropout.train(False)
for diff in compare_layer_to_torch(mlp_dropout, torch.randn(500, in_features)):
    test.assertLess(diff, 1e-3)