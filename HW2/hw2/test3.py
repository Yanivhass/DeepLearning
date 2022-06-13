import os
import re
import sys
import glob
import unittest
from typing import Sequence, Tuple

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as tvtf
from torch import Tensor

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def rotate_2d(X, deg=0):
    """
    Rotates each 2d sample in X of shape (N, 2) by deg degrees.
    """
    a = np.deg2rad(deg)
    return X @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T


def plot_dataset_2d(X, y, n_classes=2, alpha=0.2, figsize=(8, 6), title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    for c in range(n_classes):
        ax.scatter(*X[y == c, :].T, alpha=alpha, label=f"class {c}");

    ax.set_xlabel("$x_1$");
    ax.set_ylabel("$x_2$");
    ax.legend();
    ax.set_title((title or '') + f" (n={len(y)})")

np.random.seed(seed)

N = 10_000
N_train = int(N * .8)

# Create data from two different distributions for the training/validation
X1, y1 = make_moons(n_samples=N_train//2, noise=0.2)
X1 = rotate_2d(X1, deg=10)
X2, y2 = make_moons(n_samples=N_train//2, noise=0.25)
X2 = rotate_2d(X2, deg=50)

# Test data comes from a similar but noisier distribution
X3, y3 = make_moons(n_samples=(N-N_train), noise=0.3)
X3 = rotate_2d(X3, deg=40)

X, y = np.vstack([X1, X2, X3]), np.hstack([y1, y2, y3])

# Train and validation data is from mixture distribution
X_train, X_valid, y_train, y_valid = train_test_split(X[:N_train, :], y[:N_train], test_size=1/3, shuffle=False)

# Test data is only from the second distribution
X_test, y_test = X[N_train:, :], y[N_train:]

# fig, ax = plt.subplots(1, 3, figsize=(20, 5))
# plot_dataset_2d(X_train, y_train, title='Train', ax=ax[0])
# plot_dataset_2d(X_valid, y_valid, title='Validation', ax=ax[1])
# plot_dataset_2d(X_test, y_test, title='Test', ax=ax[2])

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

batch_size = 32

dl_train, dl_valid, dl_test = [
    DataLoader(
        dataset=TensorDataset(
            torch.from_numpy(X_).to(torch.float32),
            torch.from_numpy(y_)
        ),
        shuffle=True,
        num_workers=0,
        batch_size=batch_size
    )
    for X_, y_ in [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)]
]

print(f'{len(dl_train.dataset)=}, {len(dl_valid.dataset)=}, {len(dl_test.dataset)=}')

from hw2.mlp import MLP

mlp = MLP(
    in_dim=2,
    dims=[8, 16, 32, 64],
    nonlins=['relu', 'tanh', nn.LeakyReLU(0.314), 'softmax']
)

x0, y0 = next(iter(dl_train))

yhat0 = mlp(x0)

test.assertEqual(len([*mlp.parameters()]), 8)
test.assertEqual(yhat0.shape, (batch_size, mlp.out_dim))
test.assertTrue(torch.allclose(torch.sum(yhat0, dim=1), torch.tensor(1.0)))
test.assertIsNotNone(yhat0.grad_fn)


from hw2.classifier import BinaryClassifier

bmlp4 = BinaryClassifier(
    model=MLP(in_dim=2, dims=[*[10]*3, 2], nonlins=[*['relu']*3, 'none']),
    threshold=0.5
)
print(bmlp4)

# Test model
test.assertEqual(len([*bmlp4.parameters()]), 8)
test.assertIsNotNone(bmlp4(x0).grad_fn)

# Test forward
yhat0_scores = bmlp4(x0)
test.assertEqual(yhat0_scores.shape, (batch_size, 2))
test.assertFalse(torch.allclose(torch.sum(yhat0_scores, dim=1), torch.tensor(1.0)))

# Test predict_proba
yhat0_proba = bmlp4.predict_proba(x0)
test.assertEqual(yhat0_proba.shape, (batch_size, 2))
test.assertTrue(torch.allclose(torch.sum(yhat0_proba, dim=1), torch.tensor(1.0)))

# Test classify
yhat0 = bmlp4.classify(x0)
test.assertEqual(yhat0.shape, (batch_size,))
test.assertEqual(yhat0.dtype, torch.int)
test.assertTrue(all(yh_ in (0, 1) for yh_ in yhat0))

from hw2.training import ClassifierTrainer
from hw2.answers import part3_arch_hp, part3_optim_hp

torch.manual_seed(seed)

hp_arch = part3_arch_hp()
hp_optim = part3_optim_hp()

model = BinaryClassifier(
    model=MLP(
        in_dim=2,
        dims=[*[hp_arch['hidden_dims'],]*hp_arch['n_layers'], 2],
        nonlins=[*[hp_arch['activation'],]*hp_arch['n_layers'], hp_arch['out_activation']]
    ),
    threshold=0.5,
)
print(model)

loss_fn = hp_optim.pop('loss_fn')
optimizer = torch.optim.SGD(params=model.parameters(), **hp_optim)
trainer = ClassifierTrainer(model, loss_fn, optimizer)

fit_result = trainer.fit(dl_train, dl_valid, num_epochs=20, print_every=10)

test.assertGreaterEqual(fit_result.train_acc[-1], 85.0)
test.assertGreaterEqual(fit_result.test_acc[-1], 75.0)


from hw2.classifier import select_roc_thresh
optimal_thresh = select_roc_thresh(model, *dl_valid.dataset.tensors, plot=True)

from itertools import product
from tqdm.auto import tqdm
from hw2.experiments import mlp_experiment

torch.manual_seed(seed)

depths = [1, 2, 4]
widths = [2, 8, 32, 128]
exp_configs = product(enumerate(widths), enumerate(depths))
# fig, axes = plt.subplots(len(widths), len(depths), figsize=(10 * len(depths), 10 * len(widths)), squeeze=False)
test_accs = []

for (i, width), (j, depth) in tqdm(list(exp_configs)):
    model, thresh, valid_acc, test_acc = mlp_experiment(
        depth, width, dl_train, dl_valid, dl_test, n_epochs=10
    )
    test_accs.append(test_acc)
    # fig, ax = plot_decision_boundary_2d(model, *dl_test.dataset.tensors, ax=axes[i, j])
    # ax.set_title(f"{depth=}, {width=}")
    # ax.text(ax.get_xlim()[0] * .95, ax.get_ylim()[1] * .95, f"{thresh=:.2f}\n{valid_acc=:.1f}%\n{test_acc=:.1f}%",
    #         va="top")

# Assert minimal performance requirements.
# You should be able to do better than these by at least 5%.
test.assertGreaterEqual(np.min(test_accs), 75.0)
test.assertGreaterEqual(np.quantile(test_accs, 0.75), 85.0)
