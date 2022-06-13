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

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

from hw2.experiments import load_experiment, cnn_experiment
from cs236781.plot import plot_fit
from hw2.cnn import ResNet


test_image = torch.randint(low=0, high=256, size=(3, 100, 100), dtype=torch.float).unsqueeze(0)
test_image = test_image.to(device)



# # Test experiment1 implementation on a few data samples and with a small model
# cnn_experiment(
#     'test_run', seed=seed, bs_train=50, batches=10, epochs=10, early_stopping=5,
#     filters_per_layer=[32,64], layers_per_block=1, pool_every=1, hidden_dims=[100],
#     model_type='resnet',
# )
#
# # There should now be a file 'test_run.json' in your `results/` folder.
# # We can use it to load the results of the experiment.
# cfg, fit_res = load_experiment('results/test_run_L1_K32-64.json')
# _, _ = plot_fit(fit_res, train_test_overlay=True)
#
# # And `cfg` contains the exact parameters to reproduce it
# print('experiment config: ', cfg)
def plot_exp_results(filename_pattern, results_dir='results'):
    fig = None
    result_files = glob.glob(os.path.join(results_dir, filename_pattern))
    result_files.sort()
    if len(result_files) == 0:
        print(f'No results found for pattern {filename_pattern}.', file=sys.stderr)
        return
    for filepath in result_files:
        m = re.match('exp\d_(\d_)?(.*)\.json', os.path.basename(filepath))
        cfg, fit_res = load_experiment(filepath)
        fig, axes = plot_fit(fit_res, fig, legend=m[2],log_loss=True)
    del cfg['filters_per_layer']
    del cfg['layers_per_block']
    print('common config: ', cfg)


# K = [32,64]
# L = [2,4,8,16]
# for k in K:
#     for layers in L:
#         cnn_experiment(
#             f'exp1_1', seed=seed, bs_train=200, batches=150, epochs=10, early_stopping=5,
#             filters_per_layer=[k], layers_per_block=layers, pool_every=3, hidden_dims=[100],
#             model_type='cnn',
#         )
#
# K = [32,64,128,256]
# L = [2,4,8]
# for k in K:
#     for layers in L:
#         cnn_experiment(
#             f'exp1_2', seed=seed, bs_train=100, batches=300, epochs=10, early_stopping=5,
#             filters_per_layer=[k], layers_per_block=layers, pool_every=3, hidden_dims=[100],
#             model_type='cnn',
#         )
#
#
# K = [32,64]
# L = [2,4,8,16]
# for k in K:
#     for layers in L:
#         cnn_experiment(
#             f'exp1_2', seed=seed, bs_train=200, batches=150, epochs=10, early_stopping=5,
#             filters_per_layer=[k], layers_per_block=layers, pool_every=3, hidden_dims=[100],
#             model_type='cnn',
#         )
#
# ######## exp 1.4
# K = [32]
# L = [8,16,32]
# for layers in L:
#     cnn_experiment(
#         f'exp1_4', seed=seed, bs_train=100, batches=300, epochs=10, early_stopping=5,
#          filters_per_layer=[K], layers_per_block=layers, pool_every=4, hidden_dims=[100],
#          model_type='resnet',
#     )
#
# K = [64,128,256]
# L = [2,4,8]
# for layers in L:
#     cnn_experiment(
#         f'exp1_4', seed=seed, bs_train=100, batches=300, epochs=10, early_stopping=5,
#          filters_per_layer=[K], layers_per_block=layers, pool_every=4, hidden_dims=[100],
#          model_type='resnet',
#     )
#
#
from hw2.cnn import YourCNN
#
# net = YourCNN((3,100,100), 10, channels=[32]*4, pool_every=2, hidden_dims=[100]*2)
# print(net)
#
# test_image = torch.randint(low=0, high=256, size=(3, 100, 100), dtype=torch.float).unsqueeze(0)
# test_out = net(test_image)
# print('out =', test_out)

# K = [32, 64, 128]
# L = [6,9,12,15]#[3,6,9,12,15]
# for layers in L:
#     cnn_experiment(
#         f'exp2', seed=seed, bs_train=150, batches=300, epochs=30, early_stopping=5,
#          filters_per_layer=K, layers_per_block=layers, pool_every=layers, hidden_dims=[100],
#          model_type='resnet',
#     )

K = [32, 64, 128]
L = [5]

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

# original
# K = [32, 64, 128]
# L = [3,6,9,12]
# for layers in L:
#     cnn_experiment(
#             f'exp2', seed=seed, bs_train=100, batches=300, epochs=10, early_stopping=5,
#              filters_per_layer=K, layers_per_block=layers, pool_every=layers, hidden_dims=[100],
#              model_type='cnn',
#         )
#
#
# K = [32, 64, 128]
# L = [5]
# for layers in L:
#     cnn_experiment(
#         f'exp2',
#         seed=seed,
#         bs_train=128,
#         batches=500,
#         epochs=40,
#         early_stopping=5,
#         lr=5e-3,
#         reg=1e-8,
#         filters_per_layer=K,
#         layers_per_block=layers,
#         pool_every=layers,
#         hidden_dims=[100],
#         model_type='ycn',
#     )

if __name__ == '__main__':
    # next next run - 7 layers, 200 epochs, [32, 64, 128, 256]
    # next run - no 32
    K = [64, 128, 256]
    L = [9]
    for layers in L:
        cnn_experiment(
            f'exp2',
            seed=seed,
            bs_train=128,
            batches=500,
            epochs=200,
            early_stopping=15,
            lr=5e-3,
            reg=1e-8,
            filters_per_layer=K,
            layers_per_block=layers,
            pool_every=layers,
            hidden_dims=[100],
            model_type='ycn',
        )
