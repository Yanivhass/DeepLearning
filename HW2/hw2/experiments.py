import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from cs236781.train_results import FitResult

from .cnn import CNN, ResNet, YourCNN
from .mlp import MLP
from .training import ClassifierTrainer
from .classifier import ArgMaxClassifier, BinaryClassifier, select_roc_thresh

DATA_DIR = os.path.expanduser("~/.pytorch-datasets")

MODEL_TYPES = {
    ###
    "cnn": CNN,
    "resnet": ResNet,
    "ycn": YourCNN,
}


def mlp_experiment(
    depth: int,
    width: int,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    dl_test: DataLoader,
    n_epochs: int,
):
    # TODO:
    #  - Create a BinaryClassifier model.
    #  - Train using our ClassifierTrainer for n_epochs, while validating on the
    #    validation set.
    #  - Use the validation set for threshold selection.
    #  - Set optimal threshold and evaluate one epoch on the test set.
    #  - Return the model, the optimal threshold value, the accuracy on the validation
    #    set (from the last epoch) and the accuracy on the test set (from a single
    #    epoch).
    #  Note: use print_every=0, verbose=False, plot=False where relevant to prevent
    #  output from this function.
    # ====== YOUR CODE: ======

    model = BinaryClassifier(
        model=MLP(
            in_dim=2,
            dims=[*[width,]*depth, 2],
            nonlins=[*['tanh',]*depth, 'lrelu']
        ),
        threshold=0.5,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    kw = dict(print_every=10, verbose=True,plot=True)
    optimizer = torch.optim.Adam(model.parameters())
    trainer = ClassifierTrainer(model, loss_fn, optimizer)

    fit_result = trainer.fit(dl_train, dl_valid, num_epochs=n_epochs,print_every=0)
    thresh = select_roc_thresh(model, *dl_valid.dataset.tensors, plot=False)
    model.threshold = thresh
    valid_acc = fit_result.test_acc[-1]
    test_result = trainer.test_epoch(dl_test)
    test_acc = test_result.accuracy
    # ========================
    return model, thresh, valid_acc, test_acc


def cnn_experiment(
    run_name,
    out_dir="./results",
    seed=None,
    device=None,
    # Training params
    bs_train=128,
    bs_test=None,
    batches=100,
    epochs=100,
    early_stopping=3,
    checkpoints=None,
    lr=1e-3,
    reg=1e-3,
    # Model params
    filters_per_layer=[64],
    layers_per_block=2,
    pool_every=2,
    hidden_dims=[1024],
    model_type="cnn",
    # You can add extra configuration for your experiments here
    print_every=1,
    **kw,
):
    """
    Executes a single run of a Part3 experiment with a single configuration.

    These parameters are populated by the CLI parser below.
    See the help string of each parameter for it's meaning.
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    # tf = torchvision.transforms.ToTensor()
    # tf = torchvision.transforms.Compose(
    #     [torchvision.transforms.ToTensor(),
    #      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # tf_train = torchvision.transforms.Compose([
    #     tf,
    #     torchvision.transforms.AutoAugment(torchvision.transforms.AutoAugmentPolicy.CIFAR10)
    # ])
    # Experimental - 04/06
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])

    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=transform_train)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=transform_test)

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]

    # TODO: Train
    #  - Create model, loss, optimizer and trainer based on the parameters.
    #    Use the model you've implemented previously, cross entropy loss and
    #    any optimizer that you wish.
    #  - Run training and save the FitResults in the fit_res variable.
    #  - The fit results and all the experiment parameters will then be saved
    #   for you automatically.
    fit_res = None
    # ====== YOUR CODE: ======
    channels = []
    # weights = [0.001]#,0.003,0.009]
    # dropout = [0.3,0.1,0.5]
    # pool = [4,2]
    [channels.extend([x] * layers_per_block) for x in filters_per_layer]
    # for weight in weights:
    #     for drop in dropout:
    #         for p in pool:
    if model_type == 'resnet':
        params = dict(
            in_size=(3, 32, 32), out_classes=10, channels=channels,
            pool_every=pool_every, hidden_dims=hidden_dims,
            activation_type='relu',  # activation_params=dict(negative_slope=0.33),
            pooling_type='max', pooling_params=dict(kernel_size=2, stride=2),
            batchnorm=True, dropout=0.2,
            bottleneck=False
        )
    elif model_type == 'cnn':
        params = dict(
            in_size=(3, 32, 32), out_classes=10, channels=channels,
            pool_every=pool_every, hidden_dims=hidden_dims,
            activation_type='relu',  # activation_params=dict(negative_slope=0.02),
            pooling_type='max', pooling_params=dict(kernel_size=2, stride=2),
            conv_params=dict(kernel_size=3, stride=1, padding=1),
        )
    elif model_type == 'ycn':
        params = dict(
            in_size=(3, 32, 32), out_classes=10, channels=channels,
            pool_every=pool_every, hidden_dims=hidden_dims,
            activation_type='relu',  # activation_params=dict(negative_slope=0.33),
            pooling_type='max', pooling_params=dict(kernel_size=2, stride=2),
            batchnorm=True, dropout=0.4,
            bottleneck=False
        )
    model = ArgMaxClassifier(model=model_cls(**params)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40, 45], gamma=0.1)
    trainer = ClassifierTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)

    pin_memory = True if device == 'cuda' else False
    dl_train = torch.utils.data.DataLoader(ds_train, bs_train, shuffle=True, pin_memory=pin_memory, num_workers=3)
    dl_test = torch.utils.data.DataLoader(ds_test, bs_test, shuffle=True, pin_memory=pin_memory, num_workers=3)

    fit_res = trainer.fit(dl_train, dl_test, num_epochs=epochs, early_stopping=early_stopping, checkpoints=checkpoints,
                          max_batches=batches, print_every=print_every)

    # ========================

    save_experiment(run_name, out_dir, cfg, fit_res)


def save_experiment(run_name, out_dir, cfg, fit_res):
    output = dict(config=cfg, results=fit_res._asdict())

    cfg_LK = (
        f'L{cfg["layers_per_block"]}_K'
        f'{"-".join(map(str, cfg["filters_per_layer"]))}'
    )
    output_filename = f"{os.path.join(out_dir, run_name)}_{cfg_LK}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"*** Output file {output_filename} written")


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)

    config = output["config"]
    fit_res = FitResult(**output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="CS236781 HW2 Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    sp_exp.set_defaults(subcmd_fn=cnn_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./results",
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )

    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=int,
        help="Save model checkpoints to this file when test " "accuracy improves",
        default=None,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    sp_exp.add_argument("--reg", type=float, help="L2 regularization", default=1e-3)

    # # Model
    sp_exp.add_argument(
        "--filters-per-layer",
        "-K",
        type=int,
        nargs="+",
        help="Number of filters per conv layer in a block",
        metavar="K",
        required=True,
    )
    sp_exp.add_argument(
        "--layers-per-block",
        "-L",
        type=int,
        metavar="L",
        help="Number of layers in each block",
        required=True,
    )
    sp_exp.add_argument(
        "--pool-every",
        "-P",
        type=int,
        metavar="P",
        help="Pool after this number of conv layers",
        required=True,
    )
    sp_exp.add_argument(
        "--hidden-dims",
        "-H",
        type=int,
        nargs="+",
        help="Output size of hidden linear layers",
        metavar="H",
        required=True,
    )
    sp_exp.add_argument(
        "--model-type",
        "-M",
        choices=MODEL_TYPES.keys(),
        default="cnn",
        help="Which model instance to create",
    )

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))
