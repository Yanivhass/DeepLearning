import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        means = torch.ones(n_features, n_classes)
        self.weights = torch.normal(means, weight_std)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.matmul(x, self.weights)
        y_pred = torch.argmax(class_scores, 1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = torch.sum(y == y_pred)/y.shape[0]
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            num_train_samples = len(dl_train) * dl_train.batch_size
            num_validation_samples = len(dl_valid) * dl_valid.batch_size
            epoch_train_loss = 0
            epoch_train_hitcount = 0
            epoch_val_loss = 0
            epoch_val_hitcount = 0

            for batch, (train_images, train_classes) in enumerate(dl_train):
                # evaluate training set
                train_batch_pred, x_scores_train = self.predict(train_images)
                wd = weight_decay * (self.weights.norm(p=2))
                train_batch_loss = loss_fn(train_images, train_classes, x_scores_train, train_batch_pred) + wd
                grad = loss_fn.grad() + weight_decay * self.weights

                # Step
                self.weights += -learn_rate * grad

                # save loss and accuracy
                epoch_train_loss += train_batch_loss
                epoch_train_hitcount += torch.sum(train_classes == train_batch_pred)

            for batch, (val_images, val_classes) in enumerate(dl_valid):
                # predict on val set
                val_batch_pred, x_scores_val = self.predict(val_images)
                wd = weight_decay * (self.weights.norm(p=2))
                val_batch_loss = loss_fn(val_images, val_classes, x_scores_val, val_batch_pred) + wd
                # save validation set loss and accuracy
                epoch_val_loss += val_batch_loss
                epoch_val_hitcount += torch.sum(val_classes == val_batch_pred)

            epoch_train_loss = epoch_train_loss / num_train_samples
            epoch_train_accuracy = (epoch_train_hitcount / num_train_samples) * 100
            epoch_val_loss = epoch_val_loss / num_validation_samples
            epoch_val_accuracy = (epoch_val_hitcount / num_validation_samples) * 100

            # save epoch results
            train_res[1].append(epoch_train_loss)
            train_res[0].append(epoch_train_accuracy)
            valid_res[1].append(epoch_val_loss)
            valid_res[0].append(epoch_val_accuracy)

            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        dims = (self.n_classes,) + img_shape
        if has_bias:
            w_images = self.weights[1:, :].transpose(0, 1).reshape(dims)
        else:
            w_images = self.weights.reshape(dims)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp["weight_std"] = 0.001
    hp["learn_rate"] = 0.001
    hp["weight_decay"] = 0.001
    # ========================

    return hp
