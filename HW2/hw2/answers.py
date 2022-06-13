r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
1.1. The size of the Jacobian would be 512x1024x64 (Ysize x Xsize x N)
1.2. In a linear layer all input elements are used for the calculation of every
output element. This means the gradient would be calculated for all of them, and even
for input elements with a very small gradient we would still have a value != 0 
1.3. Yes, the Jacobian is needed. Calculation of the gradients is done using the chain rule.
In order to calculate the loss's derivative w.r.t X we first have to calculate Y's derivative
w.r.t X. 

2.1. The size of the Jacobian would be 512x1024x5125x64 (Ysize x Wsize x N)
2.2. same answer as before
2.3. same answer as before 

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**

Descent based algorithms require use of the gradients in order to move with the steepest
descent. In order to calculate the gradients we may use other approaches, e.g. calculating
the derivative by it's definition: ${(f(x+h)-f(x-h))}/{2h}$

The alternative methods are not as efficient, so back-propogation is better than 
the alternatives.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.0001
    lr = 0.0485  #0.1
    reg = 0.0562  #0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.001,
        0.0544,
        0.0544,
        0.0001,
        0.004,
    )
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. The no-dropout version achieved higher training accuracy - which was expect as
drop out is meant to prevent overfitting. The test accuracy for 0.4 dropout was slightly better than 
the no-dropout version, which was also expected as dropout helps generalization.

We expected a bit bigger increase in accuracy for the test set. perhaps trying again with lower dropout would 
bring better results.

2. The high dropout setting has worse results in both test and train. This is probably because 
shutting down too many neuron prevents the following layers from receiving almost any useful information. 

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**
Yes. Since cross entropy takes into account the distribution of results, we may
encounter the following situation:
where the network was first giving a high confidence output only to
the wrong class, but later in the training process will give a low confidence 
output to the right class (only slightly higher than other classes).
Since this time the output is right the accuracy will increase, but since the
distribution is flatter the loss could increase too.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**
1. Gradient descent is a general optimization algorithm. 
Backpropogation is the usage of gradients descent in neural networks to update
the network weights.
2. Both methods use the loss of n training samples in order to calculate the steepest descent.
The difference is that the vanilla gradient descent iterates over all samples before making one step, while
SGD makes a small step for each training samples.
3. 
- memory constraints - can't save in memory the results from iterating over millions or more samples.
- faster convergence - making multiple small steps can lead us to a minimum much faster than taking a small amount
 of large steps.
4.
1. Mathematically this approach would be the same. An update state is made only after the loss of all
training samples was summed, which is possible since derivative is a linear operation.
This is equal to the calculation done by the GD algorithm.
2. A possible reason could be the size of the cache for previous batches in memory. Since 
each layer has to save the grad_cache of each batch this method would still not work under the memory constraints.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3
    hidden_dims = 32
    activation = 'relu'
    out_activation = 'relu'
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.01
    weight_decay = 0.01
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**

Optimization - Since our training set accuracy is pretty good and steady,our optimization error is
 low enough

Generalization - Since our test set accuracy is just as good as the training set at the 8th epoch, when
they are both at ~93% accuracy, we think are Generalization error is low if that checkpoint 
specific checkpoint is used.

Approximation - our approximation is probably high. We reached the best possible result
this model could bring us, and improving will require a more expressive network.



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Since according to the confusion matrix there are more examples predicted as 0 when 
they are actually 1, we expect a higher FNR.
This makes sense according to the data generation process where class 1 has more noise added.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**

1. on the first scenario the risk of a false negative is relatively low.
In this case we would choose a higher threshold for decision making to make sure we don't accidentally send
patients to expensive\risky unnecessary treatments.
2. In this scenario the risk of a false negative is very high, higher than a false-positive.
We would rather send a patient through unnecessary further testing than risk high probability of death
for an undiagnozed patient.  

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**

1. the best overall result was for a wide MLP with depth =1
Decision boundaries for 1 layer are relatively simple(close to x^2) at small widths.
Only at bigger widths they become more expressive. 
The deeper models have more flexible boundaries which also overfit faster.

2. The width was meaningful mostly for the shallow network. The deeper networks were already good at fitting the
data, and adding more neurons only led them to overfit.

3. The network with depth=4 and width=8 has a much more flexible boundary and is able to fit much better.
The network with depth=4 and width=32 has a too flexible boundary and is overfitting.
Since deeper neuron get information that has already been proccessed through earlier layer, we richness of
functions that we can represent is much higher than a linear combination of one hidden layer.
On the otherhand if our model overfits it is computationally more beneficial to decrease width than depth of the network.

4. Threshold selection on the validation set improved test accuracy. This is probably because the val and test set
were generated in similar ways, and thus share a close decision boundary and ratios of areas with mixed labels.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.005
    weight_decay = 0.001
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**



1.
we calculate number of parameters using the formula:
((kernel size * in channels +1) * number of convolutions)
Regular block = (3*3*256+1)*256 + (3*3*256+1)*256 = 1,180,160
BottleneckBlock = (1*1*256+1)*64 + (3*3*64+1)*64 + (1*1*64+1)*256 = 70,016
2.
Since the bottleneck uses 3x3 conv only once, the FLOPs 
should be smaller compare to the regular residual block.
3.
Since the regular block uses 3x3 conv twice, it's ability to mix spatially is better.
On the other hand the bottleneck block has more layers overall, which increases the mixing 
of different feature maps.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**
1. For L=2 our accuracy was highest, and it got lower as L got bigger.
We think large networks tend to overfit to the data and provide worse generalization. 
2. The network got unutrainable at L=16. The network was so big, that there wasn't enough data
to properly train it.
- We could increase the size of the dataset - either by getting more images or by 
introducing augmentation.
- We could pre train the network on a larger dataset, preferably from a similar domain.
After initial training we would use CIFAR10 for fine-tuning.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**
K = 64 brought the best results. Larger values of K resulted in degraded results, since the network
got overparameterized - similar to the effect of adding extra layers in 1.1. 
For K=32 the results weren't as good as k=64 probably since the network is not expressive enough
for the task.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**
the setting
Layers=2, K=256 brought the best results. 


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**
This experiment reached much better results compared to the last 
runs. 

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q5 = r"""
**Your answer:**
1. We looked for inspiration from the state of the art CNN architectures.
-We mainly adopted hyperparameters - learning rate, WD, batch size. 
-Removed dropout from convolution layers, as they seem to interfere eachother.
-Added global max pooling before attaching the MLP classifier.

-augmented data from cifar(resize, horizontal flip)
-changed the network to receive 224x224 images

2. On experiment 2 we achieved slightly better results, as the ResNet was already pretty good. 
The modifications helped us reach ~90% accuracy on the test set, as opposed to 80% for the regular ResNet.



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
