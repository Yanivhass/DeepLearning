r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1) False. The test set allows us to estimate only our out-sample error (the samples present in the test set), although there is a certain correlation (a low test set error is probably indicative of a low training set error, but only roughly.
2)False. For instance we want to indentify cats and dogs If we split so that in test there will be only dogs and in train only cats. Than the machine will not learn anything about dogs, and think that everything is cat. So in that case there is an split that will not produce the same effect as another.
3)True. For cross-validation we only use validation-set.
4)True. Each fold represent a different split and multiple splits approximate the generalization error. 

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**
1)My friend's approach is not justified. Using the test set to improve our model takes away the ability
to test the model's generalization. Tuning hyperparameters should only be done on the validation set.
Regularization helps with overfit which might help. Since there is no closed solution for the optimal lambda the only way to find it is by experiment,
but these experiments should never be conducted on the test set.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
Increasing k will improve the generalization only to a certain point.
For a very small k the prediction will be made by the few closest examples, even if they are outliers.
For a very large k the prediction will be made as a majority vote between a lot of samples, most of them irrelevant.
In extreme cases, the prediction will be based solely on the ration between quantities of samples from each class.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**
1.The best model with respect to the training set will not necessarily be the model with the lowest generalization error.
One model might have a low training error and high validation error, while another might be average on both which is better.
2.Once you have used evaluated a model on the test set, making further improvements based on that score means you
no longer have a "neutral" data set to use for testing. This will make it very difficult the evaluate the models generalization.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
Delta is arbitrary since for every different value of Delta the matrix W will be scaled
accordingly to reach the same minima and same loss values.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**
1. The model had learnt the pixels that are on average activated for a given class. 
Errors can mostly happen for classes with similar activations (like 5 and 6).
2. This interpretation is similar in how the knn looks for the highest correlation of
activated pixels.
On the other hand knn performs a comparison per sample in the training set
instead of a fixed number of W matrices. 


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**

Based on the graph we think our learning rate is good.

For a high learning rate we will get a fast convergence but usually to a local minima which is not
the best possible score. For extremely high learning rates we will see a "wiggling" pattern around
that values since the optimizer will constantly miss the minima.
For a low learning rate we will get a slow convergence but usually to a better score. It is
not recommended to conduct the entire training with low learning rate since it might lead to learning 
of mostly low level features(as opposed to the major general patterns we seek).


"When the learning rate is too small, training is not only slower,
but may become permanently stuck with a high training error."
— Page 429, Deep Learning, 2016

Based on the accuracy our model is slightly overfitted to the training set since the training
accuracy is a bit higher than the validation accuracy.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
The ideal pattern for a residual plot would be a single straight line at 0
with zero variance around it. Our trained models seems to be a good fit, but 
there is still room for improvement.
Comparing the original top-5 plot with the CV plot, the CV plot is 
significantly better-performing.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**
1.The model is still linear as it is a linear functions of the learned parameters.
The model is always a straight line(or hyperplane) only the axis(features) are different.
2. We can fit any non linear function as long as we engineer the features in advance to 
conform to the relationship. Trying to regress to an unknown function that does not appear
as a transform over the features is impossible.
3. As pointed earlier, the decision boundary will always be a hyperplane. Adding non
linear features only allows for new dimensions that might make the classes more easily
separable.
Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**
1. The exact size of lambda is more important at smaller scales. 
e.g. a e-1 difference of regularization is much more dramatic when lambda=0.1 than
when lambda=10. To have much denser sampling at each appropriate scale np.logspace 
would be much better.

2. The model was fit once for every searched combination of hyper param.
in total 3*20 = 60 times

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
