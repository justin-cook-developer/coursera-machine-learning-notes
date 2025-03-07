# Gradient Descent in Practice

## Feature Scaling Part 1
- example:
  - $price = w_1x_1 + w_2x_2 + b$
  - $x_1$: size (feet^2)
    - range: 300-2000
    - range is relatively large
  - $x_2$: # bedrooms
    - range: 0-5
    - range is small
- When a feature has a large range, a model is more likely to choose a small weight for that feature. Likewise, when a feature has a small range, the model is more likely to choose a larger weight for that feature.
- If we were to put size on the x-axis and # of bedrooms on the y-axis for a contour plot, the resulting plot would be tall and skinny. This causes gradient descent to bounce around a lot, and takes longer to converge.
  - This type of thing tends to happen when one feature has a small range, and another has a quite large range.
- So, how do we remedy this? **I think we should scale all of our features to be in the range [0, 1]. Andrew Ng agrees!**
  - Now every features has a comparable range for their values, so the contour plots should look more like circles and the gradient descent will run quicker.
- When features have different ranges, a learning rate may be too small for one featue and too large for another. Scaling means the learning rate works well for every feature.

## Feature Scaling Part 2
### How do we actually scale our features?
1. Divide by the maximum
   1. easiest, condenses into a smaller range.
   2. max value is 1.
   3. not centered around the origin.
2. Mean normalization:
   1. Centers the data around 0 by subtracting the mean
   2. Scales the values to [-1, 1], assuming symmetric data.
   3. When to use:
      1. When you want to shift the data to zero mean but still keep it bounded between a range.
      2. **suitable for gradient descent based algorithms (linear regression, neural nets)**
      3. **works well when data is not normally distributed**
      4. useful when working with features of different scales but need a fixed range
   4. How to:
      1. Find the average of the feature, $\mu$
      2. Find the range of the features, $range = max - min$
      3. $x_1 = \frac{x_1 - \mu}{range}$
3. Z-score normalization aka standardization:
   1. Centers the data around 0.
   2. Scales the values so that they have a unit variance ($\sigma = 1$)
   3. When to use:
      1. When the feature values vary widely and are not bounded (heights, salaries)
      2. **When using distance-based algorithms like k-means, SVMs, and PCA**
      3. **Useful when data follows a Gaussian (normal) distribution**
   4. How to:
      1. Need the standard deviation of the feature, $\sigma$
      2. Need average of the features, $\mu$
      3. $x_1 = \frac{x_1 - \mu_1}{\sigma_1}$
<br>
<br>
- General rules for feature scaling:
  - aim for about $-1 \le x_j \le 1$
    - something like $-3 \le x_j \le 3$ is acceptable
    - something like $-.3 \le x_j \le .3$ is acceptable
    - **just need small ranges that are roughly comparable across features**
  - **when in doubt, rescale!**

![mean vs z-score](./pictures/feature_scaling_ex)


## Check Gradient Descent for Convergence
- A **learning curve** shows the cost on the Y-axis, and the # of iterations of gradient descent on the X-axis.
- If gradient descent is working correctly, then gradient descent should decrease at every itration. If not, alpha may be too large or there may be a bug.
- The learning curve can also show you where we stop getting significant improvement in decreasing cost.
- **Automatic convergence test** if the cost decreases less than $\epsilon$ (ex: 0.001) from one iteration to the next, stop.
- **NG says he usually looks at curves, b/c choosing epsilon is difficult.**

## Choosing the Learning Rate
- If the cost sometimes goes up, and sometimes goes down, this means gradient descent is not working correctly. This means that alpha could be too large and we are bouncing around minimum, or there could be a bug in the code.
- For a correct implementation of gradient descent, a small enough value of alpha will cause the cost to decrease on every single iteration. So, to test code, we can set alpha to be very, very small.
- **Values of alpha to try: 0.001, 0.01, 0.1, 1, ...**
  - For each value of alpha, we might just run gradient descent for a subset of the training data, and plot the cost function. Then pick the value of alpha that seems to decrease cost quickly and reliably.
  - Or pick an upper and lower bound value for alpha, then zero in on the middle.
  - Or try to find alpha that is just too large, then decrease gradually to try and find the largest alpha that is acceptable.

## Lab: Optional Lab: Feature Scaling and Learning Rate
- **when starting analysis, plotting each feature (or what you might think are most important, when dealing with large dataset) vs output gives you an idea of which features are most important**
- when computing the gradient of the cost function, all weights share the common error part of the expression, however the common error is multiplied by $x_j^{(i)}$. If the features are not scaled to similar ranges, this can lead to one feature having much larger changes in it's weight, $w_j$.
- **When normalizing values, it is important to use the values we used to normalize a feature: max, min, $\mu, \sigma$. We will use them when predicting.**
- **a learning rate of $\alpha = 0.1$ is a good start for regression with normalized features**
- Potential Project: [Ames Housing Dataset](https://jse.amstat.org/v19n3/decock.pdf)

## Feature engineering
- let's assume we are estimating house size again. we have features: x1: width of property; x2: depth of property.
  - we note that area = width x depth. make x3 = x1 * x2.
    - now we have $f_{\vec{w},b}(\vec{x}) = w_1x_1 + w_2x_2 + w_3x_3 + b$
- **feature engineering: using intuition to design new features, by transforming or combining original features**
  - sometimes by defining new features, we can get a much better model

## Polynomial Regression
- maybe a straight line doesn't fit our data well, and a polynomial (like quadratic) would work better.
  - ex: $f_{\vec{w},b}(x) = w_1x + w_2x^2 + b$
  - in the above formula we have multiple expression involving $x$
- **an important note is that the powers in polynomials can cause some terms to be much larger than others. This necessitates feature scaling for gradient descent.**
  - I'm not exactly sure when this is done. I will have to investigate later.
  - 

## Lab: Optional Lab: Feature Engineering and Polynomial Regression
- So our linear regression model isn't producing a good fit for data that looks like $y = x^2$, what do we do?
  - Instead of $f_{\vec{w},b}(x) = w_1x + b$, we want something like:
    -  $f_{\vec{w},b}(x) = w_1x + w_2x^2 + b$
 -  Well, we can't modify the underlying model for linear regression(?), so instead lets modify the input data. We can square X (training data), before passing it to linear regression.
    -  This is us creating a new feature. Using feature engineering.
- "Above, we knew that an $x^2$ term was required. It may not always be obvious which features are required. One could add a variety of potential features to try and find the most useful. For example, what if we had instead tried: $y = w_0x_0 + w_1x_1^2 + w_2x_2^3 + b$"
  - **whichever weight ($w_j$) is assigned the biggest value by gradient descent, the corresponmding feature is probably the best fit for the model. In a sense, gradient descent is picking the righ feature for us. The longer you run gradient descent, the more clear this will be.**
```
# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
X_features = ['x','x^2','x^3']
```
- An alternative view
  - Note that we are still using linear regression once we have created new features
  - Given that, the best features will be linear to the target.
- ![alternative view](./pictures/alternative_view.png)
- Scaling Features
  - feature scaling allows gradient descent to converge much faster
  - the following code shows how we would go about scaling the features
```
x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X) 

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)
```

## Optional Lab: Linear Regression using Scikit-Learn
- gradient descent: `sklearn.linear_model.SGDRegressor`
- z-score normalization: `sklearn.preprocessing.StandardScaler`
<br>
<br>
- scale/normalize the training data
```
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
```
- Create and fit the regression model
```
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
```
- view parameters
```
b_norm = sgdr.intercept_
# note that sgdr.coef_ will be a list of weights
w_norm = sgdr.coef_
```
- make predictions
```
y_pred_sgd = sgdr.predict(X_norm)
```
