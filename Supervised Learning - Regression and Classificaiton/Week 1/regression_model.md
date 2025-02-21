# Regression Model

## Linear Reression Model Part 1
- Remeber that in supervised learning, the data is trained with the right answers
- Regression models predicts numbers. There are infinitely many possible outputs.
- Linear regression is one example of a regression model. It fits a straight line, the line of best fit, to a data set.
- Terminology
    - Data set used to train the model is the **training set**
    - *x* denotes the "inpute" variable. also called the feature.
    - *y* denotes the "output" variable. also called the target variable. the true value for that variable.
    - *m* = number of training examples.
    - $(x, y)$ denotes a single training example.
    - $(x^{(i)}, y^{(i)})$ denotes a specific training example. the ith training example.

## Linear Regression Model Part 2
- our learning algorithm takes the training set, and produces a function, *f*, sometimes called the hypothesis. *f* takes *x* and predicts $\hat{y}$ (an estimate for *y*).
- $f_{w,b}(\vec{x}) = \vec{w} \cdot \vec{x} + \vec{b}$
  - sometimes just use $f(\vec{x}) = \vec{w} \cdot \vec{x}+ \vec{b}$
- a lineare regression model with one input feature is called a **univariate linear regression model**
- sometimes, a scalar will be denoted with a lowercase letter, ex: a, and a vector will be denoted in bold, ex: **a** 

## Lab: Model Representation
- ```np.array([])``` creates a numpy array
- ```np.zeros(len)``` creates an array of zeros
- $f_{w,b}(\vec{x}) = \vec{w} \cdot \vec{x} + \vec{b}$ in code is:
  - ```np.dot(x, w) + b```
- you can also just use the $*$ operator on two np arrays

## Cost Function Formula
- $w, b$ called the parameters of the model. also referred to as coefficients or weights.
  - naturally different $w, b$ product different $f$
- With linear regression, we want to choose the $w, b$ that minimizes the cost function.
- The error for a single training example is:
  - $(\hat{y}^{(i)} - y^{(i)})^2$
- The error for the whole training set is: the average error:
  - $\frac{1}{2m}  * \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$
  - in practice, we use $\frac{1}{2m}$ instead of $\frac{1}{m}$. The division by 2 makes some of the calculations easier.
- The cost function can be rewritten as:
  - $J(w,b) = \frac{1}{2m}  * \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$
- You could have another cost function, but this one is quite standard.

## Cost Function Intuition
- the cost function we defined above is the squared error cost function
- models parameters are: w, b
- the cost function measures how well the parameters fit the training data.
  - we find the best parameters by $\argmax_{w,b}{J(w,b)}$
- each value of $w$ corresponds to a different straight line fit, and $b$ creates a shift.

## Visualizing the cost function
- we use gradient descent to minimize the cost function. The cost function will look something like a soup bowl or 3d parabola. Any particulare point on the surface represents a particular choice of $w, b$
- we can also use a contour plot to show the cost function $J(w, b)$. This is a convenient way to visualize the cost function in a 2d plot.

## Visualization Examples
- points closer to the center of the contour plot will have lower values of $J(w, b)$, assuming the shape of $J(w, b)$ is a soup bowl.
- in a contour plot, we take horizontal slices of the graph of $J(w, b)$. Therefore, points on the same ellipse on a contour plot can come from different choices of $w, b$; **it is important to note that every point on an ellipse has the same value of $J(w, b)$ despite $w, b$ being different.**
- when ellipses are closer together on a contour plot, that corresponds to a higher slope from ring to ring. We need a small change in w or b to make the value of J change by one.
- travelling off the contour orthogonally will give us the greatest change in height for the shortest vector (going in the steepest direction). This is the idea behind gradient descent.
- the fact that the cost function squares $\hat{y} - y$ ensures that surface of the plot of the cost function is convex
- A function, $f(x)$, is convex if its second derivative is non-negative for all $x$ in its domain.
  - This ensures that the function's slope is **non-decreasing**, meaning it curves upward or remains straight.
  - Intuition: squaring a function magnifies positive values and smooths out variations