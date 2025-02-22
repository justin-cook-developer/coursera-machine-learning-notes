# Multiple Linear Regression

## Multiple Features (variables)
- It is often the case that we have more than one feature (variable) we are using to predict our outcome
- We denote seperate feature lists/vectors with a subscript: $x_1, x_2, etc$
  - these feature vectors are often column vectors in the input matrix, $A$
- we still use superscripts to denote a particular training example: $\vec{x}^{(i)}$
- a training example, $\vec{x}^{(i)}$ has the following composition: $[x_1^{(i)}, x_2^{(i)}, ..., x_n^{(i)}]$
  - often referred to as a row vector in the input matrix, $A$
  - row i of input matrix
- $n$ = number of features
<br>
<br>
- **so now the model is**:
  - $\vec{w} = [w_1, w_2, ..., w_n]$
    - this is a row vector
  - $f_{w,b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$

## Vectorization Part 1
- Parameters and features:
  - $\vec{w} = [w_1, w_1, w_3]$
  - $b$ is a scalar
  - $\vec{x} = [x_1, x_1, x_3]$
  - $=> n = 3$
  - code:
```
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10, 20, 30])
```
- code for the model, using vectorization:
  - numpy uses parallelization
    - across computers different cores & hw threads
      - or w/ virtual threads, but I hope just at hw level
    - not sure how to configure it to use the GPU
  - no unrolling or loops!
  - think back to 61C w/ the vectorization labs in c
    - could add block sizes, and unrolling
```
def f(x):
    return np.dot(w, x) + b
```

## Vectorization Part 2
- necessary to run large datasets efficiently
- Gradient descent update:
```
w = np.array([0.5, 1.3, ..., 3.4])
# d = the computed gradient updates for each weight
d = np.array([0.3, 0.2, ..., 0.4])
# compute w_j = w_j - 0.1 * d_j, for all j
w = w - 0.1 * d
```

## Gradient Descent for Multiple Linear Regression
- Parameters and features:
  - $\vec{w} = [w_1, w_1, w_3, ..., w_n]$
    - this is a row vector
  - $b$ is a scalar
  - $\vec{x} = [x_1, x_1, x_3, ..., x_n]$
- so now the model is:
  - $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$
- remember $m$ is the number of examples in the training set
- **so now the cost function is**
  - $J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2$
- **Gradient Descent**  for n features ($n \ge 2$):
  - $w_1 = w_1 - \alpha * \frac{1}{m} \sum_{i=1}^m (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}) * x_1^{(i)}$
    - where $\frac{\partial}{\partial{w_1}}J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^m (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}) * x_1^{(i)}$
  - ...
  - $w_n = w_n - \alpha * \frac{1}{m} \sum_{i=1}^m (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}) * x_n^{(i)}$
  - $b = b - \alpha * \frac{1}{m} \sum_{i=1}^m (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})$
- An alternative to gradient descent is the Normal Equation!
  - Only for Linear Regression!
  - Solve for w,b without iterations
  - Disadvantages:
    - Doesn't generalize to other models
    - Slow when number of features large (>10k)
  - Remember from EECS 16A, and recent YT videos.

## Optional Lab: Multiple Linear Regression
- this code computes the cost over the whole training set
```
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost
```
- this code computes the gradient of the cost fuction at $w, b$. Note we aren't using the above cost function. Just coputing the error in-line as needed for computing the gradient.
```
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw
```
- the code below is for gradient descent. It's actually pretty simple.
```
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing
```