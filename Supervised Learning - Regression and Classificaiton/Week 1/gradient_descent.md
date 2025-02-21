# Train the model with gradient descent

## Gradient Descent
- remember our cost function:
  - $J(w,b) = \frac{1}{2m}  * \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$
  - where $f_{w,b}(x^{(i)}) = \hat{y}^{(i)}$
- want $\argmin_{w,b}J(w,b)$
- **Outline**:
  - Start with some $w, b$
  - Kep changing $w, b$ to reduce $J(w, b)$
  - Until we settle at or near a minimum
- A squared error cost function will always yield a hammock or bowl shaped graph
- **It is possible to have an error function with multiple local minimums. We may end up in a local minima instead of the global minimum.**
- **We may also end up at a different local minima based on our starting point.**

## Implementing Gradient Descent
- Gradient Descent Algorithm
  - $w = w - \alpha * \frac{\partial}{\partial{w}}J(w, b)$
    - we are updating $w$ to move $\alpha$ proportion in the direction of $\frac{\partial}{\partial{w}}J(w, b)$
    - $\alpha$ is the learning rate. it controls how large of a step we take downhill.
  - $b = b - \alpha * \frac{\partial}{\partial{b}}J(w, b)$
  - simulataneously update $w$ and $b$
    - compute the right sides first and store in temp variables. then simulataneously update $w, b$.

## Gradient Descent Intuition
- repeat until convergence:
  - $w = w - \alpha * \frac{\partial}{\partial{w}}J(w, b)$
  - $b = b - \alpha * \frac{\partial}{\partial{b}}J(w, b)$

## Learning Rate
- $\alpha$ is the learning rate
- too small and it will take a very long time to converge. too large and we may bounce around the minima and never converge (it may diverge).
- when you are already at a minima, you will not move, b/c the partial derivative (slope) will be zero.
- There are many different ways to choose the learning rate. simply experiment with alpha and see what works. start large and have alpha decrease. make alpha proportional to the size of the partial derivative.

## Gradient Descent for Linear Regression
- Linear regression model
  - $f_{w,b}(x) = wx + b$
- Cost function: mean squared error:
  - $J(w, b) = \frac{1}{2m} \sum_{i = 1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2$
  - where $f_{w,b}(x^{(i)}) = \hat{y}^{(i)}$
- repeat until convergence:
  - $w = w - \alpha * \frac{\partial}{\partial{w}}J(w, b)$
  - $b = b - \alpha * \frac{\partial}{\partial{b}}J(w, b)$
- partial derivatives:
  - $\frac{\partial}{\partial w}J(w, b) => \frac{1}{m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)}) x^{(i)}$
  - $\frac{\partial}{\partial b}J(w, b) => \frac{1}{m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})$
- mean squared error always has one local = global minimum b/c mse is a convex function (soup bowl)

## Running Gradient Descent
- "Batch" Gradient Descent: Each step of gradient descent uses all the training examples.
  - this would be very slow for a large training set. Sometimes we will split the training set up so we don't have to wait so long for each gradient update.

## Optional Lab: Gradient Descent
```
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db
```

```
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing
```