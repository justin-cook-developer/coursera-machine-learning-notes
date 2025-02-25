# Programming Assignment: Week 2 practice lab: Linear Regression
- When getting started analyzing data, it is useful to check the type, contents, and shape
```
print(type(data))
print(data[:5])
print(data.shape)
```
```
# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    ### START CODE HERE ###
    # the below code is the slow way. let's try to be faster, by using NP.
#     for i in range(m):
#         x_i = x[i]
#         y_i = y[i]
        
#         f_wb = w * x_i + b
        
#         cost = (f_wb - y_i)**2
#         total_cost += cost
    
#     total_cost = total_cost / (2 * m)

# the below code is the fast way, taking advantage of np vectorization
    pred = w * x + b
    
    error = (pred - y)
    squared_error = error**2
    
    total_cost = np.mean(squared_error) * 0.5
    
    ### END CODE HERE ### 

    return total_cost
```
- skipping straight to the parallel implementation of gradient descent, below:
```
# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
    ### START CODE HERE ###
    # don't need these vectors, b/c scalars are broadcasted to vectors
#     bs = np.full(m, b)
#     ws = np.full(m, w)
    
    pred = w * x + b
    diff = pred - y
    
    dj_dw = np.mean(diff * x)
    dj_db = np.mean(diff)
    
    ### END CODE HERE ### 
        
    return dj_dw, dj_db
```