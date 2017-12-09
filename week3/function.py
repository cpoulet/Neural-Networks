import numpy as np

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = np.zeros((dim, 1))
    b = 0
    return w, b

def loss_L2(yhat, y):
    """
    Loss function

    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    return np.sum(np.absolute(y - yhat))

def loss_L2(yhat, y):
    """
    Loss function

    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    y_diff = y - yhat
    return np.sum(np.dot(y_diff, y_diff))

def softmax(x):
    """
    Calculates the softmax for each row of the input x.

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    return x_exp / x_sum

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    norm = np.linalg.norm(x, axis = 1, keepdims = True)
    return x / norm

def img_to_vector(img):
    """
    Transform an image (typicaly a 3-layers (RGB) img with shape(length, height)) in a single vector [R,...,G,...,B,...]
    
    Arguments:
    img -- A numpy array of shape(length, height, depth)

    Returns:
    v -- a vector of shape(length*height*depth, 1)
    """
    return img.reshape(img.shape[0] * img.shape[1] * img.shape[2], 1)

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))

def main():
    x = np.array([1, 2, 3]) 
    print("x                        = " + str(x))
    print("sigmoid(x)               = " + str(sigmoid(x)))
    print("sigmoid_derivative(x)    = " + str(sigmoid_derivative(x)))
    z = np.array([0, 2]) 
    print("sigmoid(z)               = " + str(sigmoid(z)))
    w, b = initialize_with_zeros(2)
    print("b                        = " + str(b))
    print("w                        = " + str(w))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print('Error : ' + str(e))
