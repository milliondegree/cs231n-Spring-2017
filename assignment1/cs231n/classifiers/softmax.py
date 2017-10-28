import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)
    for i in range(X.shape[0]):
        scores[i] -= np.amax(scores[i])
        loss += -scores[i, y[i]] + np.log(np.sum(np.exp(scores[i])))

        dW += np.outer(X[i], np.exp(scores[i])/np.sum(np.exp(scores[i])))
        dW[:, y[i]] -= X[i]

    loss /= X.shape[0]
    dW /= X.shape[0]

    loss += reg * np.sum(W ** 2)
    dW += reg * 2 * W
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # scores = np.dot(X, W)
    # scores -= np.amax(scores, axis = 1).reshape([-1, 1])
    # sum_vec = np.log(np.sum(np.exp(scores), axis = 1))
    # lossmatrix = sum_vec - scores[:, y]
    # loss = np.sum(lossmatrix)

    # # print dW.shape
    # # print X.T.shape
    # # print (np.exp(scores)/np.sum(np.exp(scores),axis=1).reshape([-1,1])).shape

    # dW += np.dot(X.T, np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape([-1, 1]))
    # index = np.zeros_like(scores)
    # index[:, y] = 1
    # dW -= X.T.dot(index)

    # loss /= X.shape[0]
    # dW /= X.shape[0]

    # loss += reg * np.sum(W ** 2)
    # dW += reg * 2 * W
    # pass

    N, D = X.shape
    _, C = W.shape

    # print N, D, C

    scores = X.dot(W)
    scores -= np.amax(scores, axis = 1).reshape([-1, 1])
    prob = np.exp(scores) / np.sum(np.exp(scores), axis = 1).reshape([-1, 1])
    y_prob = np.zeros_like(scores)
    y_prob[range(N), y] = 1

    loss = - np.sum(np.log(prob) * y_prob) / N 
    dW = np.dot(X.T, prob - y_prob) / N

    loss += reg * np.sum(W ** 2)
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

