#!/usr/bin/env python

import numpy as np
import random


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
    while not it.finished:
        ix = it.multi_index
				
        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
        x1 = x.copy()
        x2 = x.copy()
        x1[ix] += h 
        x2[ix] -= h
        random.setstate(rndstate)
        fx1 = f(x1)[0]
        random.setstate(rndstate)
        fx2 = f(x2)[0]
        numgrad = (fx1 - fx2)/2/h
        ### END YOUR CODE
        
			
        # Compare gradients
        try:
            reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        except ValueError, e:
            raise ValueError("Might be function output shape error, the output shape of the function is %s" % str(numgrad.shape))

        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad)
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"

# A more complex gradient check function
def gradcheck_detail(f, x, showModifiedMatrix = False):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
    while not it.finished:
        ix = it.multi_index
                
        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        ### YOUR CODE HERE:
        x1 = x.copy()
        x2 = x.copy()
        x1[ix] += h 
        x2[ix] -= h
        random.setstate(rndstate)
        fx1 = f(x1)[0]
        random.setstate(rndstate)
        fx2 = f(x2)[0]
        numgrad = (fx1 - fx2)/2/h
        ### END YOUR CODE
        
            
        # Compare gradients
        try:
            reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        except ValueError, e:
            raise ValueError("Might be function output shape error, the output shape of the function is %s" % str(numgrad.shape))

        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad)
            
            print "Original function result is %f" % fx
            print "adding number h is %f" % h 
            print "result for adding h  is %f, for minus h is %f" %(fx1,fx2)
            
            if showModifiedMatrix:
                print "matrix for adding h is %s" %str(x1)
                print "matrix for minus h is %s" %str(x2) 
            return

        it.iternext() # Step to next dimension

    print "Gradient check passed!"
    print ""

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ""


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    from q2_neural import forward_backward_prop
    """
    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    
    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)
    ### END YOUR CODE
    """
    tri = lambda x: (np.sum(x ** 3), 3* (x ** 2))
    exp = lambda x: (np.sum(np.exp(x)), np.exp(x))

    print "Running sanity checks..."
    gradcheck_naive(tri, np.array(123.456))      # scalar test
    gradcheck_naive(tri, np.random.randn(3,))    # 1-D test
    gradcheck_naive(tri, np.random.randn(4,5))   # 2-D test
    print ""
    gradcheck_naive(exp, np.array(123.456))      # scalar test
    gradcheck_naive(exp, np.random.randn(3,))    # 1-D test
    gradcheck_naive(exp, np.random.randn(4,5))   # 2-D test
    print ""



if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
