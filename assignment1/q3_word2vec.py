#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive,gradcheck_detail
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    norm2 = np.linalg.norm(x,2,axis = 1).reshape(x.shape[0],-1)
    x = x/norm2
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print "No error raised, test passed."
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    y_hat = softmax(np.dot(outputVectors,predicted))
    y = np.zeros(outputVectors.shape[0])
    y[target] = 1.0

    cost = -np.log(y_hat[target])
    gradPred = np.dot(outputVectors.T,y_hat - y)
    grad = np.outer(y_hat - y,predicted)
    ### END YOUR CODE

    return cost, gradPred, grad

def softmaxCostAndGradientTestWrapper(predictedandOutputVectors):
    """
    A wrapper for softmaxCostAndGradient testing by the gradient test
    """
    target = 1

    predicted = predictedandOutputVectors[:1,:].reshape([-1,])
    outputVectors = predictedandOutputVectors[1:,:]

    cost, gradPred, gradOut = softmaxCostAndGradient(predicted, target, outputVectors, None)

    return cost, np.vstack([gradPred,gradOut])

def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    #The cost now is maximize the predicted and target pair,
    #minimizing the predicted and the sampled pairs
    cost = 0
    gradPred = np.zeros(predicted.shape[0])
    grad = np.zeros(outputVectors.shape)

    for indice in indices:
        target_vector = outputVectors[indice]
        vector_product = np.dot(target_vector,predicted)
        if indice != target:
            cost += -np.log(sigmoid(-vector_product))
            gradPred += sigmoid(vector_product)*target_vector
            # use += in case of repeatance
            grad[indice,:] += sigmoid(vector_product)*predicted
        else:
            cost += -np.log(sigmoid(vector_product))
            gradPred += (sigmoid(vector_product) - 1)*target_vector
            grad[indice,:] += (sigmoid(vector_product)-1)*predicted
    ### END YOUR CODE
    #print indices

    return cost, gradPred, grad

def negSamplingCostAndGradientTestWrapper(predictedandOutputVectors, target, dataset,
                               K=10):
    predicted = predictedandOutputVectors[:1,:].reshape([-1,])
    outputVectors = predictedandOutputVectors[1:,:]

    cost, gradPred, gradOut = negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K)

    return cost, np.vstack([gradPred,gradOut])


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    currentWordIndex = tokens[currentWord]
    currentWordVector = inputVectors[currentWordIndex,:]
    costs = []
    currentWordGradIn = np.zeros(inputVectors.shape[1])
    #print currentWordIndex

    for word in contextWords:
        target = tokens[word]
        singleCost, singleGradIn, singleGradOut = word2vecCostAndGradient(currentWordVector,target,outputVectors,dataset)
        cost += singleCost
        currentWordGradIn += singleGradIn
        gradOut += singleGradOut

    gradIn[currentWordIndex,:] = currentWordGradIn
    ### END YOUR CODE

    return cost, gradIn, gradOut

def skipgramTestWrapper(currentWord, C, contextWords, tokens, paramVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    dictSize = len(tokens)
    inputVectors = paramVectors[:dictSize,:]
    outputVectors = paramVectors[dictSize:,:]
    cost, gradIn, gradOut = skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient)

    parmasGrad = np.vstack([gradIn,gradOut])

    return cost,parmasGrad

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    target = tokens[currentWord]

    contextWordVector = np.zeros(inputVectors.shape[1])
    contextWordTokens =[]
    for word in contextWords:
        contextWordTokens.append(tokens[word])
        contextWordVector += inputVectors[tokens[word],:]
    contextWordVector  = contextWordVector
    
    #use the context words to predict the target/current word
    singleCost, singleGradIn, singleGradOut = word2vecCostAndGradient(contextWordVector,target,outputVectors,dataset)
    
    ### END YOUR CODE

    cost = singleCost
    gradOut = singleGradOut

    #should be able to deal with repeated context words
    for inToken in contextWordTokens:
        gradIn[inToken,:] += singleGradIn
    
    return cost, gradIn, gradOut

def cbowTestWrapper(currentWord, C, contextWords, tokens, paramVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    dictSize = len(tokens)
    inputVectors = paramVectors[:dictSize,:]
    outputVectors = paramVectors[dictSize:,:]
    cost, gradIn, gradOut = cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient)

    parmasGrad = np.vstack([gradIn,gradOut])

    return cost,parmasGrad



#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)

def test_word2vec_partly():
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])    
    
    """
    print "gradient test for negative sampling"
    gradcheck_detail(lambda vec : negSamplingCostAndGradientTestWrapper(vec, 4, dataset,
                               K = 4), dummy_vectors, True)

    
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        softmaxCostAndGradient)
    print ""

    
    print "=============Gradient Test Result for sikp-gram and matrice============================"

    gradcheck_naive(lambda vec : skipgramTestWrapper("c", 1, ["a", "b"],
        dummy_tokens, vec, dataset,softmaxCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec : skipgramTestWrapper("b", 2, ["a", "b", "c", "c"],
        dummy_tokens, vec, dataset,softmaxCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec : skipgramTestWrapper("c", 3, ["a", "b","c","d","a","b"],
        dummy_tokens, vec, dataset,softmaxCostAndGradient), dummy_vectors)
    print ""

    print "=============Gradient Test Result for CBOW and matrice============================"
    gradcheck_naive(lambda vec : cbowTestWrapper("c", 1, ["a", "b"],
        dummy_tokens, vec, dataset,softmaxCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec : cbowTestWrapper("b", 2, ["a", "b", "c", "c"],
        dummy_tokens, vec, dataset,softmaxCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec : cbowTestWrapper("c", 3, ["a", "b","c","d","a","b"],
        dummy_tokens, vec, dataset,softmaxCostAndGradient), dummy_vectors)
    print ""

    

    print "=============Gradient Test Result for sikp-gram and negative sampling============================"
    gradcheck_naive(lambda vec : skipgramTestWrapper("c", 1, ["a", "b"],
        dummy_tokens, vec, dataset,negSamplingCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec : skipgramTestWrapper("b", 2, ["a", "b", "c", "c"],
        dummy_tokens, vec, dataset,negSamplingCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec : skipgramTestWrapper("c", 3, ["a", "b","c","d","a","b"],
        dummy_tokens, vec, dataset,negSamplingCostAndGradient), dummy_vectors)
    print ""

    print "=============Gradient Test Result for sikp-gram and negative sampling============================"
    gradcheck_naive(lambda vec : cbowTestWrapper("c", 1, ["a", "b"],
        dummy_tokens, vec, dataset,negSamplingCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec : cbowTestWrapper("b", 2, ["a", "b", "c", "c"],
        dummy_tokens, vec, dataset,negSamplingCostAndGradient), dummy_vectors)
    gradcheck_naive(lambda vec : cbowTestWrapper("c", 3, ["a", "b","c","d","a","b"],
        dummy_tokens, vec, dataset,negSamplingCostAndGradient), dummy_vectors)
    print ""
    """
    

if __name__ == "__main__":
    #U = np.array([[1.0,0],[0,1.0],[1.0,0],[0,1.0]])
    #vc = np.array([0,1.0])


    #cost,gradIn,gradOut= softmaxCostAndGradient(vc,target,U,None)
    """
    print cost
    print gradIn
    print gradOut
    
    print "test for the wrapper, print the cost and combining the result for the grads"
    cost2,grads2 = softmaxCostAndGradientTestWrapper(np.vstack([vc,U]))
    print cost2
    print grads2 
    """

    #print "gradient check the softmaxCostAndGradient"
    #random.seed(31415)
    #np.random.seed(9265)
    #dummy_vec = normalizeRows(np.random.randn(10,3))
    #gradcheck_detail(softmaxCostAndGradientTestWrapper,np.vstack([vc,U]),True)
    #gradcheck_detail(softmaxCostAndGradientTestWrapper,dummy_vec,True)
    #print ""
    test_normalize_rows()
    print "test word2vec with skip-gram and softMaxGradient"
    test_word2vec_partly()
    test_word2vec()