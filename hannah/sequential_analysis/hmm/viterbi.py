#
# Copyright (c) 2023 Hannah contributors.
#
# This file is part of hannah.
# See https://github.com/ekut-es/hannah for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import numpy as np
import itertools


# https://crawlingrobotfortress.blogspot.com/2016/07/python-recipe-for-numerically-stable.html
def viterbi(Y, logP, logA, logB):
    """
    See https://en.wikipedia.org/wiki/Viterbi_algorithm

    Parameters
    ----------
    Y : 1D array
        Observations (integer states)
    logP : array shape = (nStates ,)
        1D array of priors for initial state
        given in log probability
    logA : array (nStates,nStates)
        State transition matrix given in log probability
    logB : ndarray K x N
        conditional probability matrix
        log probabilty of each observation given each state
    """
    K = len(logP)  # Number of states
    T = len(Y)  # Number of observations
    N = np.shape(logB)[1]  # Number of states
    Y = np.int32(Y)
    assert np.shape(logA) == (K, K)
    assert np.shape(logB) == (K, N)

    # The initial guess for the first state is initialized as the
    # probability of observing the first observation given said
    # state, multiplied by the prior for that state.
    logT1 = np.zeros((K, T), "float")  # Store probability of most likely path
    logT1[:, 0] = logP + logB[:, Y[0]]

    # Store estimated most likely path
    T2 = np.zeros((K, T), "float")

    # iterate over all observations from left to right
    for i in range(1, T):
        # iterate over states 1..K (or 0..K-1 with zero-indexing)
        for s in range(K):
            # The likelihood of a new state is the likelihood of
            # transitioning from either of the previous states.
            # We incorporate a multiplication by the prior here
            log_filtered_likelihood = logT1[:, i - 1] + logA[:, s] + logB[s, Y[i]]
            best = np.argmax(log_filtered_likelihood)
            logT1[s, i] = log_filtered_likelihood[best]
            # We save which state was the most likely
            T2[s, i] = best

    # At the end, choose the most likely state, then
    # Iterate backwards over the data and fill in the state estimate
    X = np.zeros((T,), "int")  # Store our inferred hidden states
    X[-1] = np.argmax(logT1[:, -1])
    for i in range(1, T)[::-1]:
        X[i - 1] = T2[X[i], i]
    return X


# Modified viterbi from here https://crawlingrobotfortress.blogspot.com/2016/07/python-recipe-for-numerically-stable.html to work with a sliding window
def viterbi_window(Y, logP, logA, logB, size=300, class_of_interest=3):
    """
    See https://en.wikipedia.org/wiki/Viterbi_algorithm

    Parameters
    ----------
    Y : 1D array
        Observations (integer states)
    logP : array shape = (nStates ,)
        1D array of priors for initial state
        given in log probability
    logA : array (nStates,nStates)
        State transition matrix given in log probability
    logB : ndarray K x N
        conditional probability matrix
        log probabilty of each observation given each state
    size: int
        Specifying the window size
    """

    K = len(logP)  # Number of states
    T = len(Y)  # Number of observations
    N = np.shape(logB)[1]  # Number of states
    Y = np.int32(Y)
    X_final = np.zeros((T,), "int")
    assert np.shape(logA) == (K, K)
    assert np.shape(logB) == (K, N)

    # The initial guess for the first state is initialized as the
    # probability of observing the first observation given said
    # state, multiplied by the prior for that state.
    logT1 = np.zeros(
        (K, min(size + 1, T)), "int"
    )  # Store probability of most likely path
    logT1[:, 0] = logP + logB[:, Y[0]]

    # Store estimated most likely path
    T2 = np.zeros((K, min(size + 1, T)), "int")
    # Iterate over all observations from left to right
    for i in range(1, T):
        window_index = i
        if i > size:
            window_index = size
            # add new column for current observation
            logT1 = np.append(logT1, np.zeros((logT1.shape[0], 1)), 1)
            T2 = np.append(T2, np.zeros((T2.shape[0], 1)), 1)
            # remove first observation
            logT1 = logT1[:, 1:]
            T2 = T2[:, 1:]

        for s in range(K):
            log_filtered_likelihood = (
                logT1[:, window_index - 1] + logA[:, s] + logB[s, Y[i]]
            )
            best = np.argmax(log_filtered_likelihood)
            logT1[s, window_index] = log_filtered_likelihood[best]
            T2[s, window_index] = best

        # Iterate backwards
        if window_index == T - 1:
            window_index = T
        X = np.zeros((window_index,), "int")  # Store our inferred hidden states
        X[-1] = np.argmax(logT1[:, -1])
        for j in range(1, window_index)[::-1]:
            X[j - 1] = T2[X[j], j]
        if i >= size:
            X_final[i - size] = X[0]  # assign popped element as next definitive state

    # Set the end of the array equal to the window
    for x in range(size): 
        X_final[T - size + x] = X[x]
    return X_final



