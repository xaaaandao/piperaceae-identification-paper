#cython: language_level=3
#distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import collections

import numpy as np
cimport cython
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def split_dataset(index: np.ndarray, n_features: int, patch: int, np.ndarray[np.float64_t, ndim=2] x,
                  np.ndarray[np.int16_t, ndim=1] y):
    cdef np.ndarray[np.float64_t, ndim=2] new_x = np.empty((len(index) * patch, n_features), dtype=np.float64)
    cdef np.ndarray[np.int16_t, ndim=1] new_y = np.empty((len(index) * patch, ), dtype=np.int16)
    cdef int k = 0

    for ind in index:
        start = (ind * patch)
        end = start + patch
        for i in range(start, end):
            for j in range(0, n_features):
                new_x[k][j] = x[i][j]
            new_y[k] = y[i]
            k+=1

    return new_x, new_y


@cython.boundscheck(False)
@cython.wraparound(False)
def sum_rule(n_test: int, n_labels:int, patch: int, np.ndarray[np.float64_t, ndim=2] y_pred_proba):
    cdef np.ndarray[np.float64_t, ndim = 1] sum
    cdef np.ndarray[np.int16_t, ndim = 1] y_pred = np.zeros((int(n_test/patch),), dtype=np.int16)
    cdef np.ndarray[np.float64_t, ndim = 2] y_score = np.zeros((int(n_test/patch), n_labels), dtype=np.float64)

    for k, y in enumerate(range(0, n_test, patch)):
        sum = np.zeros((n_labels,))
        for i in range(y, y+patch):
            for j in range(0, n_labels):
                sum[j]=sum[j]+y_pred_proba[i][j]
        y_score[k] = sum
        y_pred[k] = np.argmax(sum)+1

    return np.sort(y_pred, kind='mergesort'), y_score


@cython.boundscheck(False)
@cython.wraparound(False)
def mult_rule(n_test: int, n_labels:int, patch: int, np.ndarray[np.float64_t, ndim=2] y_pred_proba):
    cdef np.ndarray[np.float64_t, ndim = 1] mult
    cdef np.ndarray[np.int16_t, ndim = 1] y_pred = np.zeros((int(n_test/patch),), dtype=np.int16)
    cdef np.ndarray[np.float64_t, ndim = 2] y_score = np.zeros((int(n_test/patch), n_labels), dtype=np.float64)

    for k, y in enumerate(range(0, n_test, patch)):
        mult = np.ones((n_labels,))
        for i in range(y, y+patch):
            for j in range(0, n_labels):
                mult[j]=mult[j]*y_pred_proba[i][j]
        y_score[k] = mult
        y_pred[k] = np.argmax(mult)+1

    return np.sort(y_pred, kind='mergesort'), y_score


@cython.boundscheck(False)
@cython.wraparound(False)
def y_true_no_patch(n_test: int, patch: int, np.ndarray[np.int16_t, ndim=1] y_true):
    cdef np.ndarray[np.int16_t, ndim = 1] y_true_no_patch = \
        np.empty((int(n_test/patch),), dtype=np.int16)

    s = 0
    for k, v in collections.Counter(y_true).items():
        for i in range(0, int(v/patch)):
            y_true_no_patch[s] = k
            s+=1

    return np.sort(y_true_no_patch, kind='mergesort')

@cython.boundscheck(False)
@cython.wraparound(False)
def load(n_features:int, n_samples: int, np.ndarray[np.float64_t, ndim=2] x1, np.ndarray[np.float64_t, ndim=2] x2, start=0):
    for i in range(start, start+n_samples):
        for j in range(n_features):
            x1[i][j] = x2[i][j]

    return x1, i