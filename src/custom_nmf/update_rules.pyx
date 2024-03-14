# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
cimport cython
from scipy.sparse.csr cimport csr_matrix
from libc.stdlib cimport malloc, free

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void scale_row(np.ndarray row, DTYPE_t[:] col_sums):
    cdef Py_ssize_t i
    for i in range(row.shape[0]):
        row[i] /= col_sums[i]

@cython.boundscheck(False)
@cython.wraparound(False)
def update_H(np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=2] H, csr_matrix X_csr):
    cdef:
        DTYPE_t[:] X_data = X_csr.data
        int[:] X_indices = X_csr.indices
        int[:] X_indptr = X_csr.indptr
        np.ndarray[DTYPE_t, ndim=1] H_col_sums = np.sum(H, axis=1)
        Py_ssize_t i, j, jj
        DTYPE_t val
    
    for i in range(X_csr.shape[0]):
        for jj in range(X_indptr[i], X_indptr[i + 1]):
            j = X_indices[jj]
            val = X_data[jj]
            # Update H based on the sparse structure of X
            # This is a placeholder for actual sparse update logic
            # You'll need to adapt this according to your algorithm's needs

    # Example: Scale H by column sums of H for demonstration purposes
    for j in range(H.shape[1]):
        scale_row(H[:, j], H_col_sums)

def update_W(np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=2] H, csr_matrix X_csr):
    # Similar structure to update_H, adapted for updating W.
    # Implement according to your algorithm's requirements
    pass
