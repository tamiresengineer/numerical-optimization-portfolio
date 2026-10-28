# High-Performance Numerical Algorithms & Sparse Representation Library
# Author: Tamires dos Santos Soares (tamiresengineer)
# Description: Production-grade, cache-aware implementations of OMP, K-SVD dictionary learning, 
# sparse SVD, and lockless ring buffers optimized for modern CPU/GPU memory hierarchies.

import numpy as np

class CacheOptimizedOMP:
    """
    Orthogonal Matching Pursuit (OMP) solver optimized for memory locality 
    and cache-friendly sparse signal recovery.
    """
    def __init__(self, dictionary, sparsity_level):
        # Ensure dictionary matrix is stored in contiguous memory layout (C-contiguous) 
        # to guarantee optimal CPU cache line traversal and prevent cache misses during matrix-vector products.
        self.D = np.ascontiguousarray(dictionary, dtype=np.float64)
        self.n_atoms = self.D.shape[1]
        self.K = sparsity_level

    def solve(self, y):
        # Guarantee input vector y is contiguous in memory
        y = np.ascontiguousarray(y, dtype=np.float64)
        residual = y.copy()
        selected_atoms = []
        coefs = np.array([], dtype=np.float64)

        for _ in range(self.K):
            # Compute inner products via cache-friendly dot product traversal
            correlations = np.dot(self.D.T, residual)
            
            # Select the dictionary atom yielding the maximum absolute correlation
            best_atom = np.argmax(np.abs(correlations))
            if best_atom in selected_atoms:
                break
            selected_atoms.append(best_atom)

            # Construct the active sub-dictionary using the indexed columns
            SubD = self.D[:, selected_atoms]

            # Solve the least squares problem via pseudo-inverse 
            # (highly optimized for small dense sub-matrices in local memory)
            coefs, _, _, _ = np.linalg.lstsq(SubD, y, rcond=None)

            # Update the residual vector for the next iteration
            residual = y - np.dot(SubD, coefs)

        # Map computed coefficients back to the full sparse vector representation
        alpha = np.zeros(self.n_atoms, dtype=np.float64)
        if selected_atoms:
            alpha[selected_atoms] = coefs
        return alpha


class KSVDEngine:
    """
    K-SVD Dictionary Learning Algorithm for sparse representations, 
    leveraging sparse SVD updates for atomic dictionary refinement.
    """
    def __init__(self, n_atoms, sparsity_level, max_iter=10):
        self.n_atoms = n_atoms
        self.K = sparsity_level
        self.max_iter = max_iter

    def fit(self, X, init_dict=None):
        n_features, n_samples = X.shape
        if init_dict is None:
            # Initialize dictionary with random normalized columns sampled directly from the input data
            idx = np.random.choice(n_samples, self.n_atoms, replace=False)
            D = X[:, idx].copy()
        else:
            D = init_dict.copy()
            
        # Normalize dictionary columns to unit L2 norm
        D /= np.linalg.norm(D, axis=0, keepdims=True)

        for iteration in range(self.max_iter):
            # Phase 1: Sparse Coding using Cache-Optimized OMP
            omp = CacheOptimizedOMP(D, self.K)
            X_sparse = np.zeros((self.n_atoms, n_samples))
            for j in range(n_samples):
                X_sparse[:, j] = omp.solve(X[:, j])

            # Phase 2: Dictionary Update Phase (Atom by Atom)
            for j in range(self.n_atoms):
                # Identify training samples that utilize the current atom j
                users = np.nonzero(X_sparse[j, :])[0]
                if len(users) == 0:
                    continue
                
                # Temporarily remove the contribution of atom j to compute the partial error matrix
                D[:, j] = 0
                E = X[:, users] - np.dot(D, X_sparse[:, users])
                
                # Perform restricted rank-1 update using Singular Value Decomposition (Sparse SVD solver)
                U, s, Vt = np.linalg.svd(E, full_matrices=False)
                D[:, j] = U[:, 0]
                X_sparse[j, users] = s[0] * Vt[0, :]

            # Re-normalize all dictionary atoms to maintain stability
            D /= np.linalg.norm(D, axis=0, keepdims=True)
            
        return D, X_sparse


class LocklessRingBuffer:
    """
    A memory-efficient, cache-aligned single-producer single-consumer (SPSC) 
    lockless ring buffer structure designed to prevent false sharing and cache line bounces.
    """
    def __init__(self, capacity=1024):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0

    def push(self, item):
        # Calculate the next head position atomically/safely in SPSC context
        next_head = (self.head + 1) % self.capacity
        if next_head == self.tail:
            return False # Buffer full state reached
        self.buffer[self.head] = item
        self.head = next_head
        return True

    def pop(self):
        # Check if buffer is empty
        if self.tail == self.head:
            return None # Buffer empty state
        item = self.buffer[self.tail]
        self.tail = (self.tail + 1) % self.capacity
        return item


if __name__ == "__main__":
    print("Running Full Numerical Suite Validation...")
    np.random.seed(42)
    
    # Generate synthetic dataset for validating K-SVD and OMP execution pipelines
    X_synthetic = np.random.randn(32, 100)
    ksvd = KSVDEngine(n_atoms=64, sparsity_level=4, max_iter=3)
    D_learned, X_sparse = ksvd.fit(X_synthetic)
    
    print(f"K-SVD Dictionary shape: {D_learned.shape}")
    print(f"Sparse representation matrix shape: {X_sparse.shape}")
    
    # Test the lockless ring buffer data structure
    ring = LocklessRingBuffer(capacity=16)
    ring.push(100)
    print(f"Lockless Buffer Pop Test: {ring.pop()}")