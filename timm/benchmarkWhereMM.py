import torch
import time

# Matrix size (2304x768)
rows, cols = 2304, 768

# Create a random sparse matrix 'a' with few 1s, rest zeros
a = torch.zeros((rows, cols), dtype=torch.float32)
num_nonzero = int(0.01 * rows * cols)  # Sparse, about 1% non-zero elements
indices = torch.randint(0, rows * cols, (num_nonzero,))
a.view(-1)[indices] = 1

# Create a random vector 'b' of size 768
b = torch.randn(cols)

# Benchmark torch.where
start_where = time.time()

# Create the mask and apply torch.where
mask = (a == 1)
WSum_where = torch.zeros_like(a)
WSum_where = torch.where(mask, b.unsqueeze(0).expand_as(a), WSum_where)

end_where = time.time()
print(f"torch.where time: {end_where - start_where:.6f} seconds")

# Benchmark torch.matmul with correctly shaped diagonal matrix
start_matmul = time.time()

# Create diagonal matrix from `b` (this is 768x768)
V_scaled_diag = torch.diag(b)

# Perform matrix multiplication
WSum_matmul = torch.matmul(a, V_scaled_diag)

end_matmul = time.time()
print(f"torch.matmul time: {end_matmul - start_matmul:.6f} seconds")

# Benchmark torch.sparse.mm
start_sparse_mm = time.time()

# Convert `a` to a sparse matrix
a_sparse = a.to_sparse()

# Perform sparse matrix multiplication
WSum_sparse = torch.sparse.mm(a_sparse, b.unsqueeze(1)).squeeze()

end_sparse_mm = time.time()
print(f"torch.sparse.mm time: {end_sparse_mm - start_sparse_mm:.6f} seconds")
