import torch

# Check if GPUs are available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPUs are available")
    num_devices = torch.cuda.device_count()
else:
    device = torch.device('cpu')
    print("GPUs are not available, using CPU")
    num_devices = 1

# Define the size of the matrices
N = 1000  # Size of the square matrices

# Generate random matrices on each GPU
matrices_A = []
matrices_B = []
for i in range(num_devices):
    with torch.cuda.device(i):
        matrices_A.append(torch.randn(N, N).to(device))
        matrices_B.append(torch.randn(N, N).to(device))

# Perform matrix multiplication on each GPU
results = []
for i in range(num_devices):
    with torch.cuda.device(i):
        results.append(torch.matmul(matrices_A[i], matrices_B[i]))

# Check GPU usage
if torch.cuda.is_available():
    for i in range(num_devices):
        print(f"GPU {i}:")
        print(torch.cuda.memory_allocated(i))  # Print memory allocated on each GPU
        print(torch.cuda.memory_reserved(i))  # Print memory reserved on each GPU

# Print the results
for i, result in enumerate(results):
    print(f"Result from GPU {i}:")
    print(result)


# #single gpu
# import torch

# # Check if GPU is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if device.type == 'cuda':
#     print("GPU is available")
# else:
#     print("GPU is not available, using CPU")

# # Define the size of the matrices
# N = 1000  # Size of the square matrices

# # Generate random matrices
# if device.type == 'cuda':
#     A = torch.randn(N, N).cuda()  # Random matrix A on GPU
#     B = torch.randn(N, N).cuda()  # Random matrix B on GPU
# else:
#     A = torch.randn(N, N)  # Random matrix A on CPU
#     B = torch.randn(N, N)  # Random matrix B on CPU

# # Perform matrix multiplication
# C = torch.matmul(A, B)

# # Check GPU usage
# if device.type == 'cuda':
#     print(torch.cuda.memory_allocated(device))  # Print memory allocated on GPU
#     print(torch.cuda.memory_reserved(device))  # Print memory reserved on GPU

# # Print the result
# print(C)
