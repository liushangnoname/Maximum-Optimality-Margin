import numpy as np
import LinearProgramMethod as lpm
from tqdm import tqdm
import time

# Generate Fractional Knapsack Samples
def GenerateFractionalKnapsack(N_samples, dim_features, dim_decision, price, Budget, Coeff_Mat, degree=1, additive_noise=1, scale_noise_uni=0, scale_noise_div=0.5, attack_threshold = None, attack_power = 0.0):
    # Coefficient_Matrix is of shape (dim_decision, dim_features)
    b = np.zeros((N_samples, dim_decision + 1))
    for i in range(N_samples):
        b[i] = np.concatenate((Budget, np.ones(dim_decision)), axis=0)
    A = np.zeros((N_samples, dim_decision+1, 2*dim_decision+1))
    for i in range(N_samples):
        tmp1 = [np.concatenate((np.concatenate((price, [1]), axis=0), np.zeros(dim_decision)), axis=0)]
        tmp2 = np.concatenate((np.concatenate((np.eye(dim_decision), np.zeros((dim_decision, 1))), axis=1), np.eye(dim_decision)), axis=1)
        A[i] = np.concatenate((tmp1, tmp2), axis=0)
    
    # Here the last dim of feature is considered to be constant 1
    z = 2*np.random.rand(N_samples, dim_features - 1) - np.ones((N_samples, dim_features - 1))
    # z = 2*np.random.binomial(n=1, p=0.5, size = (N_samples, dim_features - 1)) - np.ones((N_samples, dim_features - 1))
    z = np.concatenate((z, np.ones((N_samples, 1))), axis=1)
    base = z @ Coeff_Mat.T
    c = np.power(base, degree)
    vec = np.ones(N_samples) + scale_noise_uni * (2*np.random.rand(N_samples) - np.ones(N_samples))
    if attack_threshold != None:
        for i in range(N_samples):
            if z[i][0] > attack_threshold:
                vec[i] *= (1.0 + attack_power)
    diag = np.diag(vec)
    c = diag @ c
    mat = np.ones((N_samples, dim_decision)) + scale_noise_div * (2*np.random.rand(N_samples, dim_decision) - np.ones((N_samples, dim_decision)))
    c = np.multiply(c, mat)
    c += additive_noise*(np.random.exponential(size=(N_samples, dim_decision)) - np.ones((N_samples, dim_decision)))*0.5
    c = np.concatenate((c, np.zeros((N_samples, dim_decision+1))), axis=1)

    # maximization should be turned to minimization
    c = -c
    
    return z, c, A, b

# Generate Shortest Path Samples
def GenerateShortestPath(N_samples, dim_features, dim_edge_hori, dim_edge_vert, Coeff_Mat, degree=1, additive_noise=1, scale_noise_uni=0, scale_noise_div=0.5, attack_threshold = None, attack_power = 0.0):
    dim_cost = dim_edge_hori * (dim_edge_vert + 1) + (dim_edge_hori + 1) * dim_edge_vert

    # Here the last dim of feature is considered to be constant 1
    z = np.random.randn(N_samples, dim_features - 1)
    z = np.concatenate((z, np.ones((N_samples, 1))), axis=1)

    c = z @ Coeff_Mat.T
    c /=  np.sqrt(dim_features)
    c += 3 * np.ones((N_samples, dim_cost))
    c = np.power(c, degree)
    c += np.ones((N_samples, dim_cost))
    vec = np.ones(N_samples) + scale_noise_uni * (2*np.random.rand(N_samples) - np.ones(N_samples))
    if attack_threshold != None:
        for i in range(N_samples):
            if z[i][0] > attack_threshold:
                vec[i] *= (1.0 + attack_power)
    diag = np.diag(vec)
    c = diag @ c
    mat = np.ones((N_samples, dim_cost)) + scale_noise_div * (2*np.random.rand(N_samples, dim_cost) - np.ones((N_samples, dim_cost)))
    c = np.multiply(c, mat)
    c += additive_noise*(np.random.exponential(size=(N_samples, dim_cost)) - np.ones((N_samples, dim_cost)))*0.5
    

    A = np.zeros((N_samples, (dim_edge_vert + 1) * (dim_edge_hori + 1) - 1, dim_cost))
    A_tmp = np.zeros(((dim_edge_vert + 1) * (dim_edge_hori + 1) - 1, dim_cost))
    for i in range(dim_edge_vert + 1):
        for j in range(dim_edge_hori + 1):
            if i == dim_edge_vert and j == dim_edge_hori:
                continue
            if j >= 1:
                A_tmp[i*(dim_edge_hori + 1) + j][i*dim_edge_hori + j - 1] = 1
            if j <= dim_edge_hori - 1:
                A_tmp[i*(dim_edge_hori + 1) + j][i*dim_edge_hori + j] = -1
            if i >= 1:
                A_tmp[i*(dim_edge_hori + 1) + j][dim_edge_hori * (dim_edge_vert + 1) + j*dim_edge_vert + i - 1] = 1
            if i <= dim_edge_vert - 1:
                A_tmp[i*(dim_edge_hori + 1) + j][dim_edge_hori * (dim_edge_vert + 1) + j*dim_edge_vert + i] = -1
    for t in range(N_samples):
        A[t] = A_tmp
    
    b = np.zeros((N_samples, (dim_edge_vert + 1) * (dim_edge_hori + 1) - 1))
    b_tmp = np.zeros((dim_edge_vert + 1) * (dim_edge_hori + 1) - 1)
    b_tmp[0] = -1
    for t in range(N_samples):
        b[t] = b_tmp
    return z, c, A, b


# Generate Shortest Path Samples with Certain Margin
def GenerateFractionalKnapsack_Margin(N_samples, dim_features, dim_decision, price, Budget, Coeff_Mat, margin=0.05):
    if margin < 0:
        raise ValueError("Margin must be non-negative!")
    # Coefficient_Matrix is of shape (dim_decision, dim_features)
    b = np.zeros((N_samples, dim_decision + 1))
    for i in range(N_samples):
        b[i] = np.concatenate((Budget, np.ones(dim_decision)), axis=0)
    A = np.zeros((N_samples, dim_decision+1, 2*dim_decision+1))
    for i in range(N_samples):
        tmp1 = [np.concatenate((np.concatenate((price, [1]), axis=0), np.zeros(dim_decision)), axis=0)]
        tmp2 = np.concatenate((np.concatenate((np.eye(dim_decision), np.zeros((dim_decision, 1))), axis=1), np.eye(dim_decision)), axis=1)
        A[i] = np.concatenate((tmp1, tmp2), axis=0)
    
    # maximization should be turned to minimization
    
    # Here the last dim of feature is considered to be constant 1
    z = np.zeros((N_samples, dim_features - 1))
    z = np.concatenate((z, np.ones((N_samples, 1))), axis=1)
    c = np.zeros((N_samples, dim_decision))
    c = np.concatenate((c, np.zeros((N_samples, dim_decision+1))), axis=1)
    for i in tqdm(range(N_samples)):
        isReject = True
        # print("test")
        while isReject:
            z[i][:-1] = 2*np.random.rand(dim_features - 1) - np.ones(dim_features - 1)
            c[i][:dim_decision] = -Coeff_Mat @ z[i]
            x_tmp, _, dual_tmp = lpm.ComputeLP(A[i], b[i], c[i])
            reduced_cost_tmp = -c[i] + A[i].T @ dual_tmp
            margin_tmp = 2*margin
            for j in range(dim_decision):
                if reduced_cost_tmp[j] > 0:
                    if margin_tmp > reduced_cost_tmp[j]:
                        margin_tmp = reduced_cost_tmp[j]
            if margin_tmp >= 2*margin:
                isReject = False
    return z, c, A, b