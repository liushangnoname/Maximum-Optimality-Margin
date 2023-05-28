import numpy as np
import cvxpy as cp
import LinearProgramMethod as lpm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import LearningMethod as lm
import LinearProgramMethod as lpm

# # Modify Samples to Be Strictly Separable
# def ModifyMargin(A, b, c, z, basic, nonb, margin = 1.0):
#     if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0]:
#         raise ValueError("input dimensions do not coincide")
#     if A.shape[1] != b.shape[1]:
#         raise ValueError("input dimensions do not coincide")
#     (N_samples, dim_constraints, dim_target) = A.shape
#     dim_features = z.shape[1]

#     # y = np.zeros((N_samples, dim_target))
#     # mod_z = np.zeros((N_samples, dim_features))
#     # for t in range(N_samples):
#     #     for k in range(dim_constraints):
#     #         y[t][basic[t][k]] = -1
#     #     for k in range(dim_target - dim_constraints):
#     #         y[t][nonb[t][k]] = 1
        
#     #     mod_z[t] = z[t] / np.linalg.norm(z[t], 2)
#     mod_c = np.copy(c)
#     e = np.ones(dim_target - dim_constraints)
#     J_N = np.zeros((N_samples, dim_target-dim_constraints, dim_target))
#     J_B = np.zeros((N_samples, dim_constraints, dim_target))
#     Mat = np.zeros((N_samples, dim_target-dim_constraints, dim_target))
#     for i in range(N_samples):
#         for k in range(dim_constraints):
#             J_B[i][k][basic[i][k]] = 1
#         for k in range(dim_target - dim_constraints):
#             J_N[i][k][nonb[i][k]] = 1
#         try:
#             Mat[i] = (J_N[i].T - J_B[i].T @ np.linalg.inv(A[i][np.ix_(np.arange(dim_constraints), basic[i])]) 
#                     @ A[i][np.ix_(np.arange(dim_constraints), nonb[i])]).T
#             mod = np.maximum(2*margin*e - Mat[i]@c[i], np.zeros(dim_target - dim_constraints))
#             for k in range(dim_target - dim_constraints):
#                 mod_c[i][nonb[i][k]] += mod[k]
#         except np.linalg.LinAlgError:
#             print("A_B is singular.")
#             # print(A[i][np.ix_(np.arange(dim_constraints), basic[i])])
#     return mod_c


# Online Margin Learning
def OnlineMarginLearning(A, b, c, z, basic, nonb, regular_const=None, solved=False, solution=None, show_figure=False, checkpoint=10):
    if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0]:
        raise ValueError("input dimensions do not coincide")
    if A.shape[1] != b.shape[1]:
        raise ValueError("input dimensions do not coincide")
    (N_samples, dim_constraints, dim_target) = A.shape
    dim_features = z.shape[1]

    record = np.zeros(N_samples)
    wrong_count = 0
    cumulative_regret = 0.0
    Theta = np.zeros((dim_target, dim_features))
    for t in tqdm(range(N_samples)):
        # Compute Corresponding Theta
        if (t < 10*checkpoint or t % checkpoint == 0) and t > 0:
            Theta = lm.MarginLearning(A = A[:t, :, :], b = b[:t, :], z = z[:t, :], basic = basic[:t, :], nonb = nonb[:t, :], regular_const = regular_const / np.sqrt(t+1))
        hat_c_tmp = Theta @ z[t]
        est_tmp, _, _ = lpm.ComputeLP(A = A[t], b = b[t], c = hat_c_tmp)
        if solved == False:
            solution_tmp, _, _ = lpm.ComputeLP(A = A[t], b = b[t], c = c[t])
        else:
            solution_tmp = solution[t]
        cumulative_regret += c[t].T @ (est_tmp - solution_tmp)
        if np.amax(np.absolute(est_tmp - solution_tmp)) > 1e-5:
            wrong_count += 1
        record[t] = cumulative_regret
    if show_figure == True:
        plt.plot(record)
        plt.show()
        print("Total wrong decisions", wrong_count)
    return record

# Perceptron Algorithm
def Perceptron(A, b, c, z, basic, nonb, solved=False, solution=None, show_figure=False, margin=0.05, isDebug = False):
    if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0]:
        raise ValueError("input dimensions do not coincide")
    if A.shape[1] != b.shape[1]:
        raise ValueError("input dimensions do not coincide")
    (N_samples, dim_constraints, dim_target) = A.shape
    dim_features = z.shape[1]
    
    record = np.zeros(N_samples)
    eigs = []
    wrong_count = 0
    reduced_cost = np.zeros((N_samples, dim_target))
    predict_cost = np.zeros((N_samples, dim_target))
    cumulative_regret = 0.0
    Theta = np.zeros((dim_target, dim_features))
    for t in tqdm(range(N_samples)):
        # Use Predicted c to Compute LP Solutions
        predict_cost[t] = Theta @ z[t]
        est_tmp, _, dual_tmp = lpm.ComputeLP(A = A[t], b = b[t], c = predict_cost[t])
        if solved == False:
            solution_tmp, _, _ = lpm.ComputeLP(A = A[t], b = b[t], c = c[t])
        else:
            solution_tmp = solution[t]
        cumulative_regret += c[t].T @ (est_tmp - solution_tmp)
        record[t] = cumulative_regret
        if np.amax(np.absolute(est_tmp - solution_tmp)) > 1e-5:
            wrong_count += 1
            if isDebug:
                print("Wrong prediction at step", t)
                print("True c\n", c[t])
                print("Estimated c\n", predict_cost[t])
                print("True solution\n", solution_tmp)
                print("Estimated solution\n", est_tmp)
        
        isRev = True
        max_eig = 0.0
        try:
            inv_A_B = np.linalg.inv(A[t][np.ix_(np.arange(dim_constraints), basic[t])])
            eig = np.amax(np.linalg.eigvals(inv_A_B))
            if eig > max_eig:
                max_eig = eig
            eigs.append(max_eig)
        except np.linalg.LinAlgError:
            print("A_B is singular")
            isRev = False
        if isRev:
            reduced_cost[t] = Theta @ z[t] - A[t].T @ inv_A_B.T @ (Theta @ z[t])[np.ix_(basic[t])]
            if isDebug:
                print(reduced_cost[t])
            
        if isRev:
            pred_nonb = -1.0 * np.ones(dim_target)
            for i in range(dim_target):
                if reduced_cost[t][i] >= margin:
                    pred_nonb[i] = 1.0
            true_nonb = -1.0 * np.ones(dim_target)
            for j in range(dim_target - dim_constraints):
                true_nonb[nonb[t][j]] = 1.0
            for i in range(dim_target):
                if pred_nonb[i] != true_nonb[i]:
                    Theta[i] += true_nonb[i] * z[t]
                    Theta[np.ix_(basic[t], np.arange(dim_features))] -= true_nonb[i] * inv_A_B @ (np.array([A[t, :, i]]).T @ np.array([z[t]]))
        
    # return reduced_cost

    if show_figure == True:
        plt.plot(record)
        plt.show()
        print("Total wrong decisions", wrong_count)
        # plt.plot(eigs)

    return record, predict_cost, reduced_cost

# OGD SVM
def OGD_MarginLearning(A, b, c, z, basic, nonb, regular_const=0.0, solved=False, solution=None, show_figure=False, step_size=1e-1, radius=None):
    if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0]:
        raise ValueError("input dimensions do not coincide")
    if A.shape[1] != b.shape[1]:
        raise ValueError("input dimensions do not coincide")
    (N_samples, dim_constraints, dim_target) = A.shape
    dim_features = z.shape[1]

    record = np.zeros(N_samples)
    wrong_count = 0
    cumulative_regret = 0.0
    Theta = np.zeros((dim_target, dim_features))
    for t in tqdm(range(N_samples)):
        Gradient = regular_const * Theta
        e = np.ones(dim_target - dim_constraints)
        J_N = np.zeros((dim_target-dim_constraints, dim_target))
        J_B = np.zeros((dim_constraints, dim_target))
        for k in range(dim_constraints):
            J_B[k][basic[t][k]] = 1
        for k in range(dim_target - dim_constraints):
            J_N[k][nonb[t][k]] = 1
        try:
            Mat = (J_N.T - J_B.T @ np.linalg.inv(A[t][np.ix_(np.arange(dim_constraints), basic[t])]) 
                @ A[t][np.ix_(np.arange(dim_constraints), nonb[t])]).T
            s = e - Mat @ Theta @ z[t]
            vec = np.zeros((dim_target - dim_constraints, 1))
            for j in range(dim_target - dim_constraints):
                if s[j] > 0:
                    vec[j][0] = 1
            Gradient = - Mat.T @ vec @ np.array([z[t]])
            Theta -= (step_size / np.sqrt(N_samples)) * Gradient
        except np.linalg.LinAlgError:
            print("A_B is singular.")
            # print(A[i][np.ix_(np.arange(dim_constraints), basic[i])])
        
        if radius != None:
            if np.linalg.norm(Theta, 'fro') > radius:
                Theta *= (radius/np.linalg.norm(Theta, 'fro'))
        
        hat_c_tmp = Theta @ z[t]
        est_tmp, _, _ = lpm.ComputeLP(A = A[t], b = b[t], c = hat_c_tmp)
        if solved == False:
            solution_tmp, _, _ = lpm.ComputeLP(A = A[t], b = b[t], c = c[t])
        else:
            solution_tmp = solution[t]
        cumulative_regret += c[t].T @ (est_tmp - solution_tmp)
        if np.amax(np.absolute(est_tmp - solution_tmp)) > 1e-5:
            wrong_count += 1
        record[t] = cumulative_regret
    if show_figure == True:
        plt.plot(record)
        plt.show()
        print("Total wrong decisions", wrong_count)
    return record