import numpy as np
import cvxpy as cp
import LinearProgramMethod as lpm
import matplotlib.pyplot as plt
from multiprocessing import Pool

# Margin Learning
def MarginLearning(A, b, z, basic, nonb, regular_const=None):
    if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0]:
        raise ValueError("input dimensions do not coincide")
    if A.shape[1] != b.shape[1]:
        raise ValueError("input dimensions do not coincide")
    (N_samples, dim_constraints, dim_target) = A.shape
    dim_features = z.shape[1]
    if regular_const == None:
        regular_const = 1.0/np.sqrt(N_samples)
    Theta = cp.Variable((dim_target, dim_features))
    s = cp.Variable((N_samples, dim_target - dim_constraints))
    e = np.ones(dim_target - dim_constraints)
    J_N = np.zeros((N_samples, dim_target-dim_constraints, dim_target))
    J_B = np.zeros((N_samples, dim_constraints, dim_target))
    Mat = np.zeros((N_samples, dim_target-dim_constraints, dim_target))
    constraints = [s >= 0]
    objective = cp.Minimize(regular_const*0.5*cp.sum_squares(Theta) + cp.norm(cp.reshape(s, N_samples*(dim_target - dim_constraints)), 1)/N_samples)
    for i in range(N_samples):
        for k in range(dim_constraints):
            J_B[i][k][basic[i][k]] = 1
        for k in range(dim_target - dim_constraints):
            J_N[i][k][nonb[i][k]] = 1
        try:
            Mat[i] = (J_N[i].T - J_B[i].T @ np.linalg.inv(A[i][np.ix_(np.arange(dim_constraints), basic[i])]) 
                    @ A[i][np.ix_(np.arange(dim_constraints), nonb[i])]).T
            constraints = constraints + [s[i] >= e - Mat[i]@Theta@z[i]]
        except np.linalg.LinAlgError:
            print("A_B is singular.")
            # print(A[i][np.ix_(np.arange(dim_constraints), basic[i])])
    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.GUROBI)
    return Theta.value

# Kernelizing
def PolyKernel(z, benchmark_z, gamma=1.0, degree=3, coef0=1.0):
    (N_samples, dim_features) = z.shape
    (N_benchmark, dim_features) = benchmark_z.shape
    ker_z = np.zeros((N_samples, N_benchmark))
    for i in range(N_samples):
        for j in range(N_benchmark):
            ker_z[i][j] = np.power(z[i].T @ benchmark_z[j] / gamma + coef0, degree)
    return ker_z

def ExpKernel(z, benchmark_z, gamma=1.0):
    (N_samples, dim_features) = z.shape
    (N_benchmark, dim_features) = benchmark_z.shape
    ker_z = np.zeros((N_samples, N_benchmark))
    for i in range(N_samples):
        for j in range(N_benchmark):
            ker_z[i][j] = np.exp(-np.linalg.norm(z[i] - benchmark_z[j], 2) / gamma)
    return ker_z

# Kernelized Margin Learning
def KernelizedMarginLearning(A, b, z, basic, nonb, regular_const=None):
    if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0]:
        raise ValueError("input dimensions do not coincide")
    if A.shape[1] != b.shape[1]:
        raise ValueError("input dimensions do not coincide")
    (N_samples, dim_constraints, dim_target) = A.shape
    dim_features = z.shape[1]

    # Determine Regularizer
    if regular_const == None:
        regular_const = 1.0/np.sqrt(N_samples)
    # Solve Kernelized Margin Learning
    Theta = cp.Variable((dim_target, dim_features))
    s = cp.Variable((N_samples, dim_target - dim_constraints))
    e = np.ones(dim_target - dim_constraints)
    J_N = np.zeros((N_samples, dim_target-dim_constraints, dim_target))
    J_B = np.zeros((N_samples, dim_constraints, dim_target))
    Mat = np.zeros((N_samples, dim_target-dim_constraints, dim_target))
    constraints = [s >= 0]
    objective = cp.Minimize(regular_const*0.5*cp.sum_squares(Theta) + cp.norm(cp.reshape(s, N_samples*(dim_target - dim_constraints)), 1)/N_samples)
    for i in range(N_samples):
        for k in range(dim_constraints):
            J_B[i][k][basic[i][k]] = 1
        for k in range(dim_target - dim_constraints):
            J_N[i][k][nonb[i][k]] = 1
        try:
            Mat[i] = (J_N[i].T - J_B[i].T @ np.linalg.inv(A[i][np.ix_(np.arange(dim_constraints), basic[i])]) 
                    @ A[i][np.ix_(np.arange(dim_constraints), nonb[i])]).T
            constraints = constraints + [s[i] >= e - Mat[i]@Theta@z[i]]
        except np.linalg.LinAlgError:
            print("A_B is singular.")
            # print(A[i][np.ix_(np.arange(dim_constraints), basic[i])])
    prob = cp.Problem(objective, constraints)
    prob.solve(solver = cp.GUROBI)
    return Theta.value
    

# Ridge Regression
def RidgeRegression(A, b, c, z, regular_const=None):
    if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0] or A.shape[0] != c.shape[0]:
        raise ValueError("input dimensions do not coincide")
    if A.shape[1] != b.shape[1]:
        raise ValueError("input dimensions do not coincide")
    (N_samples, dim_constraints, dim_target) = A.shape
    dim_features = z.shape[1]
    if regular_const == None:
        regular_const = np.sqrt(N_samples)
    Theta = cp.Variable((dim_target, dim_features))
    hat_c = cp.Variable((N_samples, dim_target))
    constraints = []
    objective = cp.Minimize(regular_const*0.5*cp.sum_squares(Theta) + cp.sum_squares(hat_c - c))
    for i in range(N_samples):
        constraints = constraints + [Theta@z[i] == hat_c[i]]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI)
    
    return Theta.value

# OLS
def OrdinaryLeastSquares(A, b, c, z):
    if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0] or A.shape[0] != c.shape[0]:
        raise ValueError("input dimensions do not coincide")
    if A.shape[1] != b.shape[1]:
        raise ValueError("input dimensions do not coincide")
    (N_samples, dim_constraints, dim_target) = A.shape
    dim_features = z.shape[1]
    Theta = cp.Variable((dim_target, dim_features))
    hat_c = cp.Variable((N_samples, dim_target))
    constraints = []
    objective = cp.Minimize(cp.sum_squares(hat_c - c))
    for i in range(N_samples):
        constraints = constraints + [Theta@z[i] == hat_c[i]]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI)

    return Theta.value

# SPO+
def SPOplus(A, b, c, z, regular_const, step_size, batch_size, max_iter, solved=False, solution=None):
    if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0] or A.shape[0] != c.shape[0]:
        raise ValueError("input dimensions do not coincide")
    if A.shape[1] != b.shape[1]:
        raise ValueError("input dimensions do not coincide")
    (N_samples, dim_constraints, dim_target) = A.shape
    dim_features = z.shape[1]
    Theta_SPO = np.zeros((dim_target, dim_features))
    checkpoint = 100
    checkproportion = 100
    Loss = []
    solver = lpm.Solver(A, b, c)
    for t in range(int(max_iter)):
        solver_tmp = lpm.Solver(A, b, 2*z@Theta_SPO.T - c)
        if t%checkpoint == 0:
            # Compute regularized SPO+ loss
            loss_tmp = 0.5*regular_const*np.linalg.norm(Theta_SPO, 'fro')**2
            test_set = np.arange(N_samples)[::checkproportion]
            po = Pool()
            for result in po.map(solver_tmp.ComputeLP, test_set):
                i, val_get, solu_get = result
                loss_tmp -= 1/(N_samples/checkproportion) * val_get
                solution[i] = solu_get
            if solved == True:
                for i in test_set:
                    loss_tmp -= 1/(N_samples/checkproportion) * (c[i].T@solution[i])
                    loss_tmp += 2/(N_samples/checkproportion) * (Theta_SPO@z[i]).T @ solution[i]
            else:
                for result in po.map(solver.ComputeLP, test_set):
                    i, val_get, solu_get = result
                    loss_tmp -= 1/(N_samples/checkproportion) * val_get
                    loss_tmp += 2/(N_samples/checkproportion) * (Theta_SPO@z[i]).T @ solu_get

            Loss.append(loss_tmp)
        # Updata Theta
        Gradient = regular_const * Theta_SPO
        batch_set = np.random.randint(N_samples, size = (batch_size))
        for result in po.map(solver_tmp.ComputeLP, batch_set):
            i, val_get, solu_get = result
            Gradient += (2.0/batch_size) * np.reshape(-solu_get, (-1, 1))* (z[i].T)
        if solved == True:
            for i in batch_set:
                Gradient += (2.0/batch_size) * np.reshape(solution[i], (-1, 1))* (z[i].T)
        else:
            for result in po.map(solver.ComputerLP, batch_set):
                i, val_get, solu_get = result
                Gradient += (2.0/batch_size) * np.reshape(solu_get, (-1, 1))* (z[i].T)
            

        Theta_SPO -= (step_size / np.sqrt(t+1)) * Gradient
    plt.plot(Loss)
    plt.show()
    return Theta_SPO

# Naive OGD
def NaiveOnlineGradientDescent(A, b, z, solution, step_size, radius=None):
    if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0]:
        raise ValueError("input dimensions do not coincide")
    if A.shape[1] != b.shape[1]:
        raise ValueError("input dimensions do not coincide")
    (N_samples, dim_constraints, dim_target) = A.shape
    dim_features = z.shape[1]
    Theta = np.zeros((dim_target, dim_features))
    for t in range(N_samples):
        Theta -= (step_size / np.sqrt(N_samples)) * (np.array([solution[t]]).T @ np.array([z[t]]))
        if radius != None:
            if np.linalg.norm(Theta, 'fro') > radius:
                Theta *= (radius/np.linalg.norm(Theta, 'fro'))
    return Theta

# SVM-OGD
def SVM_OGD(A, b, z, basic, nonb, step_size, regular_const=0.0, radius=None):
    if A.shape[0] != z.shape[0] or A.shape[0] != b.shape[0]:
        raise ValueError("input dimensions do not coincide")
    if A.shape[1] != b.shape[1]:
        raise ValueError("input dimensions do not coincide")
    (N_samples, dim_constraints, dim_target) = A.shape
    dim_features = z.shape[1]
    Theta = np.zeros((dim_target, dim_features))
    for i in range(N_samples):
        Gradient = regular_const * Theta
        e = np.ones(dim_target - dim_constraints)
        J_N = np.zeros((dim_target-dim_constraints, dim_target))
        J_B = np.zeros((dim_constraints, dim_target))
        for k in range(dim_constraints):
            J_B[k][basic[i][k]] = 1
        for k in range(dim_target - dim_constraints):
            J_N[k][nonb[i][k]] = 1
        try:
            Mat = (J_N.T - J_B.T @ np.linalg.inv(A[i][np.ix_(np.arange(dim_constraints), basic[i])]) 
                @ A[i][np.ix_(np.arange(dim_constraints), nonb[i])]).T
            s = e - Mat @ Theta @ z[i]
            vec = np.zeros((dim_target - dim_constraints, 1))
            for j in range(dim_target - dim_constraints):
                if s[j] > 0:
                    vec[j][0] = 1
            Gradient = - Mat.T @ vec @ np.array([z[i]])
            Theta -= (step_size / np.sqrt(N_samples)) * Gradient
        except np.linalg.LinAlgError:
            print("A_B is singular.")
            # print(A[i][np.ix_(np.arange(dim_constraints), basic[i])])
        
        if radius != None:
            if np.linalg.norm(Theta, 'fro') > radius:
                Theta *= (radius/np.linalg.norm(Theta, 'fro'))
    return Theta