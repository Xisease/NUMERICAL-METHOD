import numpy as np

A = np.array([
    [4, -1, 0, -1, 0, 0],
    [-1, 4, -1, 0, -1, 0],
    [0, -1, 4, 0, 1, -1],
    [-1, 0, 0, 4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)
x0 = np.zeros_like(b, dtype=float)

epsilon = 1e-6
max_iter = 1000
# (a) Jacobi Method
def jacobi(A, b, x0, max_iter, tol):
    n = len(b)
    x = x0.copy()
    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1
        x = x_new
    return x,max_iter

# (b) Gauss-Seidel Method
def gauss_seidel(A, b, x0, max_iter, epsilon):
    x = x0.copy()
    n = len(b)
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x_new[j] if j < i else A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
            return x_new, k + 1
        x = x_new
    return x, max_iter

# (c) SOR Method (Successive Over-Relaxation)
def sor(A, b, x0, omega, max_iter, epsilon):
    x = x0.copy()
    n = len(b)
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x_new[j] if j < i else A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sigma) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
            return x_new, k + 1
        x = x_new
    return x, max_iter

# (d) Conjugate Gradient Method
def conjugate_gradient(A, b, x0, max_iter, epsilon):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)
    for i in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < epsilon:
            return x, i + 1
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, max_iter

# 執行各方法
x_jacobi, it_jacobi = jacobi(A, b, x0, max_iter, epsilon)
x_gs, it_gs = gauss_seidel(A, b, x0, max_iter, epsilon)
x_sor, it_sor = sor(A, b, x0, omega=1.1, max_iter=max_iter, epsilon=epsilon)
x_cg, it_cg = conjugate_gradient(A, b, x0, max_iter, epsilon)

# 顯示結果
print("(a)Jacobi:", x_jacobi)
print("(b)Gauss-Seidel:", x_gs)
print("(c)SOR (ω=1.1):", x_sor)
print("(d)Conjugate Gradient:", x_cg)