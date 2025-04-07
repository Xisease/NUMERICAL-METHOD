import math
import numpy as np
from scipy.integrate import quad
#第一題
def f(x):
    return math.exp(x) * math.sin(4 * x)
def composite_trapezoidal(f, a, b, n):
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h
def composite_simpson(f, a, b, n):
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n):
        coef = 4 if i % 2 == 1 else 2
        result += coef * f(a + i * h)
    return result * h / 3
def composite_midpoint(f, a, b, n):
    h = (b - a) / n
    result = 0
    for i in range(n):
        midpoint = a + (i + 0.5) * h
        result += f(midpoint)
    return result * h
a = 1
b = 2
h = 0.1
n = int((b - a) / h)
approximation_trapezoidal = composite_trapezoidal(f, a, b, n)
approximation_simpson = composite_simpson(f, a, b, n)
approximation_midpoint = composite_midpoint(f, a, b, n)
print("1")
print(f"a.Composite Trapezoidal Rule approximation: {approximation_trapezoidal}")
print(f"b.Composite Simpson's Method approximation: {approximation_simpson}")
print(f"c.Composite Midpoint Rule approximation: {approximation_midpoint}")

#第二題
def f(x):
    return x**2 * np.log(x)

def gauss_legendre_quadrature(f, a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)  # 高斯節點與權重
    t = 0.5 * (x * (b - a) + (b + a))          # 區間變換後的節點
    return 0.5 * (b - a) * np.sum(w * f(t))    # 權重加總
a = 1
b = 1.5
result_3 = gauss_legendre_quadrature(f, a, b, n=3)
result_4 = gauss_legendre_quadrature(f, a, b, n=4)
result, error = quad(f, a, b)
print("2")
print(f"n = 3 answer:{result_3}")

print(f"n = 4 answer:{result_4}")

print(f"exact value:{result}")

#第三題
def f(x, y):
    return 2 * y * np.sin(x) + np.cos(x)**2

def double_integral_Bodes_rule(f, a, b, n, m):

    h_x = (b - a) / n
    result = 0


    for i in range(n + 1):
        x = a + i * h_x
        a_y = np.sin(x)
        b_y = np.cos(x)
        h_y = (b_y - a_y) / m

        inner_sum = 0
        for j in range(m + 1):
            y = a_y + j * h_y

            if j == 0 or j == m:
                coeff_y = 7
            elif j % 3 == 0:
                coeff_y = 32
            elif j % 3 == 1:
                coeff_y = 32
            else:
                coeff_y = 12

            inner_sum += coeff_y * f(x, y)
        inner_sum *= (2 * h_y / 45)
        if i == 0 or i == n:
            coeff_x = 7
        elif i % 3 == 0:
            coeff_x = 32
        elif i % 3 == 1:
            coeff_x = 32
        else:
            coeff_x = 12

        result += coeff_x * inner_sum

    result *= (2 * h_x / 45)
    return result

def gauss_legendre_2d(f, a, b, n, m):
    x_nodes, x_weights = np.polynomial.legendre.leggauss(n)
    x_mapped = 0.5 * (b - a) * x_nodes + 0.5 * (b + a)
    result = 0

    for i in range(n):
        x = x_mapped[i]
        wx = x_weights[i]
        y1, y2 = np.sin(x), np.cos(x)
        y_nodes, y_weights = np.polynomial.legendre.leggauss(m)
        y_mapped = 0.5 * (y2 - y1) * y_nodes + 0.5 * (y2 + y1)

        inner_sum = np.sum(y_weights * f(x, y_mapped))
        result += wx * (0.5 * (y2 - y1)) * inner_sum

    return 0.5 * (b - a) * result


a, b = 0, np.pi / 4
res_simpsons = double_integral_Bodes_rule(f, a, b, 4, 4)
res_gauss = gauss_legendre_2d(f, a, b, 3, 3)
res_exact, _ = quad(lambda x: quad(lambda y: f(x, y), np.sin(x), np.cos(x))[0], 0, np.pi / 4)
print("3")
print(f"a.Simpson's Rule : {res_simpsons}")
print(f"b.Gauss–Legendre Quadrature : {res_gauss}")
print(f"c.Exact Integral : {res_exact}")
#第四題
def composite_simpson(f, a, b, n):
    h = (b - a) / n
    result = f(a) + f(b)
    for i in range(1, n):
        coef = 4 if i % 2 == 1 else 2
        result += coef * f(a + i * h)
    return result * h / 3

a1, b1 = 1e-6, 1

def f1(x):
    return x**(-1/4) * np.sin(x)

# 變數變換後的 ∫ x^(-4) sin(x) dx 
def g(t):
    return t**2 * np.sin(1/t) if t > 0 else 0

# 計算積分
res_simpsons_f1 = composite_simpson(f1, a1, b1, 4)
res_simpsons_f2 = composite_simpson(g, 0, 1, 4)  #transform

# 顯示結果
print("4")
print(f"a.Composite Simpson’s Rule for ∫ x^(-1/4) sin(x) dx: {res_simpsons_f1}")
print(f"b.Composite Simpson’s Rule for ∫ x^(-4) sin(x) dx (Transformed): {res_simpsons_f2}")
