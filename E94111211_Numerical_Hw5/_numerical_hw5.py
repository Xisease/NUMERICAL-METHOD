import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd 
print("-----第一題-----")
#第一題
def f(t, y):
    return 1 + (y / t) + (y / t) ** 2
#df_dt
def df_dt(t, y):
    #y和t的偏微分
    df_dt_partial = -y / t**2 - 2 * y**2 / t**3
    df_dy_partial = 1 / t + 2 * y / t**2
    f_val = f(t, y)
    total_derivative = df_dt_partial + df_dy_partial * f_val
    return total_derivative
def exact(t):
    return t * np.tan(np.log(t))
# 設定初始條件
t0 = 1.0
y0 = 0.0
h = 0.1
t_end = 2.0
steps = int((t_end - t0) / h)
# Euler method
def euler(t0, y0, h, steps):
    t_values = [t0]
    y_values = [y0]
    for _ in range(steps):
        y0 = y0 + h * f(t0, y0)
        t0 = t0 + h
        t_values.append(t0)
        y_values.append(y0)
    return np.array(t_values), np.array(y_values)
# Taylor method of order 2
def taylor2(t0, y0, h, steps):
    t_values = [t0]
    y_values = [y0]
    for _ in range(steps):
        f0 = f(t0, y0)
        df0 = df_dt(t0, y0)
        y0 = y0 + h * f0 + (h**2 / 2) * df0
        t0 = t0 + h
        t_values.append(t0)
        y_values.append(y0)
    return np.array(t_values), np.array(y_values)

# 計算數值解
t_euler, y_euler = euler(t0, y0, h, steps)
t_taylor2, y_taylor2 = taylor2(t0, y0, h, steps)
t_exact = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
y_exact = exact(t_exact)
table = pd.DataFrame({
    't': t_exact,
    'Exact y': y_exact,
    'Euler y': y_euler,
    'Taylor2 y': y_taylor2,
})
print(table)


# 畫圖比較
plt.plot(t_exact, y_exact, 'bs--', label='Exact Solution')  
plt.plot(t_euler, y_euler, 'r--', label="Euler's Method") 
plt.plot(t_taylor2, y_taylor2, 'k-', label="Taylor's Method Order 2")
plt.xlabel('t')
plt.ylabel('y')
plt.title('Comparison of Numerical Methods')
plt.legend()
plt.grid(True)
plt.show()
    
print("")
# 第二題

print("-----第二題-----")
# 定義微分方程
def F(t, u):
    u1, u2 = u
    du1_dt = 9*u1 + 24*u2 + 5*math.cos(t) - (1/3)*math.sin(t)
    du2_dt = -24*u1 - 52*u2 - 9*math.cos(t) + (1/3)*math.sin(t)
    return np.array([du1_dt, du2_dt])

# exact
def exact_u1(t):
    return 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)

def exact_u2(t):
    return -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)

# Runge-Kutta method
def runge_kutta_4(t0, u0, h, steps):
    t_values = [t0]
    u_values = [u0]
    for _ in range(steps):
        k1 = F(t0, u0)
        k2 = F(t0 + h/2, u0 + h/2 * k1)
        k3 = F(t0 + h/2, u0 + h/2 * k2)
        k4 = F(t0 + h, u0 + h * k3)
        u0 = u0 + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t0 = t0 + h
        t_values.append(t0)
        u_values.append(u0)
    return np.array(t_values), np.array(u_values)

# 初始條件
t0 = 0.0
u0 = np.array([4/3, 2/3])
t_end = 1.0

# h = 0.05
h1 = 0.05
steps1 = int((t_end - t0) / h1)
t_rk1, u_rk1 = runge_kutta_4(t0, u0, h1, steps1)

# h = 0.1
h2 = 0.1
steps2 = int((t_end - t0) / h2)
t_rk2, u_rk2 = runge_kutta_4(t0, u0, h2, steps2)

# 精確解
t_exact = np.linspace(t0, t_end, 10)
u1_exact_values = exact_u1(t_exact)
u2_exact_values = exact_u2(t_exact)

def generate_error_table(t_rk, u_rk, h):
    data = []
    for ti, (ui1, ui2) in zip(t_rk, u_rk):
        exact1 = exact_u1(ti)
        exact2 = exact_u2(ti)
        error1 = abs(exact1 - ui1)
        error2 = abs(exact2 - ui2)
        data.append({
            't': round(ti, 6),
            'RK4 u1': round(ui1, 6),
            'Exact u1': round(exact1, 6),
            'Error in u1': round(error1,6),
            'RK4 u2': round(ui2, 6),
            'Exact u2': round(exact2, 6),
            'Error in u2': round(error2,6)
        })
    df = pd.DataFrame(data)
    print(f"\n=== Error Table for h = {h} ===")
    print(df.to_string(index=False))

# 生成並印出誤差表格
generate_error_table(t_rk1, u_rk1, h1)
generate_error_table(t_rk2, u_rk2, h2)    
