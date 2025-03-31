from sympy import symbols, Eq, solve
import numpy as np
from scipy.interpolate import CubicHermiteSpline

def problem1():
    x = symbols('x')
    y = 0 
    equation = Eq(
        -0.440181 * ((x-0.4)*(x-0.5)*(x-0.6)) / ((0.3-0.4)*(0.3-0.5)*(0.3-0.6)) +
        -0.270320 * ((x-0.3)*(x-0.5)*(x-0.6)) / ((0.4-0.3)*(0.4-0.5)*(0.4-0.6)) +
        -0.106531 * ((x-0.4)*(x-0.3)*(x-0.6)) / ((0.5-0.4)*(0.5-0.3)*(0.5-0.6)) +
        0.05188 * ((x-0.4)*(x-0.5)*(x-0.3)) / ((0.6-0.4)*(0.6-0.5)*(0.6-0.3)),
        y  
    )
    
    solutions = solve(equation, x)#solve y = 0 ,x =?
    print(solutions[0])

def problem3():

    T = np.array([0, 3, 5, 8, 13])  # Time 
    D = np.array([0, 200, 375, 620, 990])  # Distance 
    V = np.array([75, 77, 80, 74, 72])  # Velocity 
    
    hermite_poly = CubicHermiteSpline(T, D, V) # Hermite polynomial
    
    example_time = 10
    example_distance = hermite_poly(example_time)
    hermite_velocity = hermite_poly.derivative()
    example_velocity = hermite_velocity(example_time)
    
    print("a.")
    print(f"Distance at time {example_time} seconds: {example_distance:.2f} feet")
    print(f"Velocity at time {example_time} seconds: {example_velocity:.2f} feet/second")
    
    # Convert mi/h to ft/s
    speed_limit = 55 * 1.467  
    
    # Generate fine time points to evaluate velocity
    time_points = np.linspace(0, 13, 10000)  
    velocities = hermite_velocity(time_points)  # Compute velocities at these points
    
    # Find when the velocity first exceeds the speed limit
    exceed_indices = np.where(velocities > speed_limit)[0]
    
    print("b.")
    first_exceed_time = time_points[exceed_indices[0]]
    print(f"The car first exceeds {speed_limit:.2f} ft/s at time {first_exceed_time:.2f} seconds")

    
    # Part (c): Find the predicted maximum speed
    max_velocity = np.max(velocities)
    
    print("c.")
    print(f"The predicted maximum speed is {max_velocity:.2f} feet/second")
print('numerical_hw3_problem1')
problem1()
print("")
print('numerical_hw3_problem3')
problem3()
