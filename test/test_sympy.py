import sympy as sy

x = sy.Symbol('x')
y = sy.Symbol('y')
f = x / (x + y)
f_prime = f.diff(x)
print(f_prime)