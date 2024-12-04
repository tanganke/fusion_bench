import sympy as sp
k = sp.symbols('k')

# A = sp.Matrix([
#     [1, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0],
#     [0, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0],
#     [0, 0, 0, 0, 1]
# ])

# B = sp.Matrix([
#     [k, k, 0, 0, k],
#     [k, k, k, 0, 0],
#     [0, k, k, k, 0],
#     [0, 0, k, k, k],
#     [k, 0, 0, k, k]
# ])


dim = int(input('dim:'))
A = sp.eye(dim)

def value(i, j):
    divisor = dim if (j - i) > 0 else -dim
    dif = j - i
    ans = k if (dif % divisor in [0, 1, -1, dim-1, 1-dim]) else 0
    return ans
B = sp.Matrix(dim, dim, value)

print('A:', A)
print('B:', B)

num = int(input('num:'))
for i in range(num):
    A = A * B

print(A)
print(float(A[0, 0] / A[0, 3]))