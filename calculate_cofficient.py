from sympy import symbols, solve

k = symbols('k')
equation = 447157819*k**20 - 0.3
solutions = solve(equation, k)
for solution in solutions:
    print(solution)