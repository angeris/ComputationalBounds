from pylab import *

from scipy import sparse
from scipy.sparse import linalg as slinalg
from scipy import linalg
from scipy import optimize

from tqdm import tqdm

import cvxpy as cvx

def generate_laplacian(n, L=1):
    diagonals = ones(n)
    D = sparse.diags([diagonals[:-1], -2*diagonals, diagonals[:-1]], [-1,0,1])
    return D

all_dual_vals = []

n = 30
N = 301
for w in tqdm(linspace(.01*pi, 100*pi, n)):
    # s = 5
    # w = 30*pi
    t_min = 1
    t_max = 2 - t_min
    L = (N*N)/(w*w) * array(generate_laplacian(N).todense()) + t_min*eye(N)
    # g = -sin(4*pi*w*linspace(0, 1, N))/(4*pi)**2
    # g = zeros(N)
    x = linspace(0, 1, N)
    sigma = .1
    g = sin(x*w)*exp(-(x - .5)**2/(2*(sigma)**2))
    # g = sin(x*w)
    # g[(N-1)//2:] = 0
    b = np.zeros(N)
    # b[0] = 1
    # b[(N-1)//2 - s: (N-1)//2 + s] = -1
    # g[(N-1)//2 - s:(N-1)//2 + s] = .01
    a_diag = np.ones(N)
    # a_diag[74:125] = 1
    # a_diag[2*(N-1)//5:3*(N-1)//5] = 0
    A = sparse.diags([a_diag], [0])

    # Best dual program

    l = cvx.Variable(N)
    nu = cvx.Variable(N)

    obj = - nu @ b - t_max*t_max*cvx.sum(l)/4 + .5*((a_diag * g) @ (a_diag * g))

    cons = []

    for i in range(N):
        if abs(a_diag[i]) < 1e-5:
            cons += [
                nu[i] == 0,
                (L.T @ nu)[i] == 0
            ]
            continue
        obj += -.5*cvx.quad_over_lin((L.T @ nu)[i] + .5*nu[i]*t_max - g[i], a_diag[i] - .5*cvx.quad_over_lin(nu[i], l[i]))

    cons += [
        l >= 0
    ]

    prob = cvx.Problem(cvx.Maximize(obj), cons)
    prob.solve(solver=cvx.MOSEK, verbose=True)

    print('Dual opt val : {}'.format(prob.value))
    all_dual_vals.append(prob.value)

print(all_dual_vals)