from pylab import *

from scipy import sparse
from scipy.sparse import linalg as slinalg
from scipy import linalg
from scipy import optimize

import cvxpy as cvx

def generate_laplacian(n, L=1):
    diagonals = L*ones(n)
    D = sparse.diags([diagonals[:-1], -2*diagonals, diagonals[:-1]], [-1,0,1])
    return D


N = 251
w = 1
L = (N*N) * array(generate_laplacian(N).todense())
g = -sin(4*pi*w*linspace(0, 1, N))/(4*pi)**2
# g = zeros(N)
b = sin(4*pi*w*linspace(0, 1, N))
g[(N-1)//2:] = 0
t_max = 10

# Terrible code... nobody should ever look at this.

def _obj(v):
    return .5*linalg.norm(v[:N] - g)**2

def obj_jac(v):
    return r_[v[:N] - g, zeros(N)]

def cons_eq(v):
    return L @ v[:N] + w*w*v[N:]*v[:N] - b

def cons_jac(v):
    return c_[L + w*w*diag(v[N:]), w*w*diag(v[:N])]

v_init = r_[zeros(N), ones(N)]

all_bounds = [(None, None) for _ in range(N)] + [(0, t_max) for _ in range(N)]

optim_res = optimize.minimize(_obj, v_init, 
                              method='SLSQP', jac=obj_jac,
                              bounds=all_bounds,
                              constraints={'type':'eq', 'fun':cons_eq, 'jac':cons_jac},
                              options={'disp': True, 'maxiter':1000})

print('Optimal value : {}'.format(_obj(optim_res.x)))
print('Optimal residual : {}'.format(linalg.norm(cons_eq(optim_res.x))))

subplot(211)
plot(optim_res.x[:N], label='actual')
plot(g, label='desired')
legend()
subplot(212)
step(list(range(N)), optim_res.x[N:])
show()

# Best dual program

l = cvx.Variable(N)
eta = cvx.Variable(N)

obj = -l.T * b - t_max*t_max*cvx.sum(eta)/4 + .5*linalg.norm(g)**2

min_eps = 0

for i in range(N):
    obj += -.5*cvx.quad_over_lin((L.T * l)[i] + .5*l[i]*t_max - g[i], 1 - cvx.quad_over_lin(l[i], 2*eta[i]))

cons = [
    eta >= min_eps,
    cvx.square(l) <= 2*eta - min_eps
]

prob = cvx.Problem(cvx.Maximize(obj), cons)

prob.solve(solver=cvx.MOSEK)

print('Dual opt val : {}'.format(prob.value))
# print(l.value)
# print(eta.value)

# Slightly crappier dual program
l = cvx.Variable(N)

cons = [
    l >= np.sqrt(min_eps)
]

obj = -l @ b - cvx.sum_squares(t_max * l)/4 + .5*linalg.norm(g)**2 - cvx.sum_squares(L.T @ l + .5*l*t_max - g)

prob = cvx.Problem(cvx.Maximize(obj), cons)

prob.solve(solver=cvx.MOSEK)

print('Dual slice opt val : {}'.format(prob.value))