from pylab import *

from scipy import sparse
from scipy.sparse import linalg as slinalg
from scipy import linalg
from scipy import optimize

import cvxpy as cvx

def generate_laplacian(n, L=1):
    diagonals = L*ones(n)
    D = sparse.diags([diagonals[:-1], -2*diagonals, diagonals[:-1]], [-1,0,1]).tocsr()
    return D


N = 101
w = 20*pi
t_min = 1
L = (N*N)/(w*w) * array(generate_laplacian(N).todense()) + t_min*eye(N)
x = linspace(0, 1, N)
g = sin(x*w)*exp(-(x - .5)**2/(.05))
# g = zeros(N)
b = zeros(N)
# b[(N-1)//2] = -1
# b[(N-1)//2 - s: (N-1)//2 + s] = -1
t_max = 2 - t_min
# g[2*(N-1)//5:3*(N-1)//5] = 30*t_max/N**2

# Terrible code... nobody should ever look at this.

def _obj(v):
    return .5*linalg.norm(v[:N] - g)**2

def obj_jac(v):
    return r_[v[:N] - g, zeros(N)]

def cons_eq(v):
    return L @ v[:N] + v[N:]*v[:N] - b

def cons_jac(v):
    return c_[L + diag(v[N:]), diag(v[:N])]

v_init = r_[ones(N), ones(N)]

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
plot(list(range(N)), g, label='desired')
legend()
subplot(212)
step(list(range(N)), optim_res.x[N:])
show()

# Best dual program

l = cvx.Variable(N)
eta = cvx.Variable(N)

obj = -l @ b - t_max*t_max*cvx.sum(eta)/4 + .5*linalg.norm(g)**2

min_eps = 0

for i in range(N):
    obj += -.25*cvx.quad_over_lin((L.T * l)[i] - .5*eta[i]*t_max - g[i], .5 - .25*cvx.quad_over_lin(l[i], eta[i]))

cons = [
    eta >= min_eps,
]

prob = cvx.Problem(cvx.Maximize(obj), cons)

prob.solve(solver=cvx.MOSEK, verbose=True)

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