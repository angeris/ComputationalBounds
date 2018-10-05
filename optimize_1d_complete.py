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
s = 20
w = 16*pi
L = (N*N) * array(generate_laplacian(N).todense())
# g = -sin(4*pi*w*linspace(0, 1, N))/(4*pi)**2
g = zeros(N)
b = np.zeros(N)
# b[0] = 1
b = -sin(w*linspace(0, 1, N))
g[(N-1)//2 - s:(N-1)//2 + s] = 1/(w)**2
t_max = 3
a_diag = np.zeros(N)
a_diag[(N-1)//2 - s:(N-1)//2 + s] = 1
A = sparse.diags(a_diag)

# Terrible code... nobody should ever look at this.

def _obj(v):
    return .5*linalg.norm(A @ v[:N] - g)**2

def obj_jac(v):
    return r_[A.T @ (A @ v[:N] - g), zeros(N)]

def cons_eq(v):
    return L @ v[:N] + v[N:]*v[:N] - b

def cons_jac(v):
    return c_[L + diag(v[N:]), diag(v[:N])]

v_init = r_[zeros(N), zeros(N)]

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

u = cvx.Variable(N)
t = cvx.Variable(1)

obj = -.5*t -l.T * b - t_max*t_max*cvx.sum(eta)/4 + .5*linalg.norm(g)**2

min_eps = 0

cons = [
    eta >= min_eps,
    cvx.bmat([
                [t[newaxis,:], (L.T @ l + .5*l*t_max - A.T @ g)[newaxis,:]],
                [(L.T @ l + .5*l*t_max - A.T @ g)[:,newaxis], A.T @ A - cvx.diag(u)]
            ]) >> min_eps
]

for i in range(N):
    cons.append(
        cvx.norm(cvx.hstack([2*l[i], 2*eta[i] - u[i]])) <= 2*eta[i] + u[i]
    )

prob = cvx.Problem(cvx.Maximize(obj), cons)
prob.solve(solver=cvx.MOSEK, verbose=True)

print('Dual opt val : {}'.format(prob.value))