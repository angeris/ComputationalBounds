using SparseArrays
using PyPlot
using LinearAlgebra
using JuMP
using ProgressMeter
import Gurobi

function generate_lapl(n)
    L = spdiagm(-1 => ones(n-1), 0 => -2*ones(n), 1 => ones(n-1))
    return L
end

function linspace(s, e, n)
    return collect(range(s, e, length=n))
end

function filter_square!(a, top_left, bottom_right, to_linear, value)
    for J in CartesianIndices(bottom_right - top_left)
        a[to_linear[top_left + J]] = value
    end
end

function min_sq_diag(v, h, u, n_freq; tol=1e-14)
    # minimizes sum_i(||diag(v_i)*x - h_i||^2), s.t. 0 ≤ x ≤ u
    w_comp = sum(v[i] .* h[i] for i=1:n_freq)
    inv_comp = sum(v[i].^2 for i=1:n_freq)
    return clamp.(w_comp ./ inv_comp, 0, u)
end

function solve_max_eq(L_w, curr_theta, curr_nu, rho, weights, g, b)
    A_theta = L_w + spdiagm(0 => curr_theta)
    return Symmetric(spdiagm(0 => weights.^2) + rho * A_theta' * A_theta) \ (weights .* g + rho * A_theta' * (b - curr_nu))
end

# Helper functions for problem formulation

# Formulates ||x||^2 <= y
function quad_cons(m, x, y)
    @constraint(m, norm([2*x; y-1]) <= y+1)
end

t_min = 1
t_max = 2 - t_min

N = 1001  # number of points in domain
freqs = [30*pi]
s = round(Int, 6 * (N-1) / 50)   # size of mode box in domain
n_freq = length(freqs) # number of frequencies for the given problem

weights_all = []
g_all = []
b_all = []
L_all = []

mid_point = div(N-1, 2)

beg_points = [
    CartesianIndex(mid_point-s, 0),
    CartesianIndex(mid_point-s, mid_point-s),
    CartesianIndex(mid_point-s, N-2*s)
]

end_points = [
    CartesianIndex(mid_point+s, 2*s),
    CartesianIndex(mid_point+s, mid_point+s),
    CartesianIndex(mid_point+s, N)
]

for i=1:n_freq
    global weights_all, g_all, b_all, L_all
    w = freqs[i]

    weights = ones(N)
    x_dom = linspace(-1, 1, N)

    g = sin.(w*pi*x_dom/2)./(w*pi*x_dom/2)
    g[div(N+1, 2)] = 1

    b = zeros(N)

    L = (N*N)/(w*w) * generate_lapl(N)

    push!(weights_all, weights)
    push!(g_all, g)
    push!(b_all, b)
    push!(L_all, L)
end

z_init = zeros(n_freq, N)
t_init = t_min*ones(N)

m = Model(solver=Gurobi.GurobiSolver())

@variable(m, nu[1:N, 1:n_freq])
@variable(m, t[1:N])

@objective(m, Max, -.5*sum(t) - sum(nu[:,i]' * b_all[i] for i=1:n_freq))

@showprogress "Generating constraints..." for j=1:N
    quad_cons(m, [ (L_all[i][:,j]' * nu[:,i] - (weights_all[i][j]^2)*g_all[i][j]) / weights_all[i][j] for i=1:n_freq ], t[j])
    quad_cons(m, [ (L_all[i][:,j]' * nu[:,i] + t_max*nu[j,i] - (weights_all[i][j]^2)*g_all[i][j]) / weights_all[i][j] for i=1:n_freq ], t[j])
end

status = solve(m)

nu_sol = getvalue(nu)
t_sol = getvalue(t)

lower_bound = (-.5*sum(t_sol) + sum(-nu_sol[:,i]' * b_all[i] + .5*norm(weights_all[i] .* g_all[i]).^2 for i=1:n_freq))

theta_init = zeros(N)
zero_ind = sum((L_all[i]' * nu_sol[:,i] - (weights_all[i] .^ 2) .* g_all[i] + nu_sol[:,i] .* t_max).^2 ./ (weights_all[i] .^ 2) for i=1:n_freq) .< sum((L_all[i]' * nu_sol[:,i] - (weights_all[i] .^ 2) .* g_all[i]).^2 ./ (weights_all[i] .^ 2) for i=1:n_freq)
theta_init[zero_ind] .= 0
theta_init[.~zero_ind] .= t_max

for i=1:n_freq
    z_init[i,:] .= g_all[i] - (L_all[i]'*nu_sol[:,i] + nu_sol[:,i] .* theta_init) ./ (weights_all[i] .^ 2)
end

# New primal solver
curr_z = copy(z_init)
curr_theta = copy(theta_init)
curr_nu = zeros(n_freq, N)
rho = 10
maxiter = 1000
prev_conv_val = Inf

alpha = 1.
tau = .9

last_iteration = 0

for n = 1:maxiter
    global last_iteration, curr_z, curr_theta, curr_nu
    for i=1:n_freq
        curr_z[i,:] .= solve_max_eq(L_all[i], curr_theta, curr_nu[i,:], rho, weights_all[i], g_all[i], b_all[i])
    end

    # Minimize over design
    curr_theta .= min_sq_diag([weights_all[i] .* curr_z[i,:] for i=1:n_freq], [weights_all[i] .* (b_all[i] - curr_nu[i,:] - L_all[i] * curr_z[i,:]) for i=1:n_freq], t_max, n_freq)

    # Update dual
    cons_val = 0
    for i=1:n_freq
        resid = L_all[i] * curr_z[i,:] + curr_z[i,:] .* curr_theta - b_all[i]
        curr_nu[i,:] .+= resid
        cons_val = max(maximum(abs.(resid)), cons_val)
    end

    obj_val = sum(.5*norm(weights_all[i].*(curr_z[i,:] - g_all[i]))^2 for i=1:n_freq)

    last_iteration = n

    if cons_val < 1e-4
        @show cons_val
        break
    end

    @show n
    @show cons_val
    @show obj_val

    prev_conv_val = cons_val
end

obj_val = sum(.5*norm(weights_all[i].*(curr_z[i,:] - g_all[i]))^2 for i=1:n_freq)
@show obj_val
@show lower_bound
@show last_iteration
@show obj_val/lower_bound
