using SparseArrays
using PyPlot
using LinearAlgebra
using JuMP
import Gurobi

function generate_lapl(n)
    L = spdiagm(-1 => ones(n-1), 0 => -2*ones(n), 1 => ones(n-1))
    return L
end

function linspace(s, e, n)
    return collect(range(s, e, length=n))
end

function proj(x, l, u)
    return max.(min.(x, u), l)
end

function filter_square!(a, top_left, bottom_right, to_linear, value)
    for J in CartesianIndices(bottom_right - top_left)
        a[to_linear[top_left + J]] = value
    end
end

function vec_pinv(x; tol=1e-14)
    pinv_x = copy(x)
    tol_entries = abs.(pinv_x) .> tol
    pinv_x[tol_entries] .= 1 ./ pinv_x[tol_entries]
    return pinv_x
end

function min_sq_diag(v, h, u; tol=1e-14)
    # minimizes ||diag(v)*x - h||, s.t. 0 ≤ x ≤ u
    v_pinv = vec_pinv(v)
    ul_bounds = v .* u
    return v_pinv .* proj(h, min.(0, ul_bounds), max.(0, ul_bounds))
end

function solve_max_eq!(curr_z, L_w, curr_theta, rho, W)
    A_theta = L_w + spdiagm(0 => curr_theta)
    curr_z .= Symmetric(W^2 + rho * A_theta' * A_theta) \ (W^2 * g + rho * A_theta' * (b - curr_nu))
end

t_min = 1
t_max = 2 - t_min

N = 101  # number of points in domain
nw = 25 # frequency (as multiples of pi)
w = nw*pi
s = round(Int, 7 * (N-1) / 50)   # size of mode box in domain
n_freq = 3 # number of frequencies for the given problem

L_1d = (N*N)/(w*w) * generate_lapl(N)
L = kron(L_1d, sparse(I, N, N)) + kron(sparse(I, N, N), L_1d) + t_min * spdiagm(0 => ones(N*N))

to_cartesian = CartesianIndices((N, N))
to_linear = LinearIndices((N, N))

weights = 5*ones(N*N)
g = zeros(N*N)
b = zeros(N*N)

mid_point = div(N-1, 2)

# filter_square!(weights, CartesianIndex(mid_point-s, 0), CartesianIndex(mid_point+s, 2*s), to_linear, 1)
filter_square!(weights, CartesianIndex(mid_point-s, mid_point-s), CartesianIndex(mid_point+s, mid_point+s), to_linear, 1)
filter_square!(g, CartesianIndex(mid_point-s, mid_point-s), CartesianIndex(mid_point+s, mid_point+s), to_linear, 1)
# filter_square!(weights, CartesianIndex(mid_point-s, N-2*s), CartesianIndex(mid_point+s, N), to_linear, 1)

W = spdiagm(0 => weights)

z_init = zeros(N*N)
t_init = t_min*ones(N*N)

m = Model(solver=Gurobi.GurobiSolver())

@variable(m, nu[1:N*N])
@variable(m, t[1:N*N])

@objective(m, Max, -.5*sum((t .* t) ./ (weights .^ 2)) - nu' * b)

scaled_g = (W^2) * g

@constraint(m, t .>= scaled_g - L' * nu)
@constraint(m, t .>= L' * nu - scaled_g)
@constraint(m, t .>= (t_max * nu) - (scaled_g - L' * nu))
@constraint(m, t .>= (scaled_g - L' * nu) - (t_max * nu))

status = solve(m)

nu_sol = getvalue(nu)
t_sol = getvalue(t)

lower_bound = (-.5*sum((t_sol .* t_sol) ./ (weights.^2)) - nu_sol' * b + .5*norm(W * g).^2)

theta_init = zeros(N*N)
zero_ind = abs.(L' * nu_sol - scaled_g + nu_sol .* t_max) .< abs.(L' * nu_sol - scaled_g)
theta_init[zero_ind] .= 0
theta_init[.~zero_ind] .= t_max

figure()
title("Resonator theta, primal")
imshow(reshape(theta_init, N, N), cmap="Purples")
savefig("resonator_theta_primal_n$N.png")
show()

z_init = g - (L'*nu_sol + nu_sol .* theta_init) ./ (weights .^ 2)

figure()
title("Resonator field, primal")
imshow(reshape(z_init, N, N), cmap="PuRd")
savefig("resonator_z_primal_n$N.png")
show()

# New primal solver
curr_z = copy(z_init)
curr_theta = copy(theta_init)
curr_nu = zeros(N*N)
rho = 10
maxiter = 1000
prev_conv_val = Inf

alpha = 1.
tau = .9

last_iteration = 0

for i = 1:maxiter
    global last_iteration
    solve_max_eq!(curr_z, L, curr_theta, rho, W)

    # Minimize over design
    curr_theta .= min_sq_diag(weights .* curr_z, weights .* (b - curr_nu - L * curr_z), t_max)

    # Update dual
    resid = L * curr_z + curr_z .* curr_theta - b
    curr_nu .+= resid

    obj_val = .5*norm(curr_z - g)^2
    cons_val = maximum(abs.(resid))

    last_iteration = i

    if cons_val < 1e-4
        @show cons_val
        break
    end

    @show i
    @show cons_val

    prev_conv_val = cons_val
end

obj_val = .5*norm(W * (curr_z - g))^2
@show obj_val
@show lower_bound
@show last_iteration

figure()
title("Resonator field")
imshow(reshape(curr_z, N, N), cmap="PuRd")
savefig("resonator_z_n$N.png")
show()

figure()
title("Resonator theta")
imshow(reshape(curr_theta, N, N), cmap="Purples")
savefig("resonator_theta_n$N.png")
show()
