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

function proj(x, l, u)
    return max.(min.(x, u), l)
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

# Helper functions for problem formulation

# Formulates ||x||^2 <= y
function quad_cons(m, x, y)
    @constraint(m, sum(x.^2) <= y)
end

t_min = 1
t_max = 2 - t_min

N = 101  # number of points in domain
nw = 30 # frequency (as multiples of pi)
w = nw*pi
s = round(Int, 7 * (N-1) / 50)   # size of mode box in domain
n_freq = 3 # number of frequencies for the given problem

L = (N*N)/(w*w) * generate_lapl(N) + t_min * spdiagm(0 => ones(N))

x_dom = linspace(-1, 1, N)

points = 10

weights = .001ones(N)
weights[1:points] .= 1
weights[end-points:end] .= 1
# mid_point = div(N+1, 2)
# s_width = 50
# weights[mid_point - s_width:mid_point+s_width] .= 5
# g = sin.(w*pi*x_dom/2)./(w*pi*x_dom/2)
# g[div(N+1, 2)] = 1

g = zeros(N)
g[end-points:end].=1
g[1]=0
b = zeros(N)

W = spdiagm(0 => weights)

z_init = zeros(N)
t_init = t_min*ones(N)

m = Model(with_optimizer(Gurobi.Optimizer))

@variable(m, nu[1:N])
@variable(m, t[1:N])

@objective(m, Max, -.5*sum(t) - nu' * b)

scaled_g = (W^2) * g

@showprogress "Generating constraints" for i = 1:N
    quad_cons(m, (L[:,i]' * nu - scaled_g[i]) / weights[i], t[i])
    quad_cons(m, (L[:,i]' * nu + t_max * nu[i] - scaled_g[i]) / weights[i], t[i])
end
optimize!(m)

nu_sol = [value(nu[i]) for i=1:N]
t_sol = [value(t[i]) for i=1:N]

lower_bound_cons = -.5*sum(max.(((L' * nu_sol - scaled_g) ./ weights).^2,
                                ((L' * nu_sol + t_max * nu_sol - scaled_g) / weights).^2)) - nu_sol' * b + .5*norm(W * g).^2

theta_init = zeros(N)
zero_ind = abs.(L' * nu_sol - scaled_g + nu_sol .* t_max) .< abs.(L' * nu_sol - scaled_g)
theta_init[zero_ind] .= 0
theta_init[.~zero_ind] .= t_max

z_init = g - (L'*nu_sol + nu_sol .* theta_init) ./ (weights .^ 2)

figure(figsize=(12, 8))
subplot(211)
title("Resonator theta, primal")
step(x_dom, theta_init .+ t_min, label="initial structure")

subplot(212)
plot(x_dom, z_init, label="initial field", linestyle=":")
plot(x_dom, g, label="desired field")
legend()
savefig("1d_sinc_resonator/resonator_primal_n$N.png")
close()

# New primal solver
curr_z = copy(z_init)
curr_theta = copy(theta_init)
curr_nu = zeros(N)
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
    curr_theta .= min_sq_diag(curr_z, b - curr_nu - L * curr_z, t_max)

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
A_com = spdiagm(0 => 1 ./ weights)*(L + spdiagm(0 => curr_theta))'
nu_guess =  (A_com'*A_com) \ (A_com' * (weights.^2 .* g) - b)

lower_bound_guess = -.5*sum((1 ./ weights .^ 2) .* max.((L' * nu_guess - (weights .^ 2) .* g).^2, (L' * nu_guess + t_max * nu_guess - (weights .^ 2) .* g).^2)) - nu_guess'*b + .5*norm(weights .* g)
@show lower_bound_guess
@show lower_bound_cons
@show last_iteration
@show obj_val/lower_bound_cons

figure(figsize=(12, 8))
subplot(211)
title("Resonator field")
plot(x_dom, curr_theta .+ t_min, label="structure")
subplot(212)
plot(x_dom, curr_z, label="field", linestyle=":")
plot(x_dom, g, label="desired field")
legend()
savefig("1d_sinc_resonator/resonator_n$N.png")
close()


