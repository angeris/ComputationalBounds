using SparseArrays
using PyPlot
using LinearAlgebra
using JuMP
using ProgressMeter
using JLD
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

rc("text", usetex=true)

t_min = 1
t_max = 2 - t_min

N = 251  # number of points in domain
freqs = [30*pi, 40*pi, 50*pi]
s = round(Int, 6 * (N-1) / 50)   # size of mode box in domain
n_freq = length(freqs) # number of frequencies for the given problem

to_cartesian = CartesianIndices((N, N))
to_linear = LinearIndices((N, N))

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

    weights = 5*ones(N*N)
    g = zeros(N*N)
    b = zeros(N*N)
    L_1d = (N*N)/(w*w) * generate_lapl(N)
    L = kron(L_1d, sparse(I, N, N)) + kron(sparse(I, N, N), L_1d) + t_min * spdiagm(0 => ones(N*N))

    filter_square!(weights, beg_points[i], end_points[i], to_linear, 1)
    filter_square!(g, beg_points[i], end_points[i], to_linear, 1)

    push!(weights_all, weights)
    push!(g_all, g)
    push!(b_all, b)
    push!(L_all, L)
end



z_init = zeros(n_freq, N*N)
t_init = t_min*ones(N*N)

m = Model(solver=Gurobi.GurobiSolver())

@variable(m, nu[1:N*N, 1:n_freq])
@variable(m, t[1:N*N])

@objective(m, Max, -.5*sum(t) - sum(nu[:,i]' * b_all[i] for i=1:n_freq))

@showprogress "Generating constraints..." for j=1:N*N
    quad_cons(m, [ (L_all[i][:,j]' * nu[:,i] - (weights_all[i][j]^2)*g_all[i][j]) / weights_all[i][j] for i=1:n_freq ], t[j])
    quad_cons(m, [ (L_all[i][:,j]' * nu[:,i] + t_max*nu[j,i] - (weights_all[i][j]^2)*g_all[i][j]) / weights_all[i][j] for i=1:n_freq ], t[j])
end

@time status = solve(m)

nu_sol = getvalue(nu)
t_sol = getvalue(t)

lower_bound = (-.5*sum(t_sol) + sum(-nu_sol[:,i]' * b_all[i] + .5*norm(weights_all[i] .* g_all[i]).^2 for i=1:n_freq))

theta_init = zeros(N*N)
zero_ind = sum((L_all[i]' * nu_sol[:,i] - (weights_all[i] .^ 2) .* g_all[i] + nu_sol[:,i] .* t_max).^2 ./ (weights_all[i] .^ 2) for i=1:n_freq) .< sum((L_all[i]' * nu_sol[:,i] - (weights_all[i] .^ 2) .* g_all[i]).^2 ./ (weights_all[i] .^ 2) for i=1:n_freq)
theta_init[zero_ind] .= 0
theta_init[.~zero_ind] .= t_max

for i=1:n_freq
    z_init[i,:] .= g_all[i] - (L_all[i]'*nu_sol[:,i] + nu_sol[:,i] .* theta_init) ./ (weights_all[i] .^ 2)
end

figure(figsize=(8, 5))
for i=1:n_freq
    subplot(1, 3, i)
    title("\$(z^0)^$i\$")
    imshow(reshape(z_init[i,:], N, N), cmap="Spectral")
end
savefig("resonator_z_primal_n$N.pdf", bbox_inches="tight")
close()

jldopen("output_files/dual_output.jld", "w") do file
    write(file, "theta_init", theta_init)
    write(file, "z_init", z_init)
    write(file, "n", N)
    write(file, "lower_bound", lower_bound)
    write(file, "nu_opt", nu_sol)
end

# New primal solver
curr_z = copy(z_init)
curr_theta = copy(theta_init)
# curr_z = zeros(3, N*N)
# curr_theta = ones(N*N)*t_min
curr_nu = zeros(n_freq, N*N)
rho = 100
maxiter = 600
prev_conv_val = Inf

all_feas_tol = zeros(maxiter)
all_obj = zeros(maxiter)
all_design = zeros(maxiter, N*N)
all_field = zeros(maxiter, n_freq, N*N)

alpha = 1.
tau = .9

last_iteration = 0

for n = 1:maxiter
    global last_iteration, curr_z, curr_theta, curr_nu

    @time begin
    for i=1:n_freq
        curr_z[i,:] .= solve_max_eq(L_all[i], curr_theta, curr_nu[i,:], rho, weights_all[i], g_all[i], b_all[i])
    end

    # Minimize over design
    curr_theta .= min_sq_diag([weights_all[i] .* curr_z[i,:] for i=1:n_freq], [weights_all[i] .* (b_all[i] - curr_nu[i,:] - L_all[i] * curr_z[i,:]) for i=1:n_freq], t_max, n_freq)

    end

    # Update dual
    cons_val = 0
    for i=1:n_freq
        resid = L_all[i] * curr_z[i,:] + curr_z[i,:] .* curr_theta - b_all[i]
        curr_nu[i,:] .+= resid
        cons_val += norm(resid)^2
    end

    all_feas_tol[n] = sqrt(cons_val)

    obj_val = sum(.5*norm(weights_all[i].*(curr_z[i,:] - g_all[i]))^2 for i=1:n_freq)

    all_obj[n] = obj_val
    all_design[n, :] .= curr_theta
    all_field[n, :, :] .= curr_z

    last_iteration = n

    if cons_val < 1e-4
        @show cons_val
        # break
    end

    @show n
    @show cons_val
    @show obj_val

    if ((n-1)%10 == 0)
        figure()
        title(L"$\theta$")
        imshow(reshape(curr_theta, N, N), cmap="Purples")
        savefig("iter_image/resonator_theta_n$(N)_iter_$(n).pdf", bbox_inches="tight")
        close()
    end

    prev_conv_val = cons_val
end

obj_val = sum(.5*norm(weights_all[i].*(curr_z[i,:] - g_all[i]))^2 for i=1:n_freq)
@show obj_val
@show lower_bound
@show last_iteration

jldopen("output_files/primal_output_dual_initialization.jld", "w") do file
    write(file, "theta_opt", curr_theta)
    write(file, "z_opt", curr_z)
    write(file, "all_obj", all_obj)
    write(file, "all_feas_tol", all_feas_tol)
    write(file, "n", N)
    write(file, "all_design", all_design)
    write(file, "all_field", all_field)
end

figure(figsize=(8, 5))
for i=1:n_freq
    subplot(1, 3, i)
    title("\$z^$i\$")
    imshow(reshape(curr_z[i,:], N, N), cmap="Spectral")
end
savefig("resonator_z_n$N.pdf", bbox_inches="tight")
close()

figure(figsize=(8, 3))
for i=1:n_freq
    subplot(1, 3, i)
    title("\$S^$i\$")
    imshow(reshape(g_all[i], N, N), cmap="Purples")
end
savefig("boxes_n$N.pdf", bbox_inches="tight")
close()


figure(figsize=(8,5))
subplot(121)
title(L"$\theta^0$")
imshow(reshape(theta_init, N, N), cmap="Purples")
subplot(122)
title(L"$\theta$")
imshow(reshape(curr_theta, N, N), cmap="Purples")
savefig("resonator_theta_n$N.pdf", bbox_inches="tight")
close()

figure(figsize=(8, 5.5))
for i=1:n_freq
    subplot(2, 3, i)
    title("\$z^$i\$")
    imshow(reshape(curr_z[i,:], N, N), cmap="Spectral")
    subplot(2, 3, 3+i)
    title("\$(z^$i)^0\$")
    imshow(reshape(z_init[i,:], N, N), cmap="Spectral")
end
savefig("resonator_z_comparison_n$N.pdf", bbox_inches="tight")
close()

figure(figsize=(10, 3))
subplot(1, 2, 1)
title("Objective per iteration")
plot(1:maxiter, all_obj)
ylabel("Objective")
xlabel("Iteration")

subplot(1, 2, 2)
title("Feasibility residual")
semilogy(1:maxiter, all_feas_tol)
ylabel(L"$\|(z^1, z^2, z^3)\|_2^2$")
xlabel("Iteration")

savefig("objective_iteration_comparison_ab_initio.pdf", bbox_inches="tight")
close()
