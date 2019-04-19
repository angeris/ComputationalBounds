using PyPlot
using JuMP
using ProgressMeter
# using JLD
using MosekTools

# Helper functions for problem formulation
include("utilities.jl")

rc("text", usetex=true)

const PLOT_ITERATES = false

const t_min = 1
const t_max = 2 - t_min

const N = 251  # number of points in domain
const freqs = [30*pi, 40*pi, 50*pi]
const s = round(Int, 6 * (N-1) / 50)   # size of mode box in domain
const n_freq = length(freqs) # number of frequencies for the given problem

to_cartesian = CartesianIndices((N, N))
to_linear = LinearIndices((N, N))

# Generates all required matrices
weights_all = []  # List containing a vector for each mode with the appropriate weights (W^i in the paper)
z_hat_all = []    # List containing a vector for each mode with the desired field (\hat z^i in the paper)
b_all = []        # List containing a vector for each mode with the excitation (b^i in the paper)
L_all = []        # List containing a matrix for each mode with the desired physics equation (A^i in the paper)

# Indices of all squares found in figures
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

# Syntactic sugar for kronecker product
⊗ = kron

# Generates the above lists for the example problem described in the paper,
# modify this to fit your own problem.
for i=1:n_freq
    global weights_all, z_hat_all, b_all, L_all
    w = freqs[i]

    weights = 5*ones(N*N)
    z_hat = zeros(N*N)
    b = zeros(N*N)
    L_1d = (N*N)/(w*w) * generate_lapl(N)
    L = L_1d ⊗ sparse(I, N, N) + sparse(I, N, N) ⊗ L_1d + t_min * I

    # Generates the square masks of figure 1.
    filter_square!(weights, beg_points[i], end_points[i], to_linear, 1)
    filter_square!(z_hat, beg_points[i], end_points[i], to_linear, 1)

    push!(weights_all, weights)
    push!(z_hat_all, z_hat)
    push!(b_all, b)
    push!(L_all, L)
end

z_init = zeros(n_freq, N*N)
t_init = t_min*ones(N*N)

@info "Generating model"

m = Model(with_optimizer(Mosek.Optimizer))

@variable(m, nu[1:N*N, 1:n_freq])
@variable(m, t[1:N*N])

@info "Generating objective"

@objective(m, Max, -.5*sum(t) - sum(sum(nu[j,i]' * b_all[i][j] for j=1:N^2 if !iszero(b_all[i][j])) for i=1:n_freq))

@showprogress "Generating constraints..." for j=1:N*N
    quad_cons(m, [ (sum(val * nu[k,i] for (k, val) = zip(findnz(L_all[i][:, j])...)) -
                        (weights_all[i][j]^2)*z_hat_all[i][j]) / weights_all[i][j] for i=1:n_freq ], t[j])
    quad_cons(m, [ (sum(val * nu[k,i] for (k, val) = zip(findnz(L_all[i][:, j])...)) +
                        t_max*nu[j,i] - (weights_all[i][j]^2)*z_hat_all[i][j]) / weights_all[i][j] for i=1:n_freq ], t[j])
end

@time optimize!(m)

nu_sol = value.(nu)
t_sol = value.(t)

@show lower_bound = (-.5*sum(t_sol) + sum(-nu_sol[:,i]' * b_all[i] + .5*norm(weights_all[i] .* z_hat_all[i]).^2 for i=1:n_freq))

# Equation (10) in the paper
theta_init = zeros(N*N)
zero_ind = sum((L_all[i]' * nu_sol[:,i] - (weights_all[i] .^ 2) .* z_hat_all[i] + nu_sol[:,i] .* t_max).^2 ./ (weights_all[i] .^ 2) for i=1:n_freq) .< 
                sum((L_all[i]' * nu_sol[:,i] - (weights_all[i] .^ 2) .* z_hat_all[i]).^2 ./ (weights_all[i] .^ 2) for i=1:n_freq)
theta_init[zero_ind] .= 0
theta_init[.~zero_ind] .= t_max

# Equation (11) in the paper
for i=1:n_freq
    z_init[i,:] .= z_hat_all[i] - (L_all[i]'*nu_sol[:,i] + nu_sol[:,i] .* theta_init) ./ (weights_all[i] .^ 2)
end

figure(figsize=(8, 5))
for i=1:n_freq
    subplot(1, 3, i)
    title("\$(z^0)^$i\$")
    imshow(reshape(z_init[i,:], N, N), cmap="Spectral")
end
savefig("resonator_z_primal_n$N.pdf", bbox_inches="tight")
close()

# NOTE: Uncomment these lines if you want to save the output! WARNING: The file may be large.
# jldopen("output_files/dual_output.jld", "w") do file
#     write(file, "theta_init", theta_init)
#     write(file, "z_init", z_init)
#     write(file, "n", N)
#     write(file, "lower_bound", lower_bound)
#     write(file, "nu_opt", nu_sol)
# end

# Approximately solve the primal problem
curr_z = copy(z_init)
curr_theta = copy(theta_init)
curr_nu = zeros(n_freq, N*N)
rho = 100
maxiter = 500
prev_conv_val = Inf

# Keeps the list of historic values
all_feas_tol = zeros(maxiter)
all_obj = zeros(maxiter)
all_design = zeros(maxiter, N*N)
all_field = zeros(maxiter, n_freq, N*N)

alpha = 1.
tau = .9

last_iteration = 0

# Generate an initial symbolic factorization for reuse.
F = cholesky(L_all[1]'*L_all[1] + spdiagm(0 => weights_all[1].^2))

for n = 1:maxiter
    global last_iteration, curr_z, curr_theta, curr_nu

    @time begin
        # Solve the field equations (appendix, section 8.1, z update)
        for i=1:n_freq
            curr_z[i,:] .= solve_max_eq!(curr_z[i,:], L_all[i], curr_theta, curr_nu[i,:], rho, weights_all[i], z_hat_all[i], b_all[i], F)
        end

        # Minimize over design (appendix, section 8.1, \theta update)
        curr_theta .= min_sq_diag([weights_all[i] .* curr_z[i,:] for i=1:n_freq], [weights_all[i] .* (b_all[i] - curr_nu[i,:] - L_all[i] * curr_z[i,:]) for i=1:n_freq], t_max, n_freq)

        # Update dual (appendix, section 8.1, \nu update)
        cons_val = 0
        for i=1:n_freq
            resid = L_all[i] * curr_z[i,:] + curr_z[i,:] .* curr_theta - b_all[i]
            curr_nu[i,:] .+= resid
            cons_val += norm(resid)^2
        end
    end
    all_feas_tol[n] = sqrt(cons_val)

    obj_val = sum(.5*norm(weights_all[i].*(curr_z[i,:] - z_hat_all[i]))^2 for i=1:n_freq)

    all_obj[n] = obj_val
    all_design[n, :] .= curr_theta
    all_field[n, :, :] .= curr_z

    last_iteration = n

    if cons_val < 1e-4
        @info "Breaking due to tolerance." 
        @show cons_val
        break
    end

    @show n
    @show cons_val
    @show obj_val

    if PLOT_ITERATES && ((n-1)%10 == 0)
        figure()
        title(L"$\theta$")
        imshow(reshape(curr_theta, N, N), cmap="Purples")
        savefig("iter_image/resonator_theta_n$(N)_iter_$(n).pdf", bbox_inches="tight")
        close()
    end

    prev_conv_val = cons_val
end

obj_val = sum(.5*norm(weights_all[i].*(curr_z[i,:] - z_hat_all[i]))^2 for i=1:n_freq)
@show obj_val
@show lower_bound
@show last_iteration

# NOTE: Uncomment these lines if you want to save the output! WARNING: The file may be large.
# jldopen("output_files/primal_output_dual_initialization.jld", "w") do file
#     write(file, "theta_opt", curr_theta)
#     write(file, "z_opt", curr_z)
#     write(file, "all_obj", all_obj)
#     write(file, "all_feas_tol", all_feas_tol)
#     write(file, "n", N)
#     write(file, "all_design", all_design)
#     write(file, "all_field", all_field)
# end


# Plotting
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
    imshow(reshape(z_hat_all[i], N, N), cmap="Purples")
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
