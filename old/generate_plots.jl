using JLD
using PyPlot

maxiter = 600
N = 251

primal_ab_initio = load("output_files/primal_output_ab_initio.jld")
primal_dual_init = load("output_files/primal_output_dual_initialization.jld")
dual_problem = load("output_files/dual_output.jld")

rc("text", usetex=true)
rc("font", family="serif")

# # Generate plots of convergence
# figure(figsize=(10, 3))
# subplot(1, 2, 1)
# title("Objective per iteration")
# plot(1:maxiter, primal_ab_initio["all_obj"], label="Simple initialization")
# plot(1:maxiter, primal_dual_init["all_obj"], label="Dual initialization")
# ylabel("Objective")
# xlabel("Iteration")
# legend()

# subplot(1, 2, 2)
# title("Feasibility residual")
# semilogy(1:maxiter, primal_ab_initio["all_feas_tol"], label="Simple initialization")
# semilogy(1:maxiter, primal_dual_init["all_feas_tol"], label="Dual initialization")
# ylabel(L"$\|(z^1, z^2, z^3)\|_2$")
# xlabel("Iteration")
# legend()

# savefig("objective_iteration_comparison.pdf", bbox_inches="tight")
# close()

# @show ab_initio_stop_idx = findfirst(primal_ab_initio["all_feas_tol"] .< 1e-2)
@show dual_init_stop_idx = findfirst(primal_dual_init["all_feas_tol"] .< 1e-2)

# Generate plots of final designs
figure(figsize=(7, 3))
# subplot(131)
# title("ADMM with simple initialization")
# imshow(reshape(primal_ab_initio["all_design"][ab_initio_stop_idx,:], N, N), cmap="Purples")
subplot(121)
title("Suggested initial design")
imshow(reshape(dual_problem["theta_init"], N, N), cmap="Purples")
axis("off")
subplot(122)
title("Locally-optimized design")
imshow(reshape(primal_dual_init["all_design"][dual_init_stop_idx,:], N, N), cmap="Purples")
axis("off")
tight_layout(pad=.5)
savefig("resonator_theta_initialization_comparison_toc.pdf")
close()

# # Print out values with stopping condition
# @show primal_ab_initio["all_obj"][ab_initio_stop_idx]
# @show primal_dual_init["all_obj"][dual_init_stop_idx]
# @show primal_ab_initio["all_feas_tol"][ab_initio_stop_idx]
# @show primal_dual_init["all_feas_tol"][dual_init_stop_idx]


# # Generate plots of final fields
# figure(figsize=(8,6))
# for i=1:3
#     subplot(2, 3, i)
#     title("\$(z^{$i})^0\$")
#     imshow(reshape(dual_problem["z_init"][i, :], N, N), cmap="Spectral")
# end
# for i=1:3
#     subplot(2, 3, 3+i)
#     title("\$z^{$i}\$")
#     imshow(reshape(primal_dual_init["all_field"][dual_init_stop_idx,i,:], N, N), cmap="Spectral")
# end

# savefig("resonator_z_initialization_comparison.pdf", bbox_inches="tight")
# close()
