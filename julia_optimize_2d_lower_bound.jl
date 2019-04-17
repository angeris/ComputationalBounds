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

# Compute the dual function

n = 10

N = 201

nw = 50

w = nw*pi

# w = 30*pi
t_min = 1
t_max = 2 - t_min

L_1d = (N*N)/(w*w) * generate_lapl(N)
L = kron(L_1d, sparse(I, N, N)) + kron(sparse(I, N, N), L_1d)

x = linspace(0, 1, N)
y = linspace(0, 1, N)

mosaic(x::Float64, y::Float64; sigma=.2) = sin(x*w)*sin(y*w)*exp(-(x .- .5).^2 ./ (2*(sigma)^2))*exp.(-(y .- .5).^2 ./ (2*(sigma)^2))

b = zeros(N*N)
g = [mosaic(x, y) for x in x for y in y]

a_diag = ones(N*N)

K = spdiagm(0 => a_diag)


# Dual problem ----------
# Computation of dual
m = Model(solver=Gurobi.GurobiSolver())

@variable(m, nu[1:N*N])
@variable(m, t[1:N*N])

@objective(m, Max, -.5*sum(t .* t) - nu' * b)

@constraint(m, t .>= g - L' * nu)
@constraint(m, t .>= L' * nu - g)
@constraint(m, t .>= (t_max * nu) - (g - L' * nu))
@constraint(m, t .>= (g - L' * nu) - (t_max * nu))

status = solve(m)

nu_sol = getvalue(nu)
t_sol = getvalue(t)

t_init = zeros(N*N)
zero_ind = abs.(L' * nu_sol - g + nu_sol .* t_max) .< abs.(L' * nu_sol - g)
t_init[zero_ind] .= 0
t_init[.~zero_ind] .= t_max

# Solve for new field given constraints
A_t_init = L + spdiagm(0 => t_init)
A_ldlt = ldlt(Symmetric([K^2 A_t_init'; A_t_init spzeros(N*N, N*N)]))
z_init = A_ldlt \ [A_t_init'*g; b]

# z_init = g - L' * nu_sol - t_init .* nu_sol

@show -.5*sum(t_sol .* t_sol) - nu_sol' * b + .5*norm(g)^2

figure()
title("Non-optimized primal")
imshow(reshape(t_init, N, N), cmap="Blues")
savefig("primal_integral_value_2d_gaussian_N$(N)_nw$nw.png")
show()

figure()
title("Non-optimized primal field")
imshow(reshape(z_init, N, N), cmap="Purples")
savefig("primal_integral_value_2d_gaussian_field_N$(N)_nw$nw.png")
show()


# End Dual problem ----------

m_new = Model(solver=Gurobi.GurobiSolver())

@variable(m_new, l[1:N*N])
@variable(m_new, nu_new[1:N*N])
@variable(m_new, t_new[1:N*N])
@variable(m_new, q[1:N*N])

@objective(m_new, Max, -.5*sum(t_new) - nu_new' * b - (t_max^2)*sum(l)/4 +.5*norm(K * g)^2)

for i = 1:N*N
    s_cons = [ 2*(L[:,i]' * nu_new + .5*nu_new[i]*t_max - K[i, i] * g[i]),
               t_new[i] - (K[i, i] - .5*q[i]) ]

    @constraint(m_new, norm(s_cons) <= t_new[i] + K[i, i] - .5*q[i])
    @constraint(m_new, norm([2*nu_new[i], l[i] - q[i]]) <= l[i] + q[i])
    @constraint(m_new, l[i] >= 0)
end

status = solve(m_new)

# Get initial points
P_main = [
             [.5*K^2 .5*spdiagm(0 => getvalue(nu))];
             [.5*spdiagm(0 => getvalue(nu)) spdiagm(0 => getvalue(l))]
         ]
q_main = [
            -K*g + L'*getvalue(nu);
            -t_max*getvalue(l)
         ]
#
all_init = -.5*(P_main \ q_main)

z_init_smooth = all_init[1:N*N]
t_init_smooth = proj(all_init[N*N+1:end], 0., t_max)

figure()
title("Non-optimized, smooth primal")
imshow(reshape(t_init_smooth, N, N), cmap="Blues")
savefig("primal_smooth_integral_value_2d_gaussian_N$(N)_nw$nw.png")
show()

figure()
title("Non-optimized, smooth primal field")
imshow(reshape(z_init_smooth, N, N), cmap="Purples")
savefig("primal_smooth_integral_value_2d_gaussian_field_N$(N)_nw$nw.png")
show()

primal_value = -.5*sum(getvalue(t_new)) - getvalue(nu_new)' * b - (t_max^2)*sum(getvalue(l))/4 +.5*norm(K * g)^2

@show primal_value

#
# z_init = all_init[1:N]
# theta_init = proj(all_init[N+1:end], 0, t_max)
# @show typeof(theta_init)
#
# # figure(figsize=(12,8))
# # subplot(211)
# # title("Convex problem primal : $(floor(Int, all_w[idx]/pi))π")
# # plot(z_init)
# # plot(theta_init)
#
# # Primal solver
# # m_primal = Model(solver=IpoptSolver())
# #
# # @variable(m_primal, z[i=1:N], start=z_init[i])
# # @variable(m_primal, 0 <= theta[i=1:N] <= t_max, start=theta_init[i])
# #
# # for i = 1:N
# #     @NLconstraint(m_primal, sum(L[i,j] * z[j] for j=1:N if L[i,j] != 0) + theta[i] * z[i] == b[i])
# # end
# # @NLobjective(m_primal, Min, sum((z[i] - g[i])^2 for i=1:N))
# #
# # status = solve(m_primal)
# # println("Final solution $(.5*norm(getvalue(z) - g)^2)")
# # all_primal[idx] = .5*norm(getvalue(z) - g)^2
#
# New primal solver
curr_z = copy(z_init)
curr_theta = copy(t_init)
curr_nu = zeros(N*N)
rho = 100
maxiter = 1000
prev_conv_val = Inf

alpha = 10
tau = .9

last_iteration = 0

for i = 1:maxiter
    global last_iteration
    # Minimize over field
    A_theta = L + spdiagm(0 => curr_theta)

    curr_z .= Symmetric(K^2 + rho * A_theta' * A_theta) \ (K * g + rho * A_theta' * (b - curr_nu))

    # Minimize over design
    curr_theta .= min_sq_diag(curr_z, b - curr_nu - L * curr_z, t_max)

    # Update dual
    resid = L * curr_z + curr_z .* curr_theta - b
    curr_nu .+= resid

    obj_val = .5*norm(curr_z - g)^2
    cons_val = norm(resid)

    last_iteration = i

    if cons_val < 1e-4
        @show cons_val
        break
    end

    @show i
    @show cons_val

    # if cons_val >= tau*prev_conv_val
    #     rho *= alpha
    # end

    prev_conv_val = cons_val
end

obj_val = .5*norm(curr_z - g)^2
@show obj_val
@show last_iteration

figure()
title("Optimized structure")
imshow(reshape(curr_theta, N, N), cmap="Blues")
savefig("optimized_integral_value_2d_gaussian_N$(N)_nw$nw.png")
show()

figure()
title("Optimized structure field")
imshow(reshape(curr_z, N, N), cmap="Purples")
savefig("optimized_2d_gaussian_field_N$(N)_nw$nw.png")
show()


# subplot(212)
# title("Optimized structure : $(floor(Int, all_w[idx]/pi))π")
# plot(curr_z)
# plot(curr_theta)
# savefig("optimized_structure_$(floor(Int, all_w[idx]/pi)).png")
# close()
