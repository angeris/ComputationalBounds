using SparseArrays
using PyPlot
using LinearAlgebra
using JuMP
import Gurobi

function generate_lapl(n)
    L = SymTridiagonal(spdiagm(-1 => ones(n-1), 0 => -2*ones(n), 1 => ones(n-1)))
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

all_primal = zeros(n)
all_dual = zeros(n)
all_dual_2 = zeros(n)
all_w = linspace(10*pi, 100*pi, n)
N = 1001

for idx = 1:n
    w = all_w[idx]
    # w = 30*pi
    t_min = 1
    t_max = 2 - t_min

    L = SymTridiagonal((N*N)/(w*w) * generate_lapl(N) + t_min * spdiagm(0 => ones(N)))

    x = linspace(0, 1, N)
    sigma = .1
    # g = sin.(x*w).*exp.(-(x .- .5).^2 ./ (2*(sigma)^2))
    g = zeros(N)
    g[1] = 10
    g[end] = -10
    # g = sin.(x*w)
    # g[div(N-1,2):end] .= 0
    b = zeros(N)

    a_diag = zeros(N)
    a_diag[1] = 1
    a_diag[end] = 1
    K = spdiagm(0 => a_diag)
    f_max = 10*ones(N)


    # Dual problem ----------
    # Computation of dual
    m = Model(solver=Gurobi.GurobiSolver())

    @variable(m, nu[1:N])
    @variable(m, t[1:N])
    @variable(m, k[1:N])

    @objective(m, Max, -.5*sum(t) - nu' * b - .5 * k' * f_max)

    for i=1:N
        @constraint(m, norm([2*(a_diag[i] * g[i] - L[:,i]' * nu), t[i] - (a_diag[i] + k[i])]) <= t[i] + a_diag[i] + k[i])
        @constraint(m, norm([2*((t_max * nu[i]) - (a_diag[i] * g[i] - L[:,i]' * nu)), t[i] - (a_diag[i] + k[i])]) <= t[i] + a_diag[i] + k[i])
    end

    @constraint(m, k .>= 0)
    status = solve(m)
    all_dual_2[idx] = - .5*sum(getvalue(t)) - getvalue(nu)' * b - .5 * getvalue(k)' * f_max + .5*norm(g).^2
    @show all_dual_2

    nu_sol = getvalue(nu)
    t_sol = getvalue(t)

    theta_init = zeros(N)
    zero_ind = abs.(L' * nu_sol - g + nu_sol .* t_max) .< abs.(L' * nu_sol - g)
    theta_init[zero_ind] .= 0
    theta_init[.~zero_ind] .= t_max

    A_t_init = L + spdiagm(0 => theta_init)

    # m_new = Model(solver=Gurobi.GurobiSolver())
    #
    # @variable(m_new, l[1:N])
    # @variable(m_new, nu_new[1:N])
    # @variable(m_new, t_new[1:N])
    # @variable(m_new, q[1:N])
    #
    # @objective(m_new, Max, -.5*sum(t_new) - nu_new' * b - (t_max^2)*sum(l)/4 +.5*norm(K * g)^2)
    #
    # for i = 1:N
    #     s_cons = [ 2*(L[:,i]' * nu_new + .5*nu_new[i]*t_max - K[i, i] * g[i]),
    #                t_new[i] - (K[i, i] - .5*q[i]) ]
    #
    #     @constraint(m_new, norm(s_cons) <= t_new[i] + K[i, i] - .5*q[i])
    #     @constraint(m_new, norm([2*nu_new[i], l[i] - q[i]]) <= l[i] + q[i])
    #     @constraint(m_new, l[i] >= 0)
    # end
    #
    # status = solve(m_new)
    #
    # all_dual[idx] = -.5*sum(getvalue(t_new)) - getvalue(nu_new)' * b - (t_max^2)*sum(getvalue(l))/4 +.5*norm(K * g)^2
    # @show all_dual[idx]

    # End Dual problem ----------

    # Get initial points
    # P_main = [
    #              [.5*K^2 .5*spdiagm(0 => getvalue(nu))];
    #              [.5*spdiagm(0 => getvalue(nu)) spdiagm(0 => getvalue(l))]
    #          ]
    # q_main = [
    #             -K*g + L'*getvalue(nu);
    #             -t_max*getvalue(l)
    #          ]
    #
    # all_init = -.5*(P_main \ q_main)
    #
    # z_init = all_init[1:N]
    # theta_init = proj(all_init[N+1:end], 0, t_max)
    # theta_init = all_init[N+1:end]
    # @show typeof(theta_init)

    z_init = 1 ./ (a_diag .^ 2) .* (K*g - A_t_init' * nu_sol)

    figure(figsize=(12,8))
    subplot(211)
    title("New convex problem primal : $(floor(Int, all_w[idx]/pi))π")
    plot(z_init)
    plot(theta_init)
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
    # curr_theta = copy(theta_init)
    curr_theta = rand(N)
    curr_nu = zeros(N)
    rho = 100
    maxiter = 1000
    prev_conv_val = Inf

    alpha = 10
    tau = .9

    last_iteration = 0


    for i = 1:maxiter
        # Minimize over field
        A_theta = L + spdiagm(0 => curr_theta)

        curr_z .= Symmetric(K^2 + rho * A_theta' * A_theta) \ (K * g + rho * A_theta' * (b - curr_nu))

        # Minimize over design
        curr_theta .= min_sq_diag(curr_z, b - curr_nu - L * curr_z, t_max)

        # Update dual
        resid = L * curr_z + curr_z .* curr_theta - b
        curr_nu .+= resid

        obj_val = .5*norm(K*curr_z - g)^2
        cons_val = norm(resid)

        last_iteration = i

        if cons_val < 1e-4
            @show cons_val
            break
        end

        # if cons_val >= tau*prev_conv_val
        #     rho *= alpha
        # end

        prev_conv_val = cons_val
    end

    obj_val = .5*norm(K*curr_z - g)^2
    @show obj_val
    @show last_iteration

    @show all_dual_2[idx]/obj_val

    all_primal[idx] = obj_val

    subplot(212)
    title("Optimized structure : $(floor(Int, all_w[idx]/pi))π")
    plot(curr_z)
    plot(curr_theta)
    savefig("new_optimized_structure_1d_$(floor(Int, all_w[idx]/pi)).png")
    close()
end

# figure(figsize=(12, 8))
# title("Dual value vs. primal value")
# plot(floor.(Int, all_w./pi), all_dual, label="Dual value", color="blue")
# plot(floor.(Int, all_w./pi), all_dual_2, label="Dual new value", color="red")
# plot(floor.(Int, all_w./pi), all_primal, label="Primal value", linestyle=":", color="black")
# legend()
# xlabel("Frequency/Length")
# ylabel(L"$\|f - \hat f\|_2^2$")
# savefig("complete_output_plot_$(N)_new_optim.pdf")
# close()
