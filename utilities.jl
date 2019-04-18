using LinearAlgebra
using SparseArrays

"Generates a laplacian with Dirichlet BCs"
function generate_lapl(n)
    L = spdiagm(-1 => ones(n-1), 0 => -2*ones(n), 1 => ones(n-1))
    return L
end

"Idk, this seemed easier (I've now somewhat learned the new ways)"
function linspace(s, e, n)
    return collect(range(s, e, length=n))
end

"Sets a rectangle with `top_left` and `bottom_right` coordinates in an array to value `value`"
function filter_square!(a, top_left, bottom_right, to_linear, value)
    for J in CartesianIndices(bottom_right - top_left)
        a[to_linear[top_left + J]] = value
    end
end

"""
Minimizes  ∑ᵢ ‖diag(v_i)*θ - h_i‖², s.t. 0 ≤ θ ≤ u over θ.

Corresponds to the multi-mode version of the θ update in section 8.1 in the appendix.
"""
function min_sq_diag(v, h, u, n_freq; tol=1e-14)
    w_comp = sum(v[i] .* h[i] for i=1:n_freq)
    inv_comp = sum(v[i].^2 for i=1:n_freq)
    return clamp.(w_comp ./ inv_comp, 0, u)
end

# TODO: This is slow and sad. A CG method (or saving and reusing the symbolic factorization) would
# make it much faster.
"""
Minimizes ‖W(z - z_hat)‖² + ρ‖(A + diag(θ))z - b + ν‖² over z.
"""
function solve_max_eq(A, θ, ν, ρ, W, z_hat, b)
    A_θ = A + spdiagm(0 => θ)
    return Symmetric(Diagonal(W.^2) + ρ * A_θ' * A_θ) \ (W .* z_hat + ρ * A_θ' * (b - ν))
end

# Formulates ‖x‖² ≤ y as an SOC
function quad_cons(m, x, y)
    @constraint(m, sum(x.^2) ≤ y )
end
