using JuMP
import Gurobi

A = randn(2, 2)

m = Model(with_optimizer(Gurobi.Optimizer))

@variable(m, v[1:2])
@constraint(m, A*v ∈ SecondOrderCone())
@objective(m, Max, sum(v))
solve!(m)
