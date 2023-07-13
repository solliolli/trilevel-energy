
using JuMP, Ipopt, Gurobi


function build_model_gurobi(ins; start_value=nothing, r=0.01, xval=-1, timelimit=600)

    N_F = ins.N_F
    N_J = ins.N_J
    N_K = ins.N_K
    F = ins.F
    J = ins.J
    R = ins.R
    K = ins.K
    A = ins.A
    PTDF = ins.PTDF
    η = ins.η
    θ = ins.θ
    N_S = ins.N_S
    S = ins.S
    α = ins.α
    β = ins.β
    z_max = ins.z_max
    ζ = ins.ζ
    T⁺ = ins.T⁺
    T⁻ = ins.T⁻
    pr = ins.pr

    env = Gurobi.Env()

    # Duality gap constraint RHS
    τ = 0

    mpc_pdc = Model()
    optimizer = optimizer_with_attributes(
        () -> Gurobi.Optimizer(env)
    )
    set_optimizer(mpc_pdc, optimizer)
    # set_optimizer_attribute(mpc_pdc, "Presolve", 0)
    set_optimizer_attribute(mpc_pdc, "NonConvex", 2)
    set_optimizer_attribute(mpc_pdc, "TimeLimit", timelimit)
    set_optimizer_attribute(mpc_pdc, "FeasibilityTol", 1E-4)
    # set_optimizer_attribute(mpc_pdc, "MIPGap", 1E-6)
    # set_optimizer_attribute(mpc_pdc, "IntFeasTol", 1E-9)

    invβ = zeros(N_K,N_S) 
    for k in K
        for s in S
            invβ[k,s] = 1/β[s,k]
        end
    end
    dhat = [1/sum(invβ[k,s] for k in K) for s in S]
    d = zeros(N_K, N_S)
    for k in K
        for s in S
            d[k,s] = dhat[s]*invβ[k,s]
        end
    end
    L = zeros(N_K, N_K, N_S)
    for k1 in K, k2 in K
        for s in S
            L[k1,k2,s] = -dhat[s]*invβ[k1,s]*invβ[k2,s]
        end
    end
    for k in K
        for s in S
            L[k,k,s] = dhat[s]*invβ[k,s]*(sum(invβ[k,s] for k in K) - invβ[k,s])
        end
    end

    @variable(mpc_pdc, z[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpc_pdc, y[k in K, s in S])
    @variable(mpc_pdc, λ[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpc_pdc, r⁺[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpc_pdc, r⁻[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpc_pdc, λd[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpc_pdc, r⁺d[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpc_pdc, r⁻d[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpc_pdc, ϕ⁺[a in A, s in S] >= 0)
    @variable(mpc_pdc, ϕ⁻[a in A, s in S] >= 0)
    @variable(mpc_pdc, ψ[k in K, s in S] >= 0)
    @variable(mpc_pdc, ϵ >= 0)
    @variable(mpc_pdc, w[k in K, s in S])
    @variable(mpc_pdc, x>=0)
    @variable(mpc_pdc, 1>=ρ>=0)

    if start_value !== nothing
        allvars = all_variables(mpc_pdc)
        set_start_value.(allvars, start_value)
    end

    @objective(mpc_pdc, Max, sum(sum(pr[s]*(r*θ[k] - (1-r)*η[j])*z[i,j,k,s] for i in F, j in J, s in S) for k in K) - (x + ρ)*1E-3)

    @expression(mpc_pdc, Z[s in S], sum(z[i,j,k,s] for i in F, j in J, k in K))
    @expression(mpc_pdc, Zi[i in F, s in S], sum(z[i,j,k,s] for j in J, k in K))

    @expression(mpc_pdc, primal_obj[i in F, s in S], (-dhat[s]*Z[s] + sum((α[s,k]-w[k,s])*d[k,s] for k in K))*sum(z[i,j,k,s] for j in J, k in K) - sum((ζ[i,j,k]+x*η[j]-w[k,s])*z[i,j,k,s] for j in J, k in K))
    @expression(mpc_pdc, dual_obj[i in F, s in S], dhat[s]*Zi[i,s]^2 + sum(z_max[i,j,k,s]*λ[i,j,k,s] for j in J, k in K))
    @expression(mpc_pdc, ϵ_rhs, τ - sum(dual_obj[i,s]-primal_obj[i,s] for i in F, s in S))

    @expression(mpc_pdc, ϕ⁺_rhs[a in A, s in S], T⁺[a] - sum(PTDF[a,k]*y[k,s] for k in K))
    @expression(mpc_pdc, ϕ⁻_rhs[a in A, s in S], T⁻[a] + sum(PTDF[a,k]*y[k,s] for k in K))
    @expression(mpc_pdc, ψ_rhs[k in K, s in S], sum(((j ∈ R ? 1 : 0) - ρ)*z[i,j,k,s] for i in F, j in J))
    @expression(mpc_pdc, λ_rhs[i in F, j in J, k in K, s in S], z_max[i,j,k,s]*ϵ - z[i,j,k,s])
    @expression(mpc_pdc, z_rhs[i in F, j in J, k in K, s in S], (2*dhat[s]*(Z[s]+Zi[i,s]) - sum((α[s,k] - w[k,s])*d[k,s] for k in K) + (ζ[i,j,k] + x*η[j] - w[k,s]))*ϵ - dhat[s]*(Z[s]+Zi[i,s]) + λd[i,j,k,s] - ((j ∈ R ? 1 : 0) - ρ)*ψ[k,s])
    @expression(mpc_pdc, λd_rhs[i in F, j in J, k in K, s in S], z_max[i,j,k,s] - z[i,j,k,s])
    @expression(mpc_pdc, zd_rhs[i in F, j in J, k in K, s in S], dhat[s]*(Z[s]+Zi[i,s]) - sum((α[s,k] - w[k,s])*d[k,s] for k in K) + (ζ[i,j,k] + x*η[j] - w[k,s]) + λ[i,j,k,s] )

    @constraint(mpc_pdc, [a in A, s in S], ϕ⁺_rhs[a,s] >= 0)
    @constraint(mpc_pdc, [a in A, s in S], ϕ⁻_rhs[a,s] >= 0)
    @constraint(mpc_pdc, [k in K, s in S], ψ_rhs[k,s] >= 0)
    @constraint(mpc_pdc, [i in F, j in J, k in K, s in S], λ_rhs[i,j,k,s] >= 0)
    @constraint(mpc_pdc, [i in F, j in J, k in K, s in S], z_rhs[i,j,k,s] >= 0)
    @constraint(mpc_pdc, [i in F, j in J, k in K, s in S], λd_rhs[i,j,k,s] >= 0)
    @constraint(mpc_pdc, [i in F, j in J, k in K, s in S], zd_rhs[i,j,k,s] >= 0)
    @constraint(mpc_pdc, ϵ_rhs >= 0)

    @constraint(mpc_pdc, [a in A, s in S], [ϕ⁺[a,s], ϕ⁺_rhs[a,s]] in SOS1())
    @constraint(mpc_pdc, [a in A, s in S], [ϕ⁻[a,s], ϕ⁻_rhs[a,s]] in SOS1())
    @constraint(mpc_pdc, [k in K, s in S], [ψ[k,s], ψ_rhs[k,s]] in SOS1())
    @constraint(mpc_pdc, [i in F, j in J, k in K, s in S], [z[i,j,k,s], z_rhs[i,j,k,s]] in SOS1())
    @constraint(mpc_pdc, [i in F, j in J, k in K, s in S], [λ[i,j,k,s], λ_rhs[i,j,k,s]] in SOS1())
    @constraint(mpc_pdc, [i in F, j in J, k in K, s in S], [z[i,j,k,s], zd_rhs[i,j,k,s]] in SOS1())
    @constraint(mpc_pdc, [i in F, j in J, k in K, s in S], [λd[i,j,k,s], λd_rhs[i,j,k,s]] in SOS1())
    @constraint(mpc_pdc, [ϵ, ϵ_rhs] in SOS1())

    @constraint(mpc_pdc, wheelingfee[k in K, s in S], w[k,s] == sum(PTDF[a,k]*(ϕ⁺[a,s]-ϕ⁻[a,s]) for a in A))

    @constraint(mpc_pdc, marketclearing[k in K, s in S], d[k,s]*Z[s] - sum(z[i,j,k,s] for i in F, j in J) + sum((α[s,k1]-w[k1,s])*L[k,k1,s] for k1 in K) == y[k,s])
    if xval>-1E-9
        @constraint(mpc_pdc, x == xval)
    end

    return mpc_pdc
end


function build_model_gurobi_Gabriel_et_al(ins; start_value=nothing, r=0.01)

    N_F = ins.N_F
    N_J = ins.N_J
    N_K = ins.N_K
    F = ins.F
    J = ins.J
    R = ins.R
    K = ins.K
    A = ins.A
    PTDF = ins.PTDF
    η = ins.η
    θ = ins.θ
    N_S = ins.N_S
    S = ins.S
    α = ins.α
    β = ins.β
    z_max = ins.z_max
    ζ = ins.ζ
    T⁺ = ins.T⁺
    T⁻ = ins.T⁻
    pr = ins.pr

    env = Gurobi.Env()

    mpcc = Model()
    optimizer = optimizer_with_attributes(
        () -> Gurobi.Optimizer(env)
    )
    set_optimizer(mpcc, optimizer)
    # set_optimizer_attribute(mpcc, "Presolve", 0)
    set_optimizer_attribute(mpcc, "NonConvex", 2)
    set_optimizer_attribute(mpcc, "FeasibilityTol", 1E-4)
    set_optimizer_attribute(mpcc, "TimeLimit", 600)
    # set_optimizer_attribute(mpcc, "MIPGap", 1E-6)
    # set_optimizer_attribute(mpcc, "IntFeasTol", 1E-9)

    invβ = zeros(N_K,N_S) 
    for k in K
        for s in S
            invβ[k,s] = 1/β[s,k]
        end
    end
    dhat = [1/sum(invβ[k,s] for k in K) for s in S]
    d = zeros(N_K, N_S)
    for k in K
        for s in S
            d[k,s] = dhat[s]*invβ[k,s]
        end
    end
    L = zeros(N_K, N_K, N_S)
    for k1 in K, k2 in K
        for s in S
            L[k1,k2,s] = -dhat[s]*invβ[k1,s]*invβ[k2,s]
        end
    end
    for k in K
        for s in S
            L[k,k,s] = dhat[s]*invβ[k,s]*(sum(invβ[k,s] for k in K) - invβ[k,s])
        end
    end

    @variable(mpcc, z[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpcc, zbar[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpcc, β_z[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpcc, δ_z[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpcc, λ[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpcc, λbar[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpcc, β_λ[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpcc, δ_λ[i in F, j in J, k in K, s in S] >= 0)
    @variable(mpcc, ζvar_z)
    @variable(mpcc, η_z[i in F, j in J, k in K, s in S])
    @variable(mpcc, ζvar_λ)
    @variable(mpcc, η_λ[i in F, j in J, k in K, s in S])
    @variable(mpcc, y[k in K, s in S])
    @variable(mpcc, ϕ⁺[a in A, s in S] >= 0)
    @variable(mpcc, ϕ⁻[a in A, s in S] >= 0)
    @variable(mpcc, ψ[k in K, s in S] >= 0)
    @variable(mpcc, w[k in K, s in S])
    @variable(mpcc, x>=0)
    @variable(mpcc, 1>=ρ>=0)

    if start_value !== nothing
        allvars = all_variables(mpcc)
        set_start_value.(allvars, start_value)
    end

    @objective(mpcc, Max, sum(sum(pr[s]*(r*θ[k] - (1-r)*η[j])*z[i,j,k,s] for i in F, j in J, s in S) for k in K) - (x + ρ)*1E-3)

    @expression(mpcc, Z[s in S], sum(z[i,j,k,s] for i in F, j in J, k in K))
    @expression(mpcc, Zi[i in F, s in S], sum(z[i,j,k,s] for j in J, k in K))
    @expression(mpcc, B[i in F, j in J, k in K, s in S, i1 in F, j1 in J, k1 in K, s1 in S], s==s1 ? (i==i1 ? 2*dhat[s] : dhat[s]) : 0)
    @expression(mpcc, q_z[i in F, j in J, k in K, s in S], - sum( (α[s,k1]-w[k1,s])*d[k1,s] for k1 in K) + ζ[i,j,k] + x*η[j] - w[k,s])
    @expression(mpcc, q_λ[i in F, j in J, k in K, s in S], z_max[i,j,k,s])

    @expression(mpcc, ϕ⁺_rhs[a in A, s in S], T⁺[a] - sum(PTDF[a,k]*y[k,s] for k in K))
    @expression(mpcc, ϕ⁻_rhs[a in A, s in S], T⁻[a] + sum(PTDF[a,k]*y[k,s] for k in K))
    @expression(mpcc, ψ_rhs[k in K, s in S], sum(((j ∈ R ? 1 : 0) - ρ)*z[i,j,k,s] for i in F, j in J))

    @expression(mpcc, zbar_rhs[i in F, j in J, k in K, s in S], q_z[i,j,k,s] + 2*sum(B[i,j,k,s,i1,j1,k1,s1] * zbar[i1,j1,k1,s1] for i1 in F, j1 in J, k1 in K, s1 in S) - (sum(B[i,j,k,s,i1,j1,k1,s1] * β_z[i1,j1,k1,s1] for i1 in F, j1 in J, k1 in K, s1 in S) - β_λ[i,j,k,s]))
    @expression(mpcc, λbar_rhs[i in F, j in J, k in K, s in S], q_λ[i,j,k,s] - β_z[i,j,k,s])

    @expression(mpcc, β_z_rhs[i in F, j in J, k in K, s in S], q_z[i,j,k,s] + sum(B[i,j,k,s,i1,j1,k1,s1] * zbar[i1,j1,k1,s1] for i1 in F, j1 in J, k1 in K, s1 in S) + λbar[i,j,k,s])
    @expression(mpcc, β_λ_rhs[i in F, j in J, k in K, s in S], q_λ[i,j,k,s] - zbar[i,j,k,s])

    @expression(mpcc, z_rhs[i in F, j in J, k in K, s in S], -(sum(B[i,j,k,s,i1,j1,k1,s1] * δ_z[i1,j1,k1,s1] for i1 in F, j1 in J, k1 in K, s1 in S) - δ_λ[i,j,k,s]) - ζvar_z*q_z[i,j,k,s] - 2*sum(B[i,j,k,s,i1,j1,k1,s1] * η_z[i1,j1,k1,s1] for i1 in F, j1 in J, k1 in K, s1 in S))
    @expression(mpcc, λ_rhs[i in F, j in J, k in K, s in S], -δ_z[i,j,k,s] - ζvar_λ*q_λ[i,j,k,s])

    @expression(mpcc, δ_z_rhs[i in F, j in J, k in K, s in S], q_z[i,j,k,s] + sum(B[i,j,k,s,i1,j1,k1,s1] * z[i1,j1,k1,s1] for i1 in F, j1 in J, k1 in K, s1 in S) + λ[i,j,k,s])
    @expression(mpcc, δ_λ_rhs[i in F, j in J, k in K, s in S], q_λ[i,j,k,s] - z[i,j,k,s])



    @constraint(mpcc, [a in A, s in S], ϕ⁺_rhs[a,s] >= 0)
    @constraint(mpcc, [a in A, s in S], ϕ⁻_rhs[a,s] >= 0)
    @constraint(mpcc, [k in K, s in S], ψ_rhs[k,s] >= 0)
    @constraint(mpcc, [i in F, j in J, k in K, s in S], z_rhs[i,j,k,s] >= 0)
    @constraint(mpcc, [i in F, j in J, k in K, s in S], zbar_rhs[i,j,k,s] >= 0)
    @constraint(mpcc, [i in F, j in J, k in K, s in S], β_z_rhs[i,j,k,s] >= 0)
    @constraint(mpcc, [i in F, j in J, k in K, s in S], δ_z_rhs[i,j,k,s] >= 0)
    @constraint(mpcc, [i in F, j in J, k in K, s in S], λ_rhs[i,j,k,s] >= 0)
    @constraint(mpcc, [i in F, j in J, k in K, s in S], λbar_rhs[i,j,k,s] >= 0)
    @constraint(mpcc, [i in F, j in J, k in K, s in S], β_λ_rhs[i,j,k,s] >= 0)
    @constraint(mpcc, [i in F, j in J, k in K, s in S], δ_λ_rhs[i,j,k,s] >= 0)



    @constraint(mpcc, [a in A, s in S], [ϕ⁺[a,s], ϕ⁺_rhs[a,s]] in SOS1())
    @constraint(mpcc, [a in A, s in S], [ϕ⁻[a,s], ϕ⁻_rhs[a,s]] in SOS1())
    @constraint(mpcc, [k in K, s in S], [ψ[k,s], ψ_rhs[k,s]] in SOS1())
    @constraint(mpcc, [i in F, j in J, k in K, s in S], [z[i,j,k,s], z_rhs[i,j,k,s]] in SOS1())
    @constraint(mpcc, [i in F, j in J, k in K, s in S], [zbar[i,j,k,s], zbar_rhs[i,j,k,s]] in SOS1())
    # @constraint(mpcc, [i in F, j in J, k in K, s in S], [zbar[i,j,k,s], β_z_rhs[i,j,k,s]] in SOS1())
    @constraint(mpcc, [i in F, j in J, k in K, s in S], [β_z[i,j,k,s], β_z_rhs[i,j,k,s]] in SOS1())
    @constraint(mpcc, [i in F, j in J, k in K, s in S], [δ_z[i,j,k,s], δ_z_rhs[i,j,k,s]] in SOS1())
    @constraint(mpcc, [i in F, j in J, k in K, s in S], [λ[i,j,k,s], λ_rhs[i,j,k,s]] in SOS1())
    @constraint(mpcc, [i in F, j in J, k in K, s in S], [λbar[i,j,k,s], λbar_rhs[i,j,k,s]] in SOS1())
    # @constraint(mpcc, [i in F, j in J, k in K, s in S], [λbar[i,j,k,s], β_λ_rhs[i,j,k,s]] in SOS1())
    @constraint(mpcc, [i in F, j in J, k in K, s in S], [β_λ[i,j,k,s], β_λ_rhs[i,j,k,s]] in SOS1())
    @constraint(mpcc, [i in F, j in J, k in K, s in S], [δ_λ[i,j,k,s], δ_λ_rhs[i,j,k,s]] in SOS1())

    @constraint(mpcc, sum(q_z[i,j,k,s]*(z[i,j,k,s]-zbar[i,j,k,s]) for i in F, j in J, k in K, s in S) == 0)
    @constraint(mpcc, sum(q_λ[i,j,k,s]*(λ[i,j,k,s]-λbar[i,j,k,s]) for i in F, j in J, k in K, s in S) == 0)
    @constraint(mpcc, [i in F, j in J, k in K, s in S], sum(2*B[i,j,k,s,i1,j1,k1,s1] * (z[i1,j1,k1,s1] - zbar[i1,j1,k1,s1]) for i1 in F, j1 in J, k1 in K, s1 in S) == 0)

    @constraint(mpcc, wheelingfee[k in K, s in S], w[k,s] == sum(PTDF[a,k]*(ϕ⁺[a,s]-ϕ⁻[a,s]) for a in A))

    @constraint(mpcc, marketclearing[k in K, s in S], d[k,s]*Z[s] - sum(z[i,j,k,s] for i in F, j in J) + sum((α[s,k1]-w[k1,s])*L[k,k1,s] for k1 in K) == y[k,s])

    # @constraint(mpcc, x==0)
    
    return mpcc
end