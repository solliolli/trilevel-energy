

nodes = ["Finland", "Sweden", "Norway", "Denmark", "Baltics"]
conv_sources = ["nuclear", "coal", "gas_cc", "gas_oc", "biomass"]
vres_sources =  ["solar", "wind_onshore", "wind_offshore", "hydro"]
sources = [conv_sources; vres_sources]

include("data_processing.jl")   # gives an instance called "ins"
include("functions.jl")         # contains the functions for building models

mpc_pdc = build_model_gurobi(ins, r=0.085, timelimit=7200)
optimize!(mpc_pdc)

# Extracting results
xval = value.(mpc_pdc[:x])
ρval = value.(mpc_pdc[:ρ])
yval = value.(mpc_pdc[:y])
wval = value.(mpc_pdc[:w])
zval = value.(mpc_pdc[:z])

prices = [ins.α[s,k]-ins.β[s,k]*(sum(zval[:,:,k,s])+yval[k,s]) for k in ins.K, s in ins.S]
@show prices-wval # Should be constant across nodes for each day
@show prices

# Line flows from PTDFs and y-values
for a in ins.A, s in ins.S
    f = sum(ins.PTDF[a,k]*yval[k,s] for k in ins.K)
    if abs(f)>1E-6
        println("line $a, scenario $s: flow between $(-ins.T⁻[a]) and $(ins.T⁺[a]): $f")
    end
end

ztot = [sum(zval[:,:,k,s]) for k in ins.K, s in ins.S]
renewableshare = [sum(zval[:,ins.R,k,s]) for k in ins.K, s in ins.S]./ztot
welfare_arr = [ins.α[s,k]*ztot[k,s] - (ins.β[s,k]/2)*ztot[k,s]^2  - sum((ins.ζ[i,j,k] + ins.η[j]*xval)*zval[i,j,k,s] for i in ins.F, j in ins.J) for k in ins.K, s in ins.S]
welfare = sum(ins.pr[s]*welfare_arr[k,s] for k in ins.K, s in ins.S)

@show [sum(ins.θ[k]*ztot[k,s] for k in ins.K) for s in ins.S]                       # Total value of production for each day
@show [sum(ins.η[j]*sum(zval[:,j,:,s]) for j in ins.J) for s in ins.S]              # Total emissions for each day
@show sum(ins.pr[s]*sum(ins.θ[k]*ztot[k,s] for k in ins.K) for s in ins.S)          # Daily average value of production
@show sum(ins.pr[s]*sum(ins.η[j]*sum(zval[:,j,:,s]) for j in ins.J) for s in ins.S) # Daily average emissions
@show [sum(ins.pr[s]*sum(zval[:,j,:,s]) for s in ins.S) for j in ins.J]             # Daily average production by fuel type
@show xval                                                                          # Tax
@show ρval                                                                          # Minimum renewable share decided by the regulator
@show [sum(zval[:,ins.R,:,s])/sum(ztot[:,s]) for s in ins.S]                        # Renewable share of production for each day



################
### PLOTTING ###
################

using Plots
# These are obtained above by varying the value of r
values_r = [169539, 170524, 190618]
emissions_r = [0, 86.17, 1835.5]

# These are Pareto optimal solutions obtained by varying the value of r or fixing the value of x
values = [169539, 170524, 174641, 181031, 185829, 190618]
emissions = [0, 86.17, 587.4, 1111.25, 1474.6, 1835.5]
xvals = [47.07; 19.21; 10; 5; 2.5; 0]

Plots.scatter(values./maximum(values), emissions./maximum(emissions), label=false, xlabel = "value of production", ylabel = "total emissions", markershape=:x, markerstrokewidth = 2)
plot!(values_r./maximum(values_r), emissions_r./maximum(emissions_r), label=false, color=:black, ls=:dash)

annotate!(0.902, 0.05, ("115 €/ton", 10))
annotate!(0.9125, -0.05, ("76 €/ton", 10))
annotate!(0.9175, 0.025, ("63 €/ton", 10))
annotate!(0.92, 0.25, ("50 €/ton", 10))
annotate!(0.9325, 0.4, ("40 €/ton", 10))
annotate!(0.95, 0.55, ("30 €/ton", 10))
annotate!(0.9625, 0.66, ("20 €/ton", 10))
annotate!(0.98, 0.87, ("10 €/ton", 10))
annotate!(1, 0.95, ("0 €/ton", 10))
Plots.pdf("casestudy_pareto")


