using JLD2

filepath = "\\fivenodedata_24h.jld2"

N_F = 5
N_J = 9
N_K = 5
F = 1:N_F
J = 1:N_J
R = 6:9
K = 1:N_K

# Getting the PTDF values requires some algebra

distances = [   0	 597 835  1248 556;
                597	 0	 261  726  896;
                835	 261 0	  667  934;
                1248 726 667  0    876;
                556	 896 934  876	0]

mat1 = zeros(6,6)
mat2 = zeros(6,5)
A = collect(1:6)
A_spec = [(1,2),(1,5),(2,3),(2,4),(2,5),(3,4)]
R_a = Dict(a => distances[a[1],a[2]] for a in A_spec) # The reactance values are directly proportional to the length of the transmission line, and the scaling is irrelevant
cycles = [(2,3,4,2), (1,2,5,1)]
for i in 1:4
    mat2[i,i] = 1
    for a in A
        if A_spec[a][1]==i
            mat1[i,a] = 1
        elseif A_spec[a][2]==i
            mat1[i,a] = -1
        end
    end
end
for (k,cycle) in enumerate(cycles)
    for idx in 1:length(cycle)-1
        if cycle[idx] < cycle[idx+1]
            a = findfirst(x -> x==(cycle[idx],cycle[idx+1]), A_spec)
            mat1[4+k,a] = R_a[(cycle[idx],cycle[idx+1])]
        elseif cycle[idx] > cycle[idx+1]
            a = findfirst(x -> x==(cycle[idx+1],cycle[idx]), A_spec)
            mat1[4+k,a] = -R_a[(cycle[idx+1],cycle[idx])]
        end
    end
end

PTDF = -inv(mat1)*mat2

emissions = [0 0.34 0.2 0.2 0.39 0 0 0 0]
efficiency = [0.34 0.466 0.615 0.395 0.448 1 1 1 1]
emissions_per_mwh_produced = emissions./efficiency
η = emissions_per_mwh_produced

electricity_production_2020 = [68800 169200 154500 28100 6400+5700+4200]*1E3 # https://en.wikipedia.org/wiki/List_of_countries_by_electricity_production
GDP_2020 = [270673 537610 362009 352243 31005+33478+55688]*1E6 # https://statisticstimes.com/economy/countries-by-gdp.php
θ = GDP_2020./electricity_production_2020/1000

t = 1
N_S = 3
S = 1:N_S
α = load((@__DIR__)*filepath, "id_intercept")[:,t,:]
β = load((@__DIR__)*filepath, "id_slope")[:,t,:]


z_max = zeros(N_F, N_J, N_K, N_S)
conv_installed_capacities = load((@__DIR__)*filepath, "conv_installed_capacities")
vres_installed_capacities = load((@__DIR__)*filepath, "vres_installed_capacities")
vres_availability_factor = load((@__DIR__)*filepath, "vres_availability_factor")
hydro_cap = load((@__DIR__)*filepath, "hydro_params")
for i in F
    for k in K
        for s in S
            for j in 1:5
                z_max[i,j,k,s] = conv_installed_capacities[k,i,j]
            end
            for j in 6:8
                z_max[i,j,k,s] = vres_installed_capacities[k,i,j-5]*vres_availability_factor[s,t,k,j-5]
            end
            z_max[i,9,k,s] = hydro_cap[s,t,k,i]
        end
    end
end


ζ = zeros(N_F, N_J, N_K)
ζ_rand = zeros(N_F, N_J, N_K)
conv_operational_costs = load((@__DIR__)*filepath, "conv_operational_costs")
for i in F
    for j in 1:5
        for k in K
            ζ[i,j,k] = conv_operational_costs[k,i,j] 
        end
    end
end


T⁺ = zeros(length(A))
T⁻ = zeros(length(A))
line_capacities = load((@__DIR__)*filepath, "line_capacities")
for k1 in K
    for k2 in K
        if k1<k2
            a = findfirst(x -> x==(k1,k2), A_spec)
            if a !== nothing
                T⁺[a] = line_capacities[k1,k2]
            end
        elseif k1>k2
            a = findfirst(x -> x==(k2,k1), A_spec)
            if a !== nothing
                T⁻[a] = line_capacities[k1,k2]
            end
        end
    end
end

mutable struct Instance
    N_F
    N_J
    N_K
    F
    J
    R
    K
    A
    PTDF
    η
    θ
    N_S
    S
    α
    β
    z_max
    ζ
    T⁺
    T⁻
    pr
end

ins = Instance(N_F,N_J,N_K,F,J,R,K,A,PTDF,η,θ,N_S,S,α,β,z_max,ζ,T⁺,T⁻,[0.4,0.32,0.28])

function random_instance(N_F, N_J, N_K, N_S)
    F = 1:N_F
    J = 1:N_J
    R = 1:rand(1:N_J)
    K = 1:N_K

    loc = []
    for k in K
        push!(loc, (rand(),rand()))
    end

    A_spec = []
    for i in 1:N_K, j in (i+1):N_K
        push!(A_spec, (i,j))
    end

    N_A = length(A_spec)
    A = collect(1:N_A)
    mat1 = zeros(N_A,N_A)
    mat2 = zeros(N_A,N_K)
    R_a = Dict(a => sqrt((loc[a[1]][1]-loc[a[2]][1])^2 + (loc[a[1]][2]-loc[a[2]][2])^2) for a in A_spec) # The reactance values are directly proportional to the length of the transmission line, and the scaling is irrelevant
    
    cycles = []
    for startnode in 1:(N_K-2), endnode in (startnode+2):N_K
        cycle = collect(startnode:endnode)
        cycle = [cycle; startnode]
        push!(cycles, cycle)
    end

    for i in 1:(N_K-1)
        mat2[i,i] = 1
        for a in A
            if A_spec[a][1]==i
                mat1[i,a] = 1
            elseif A_spec[a][2]==i
                mat1[i,a] = -1
            end
        end
    end
    for (k,cycle) in enumerate(cycles)
        for idx in 1:length(cycle)-1
            if cycle[idx] < cycle[idx+1]
                a = findfirst(x -> x==(cycle[idx],cycle[idx+1]), A_spec)
                mat1[N_K-1+k,a] = R_a[(cycle[idx],cycle[idx+1])]
            elseif cycle[idx] > cycle[idx+1]
                a = findfirst(x -> x==(cycle[idx+1],cycle[idx]), A_spec)
                mat1[N_K-1+k,a] = -R_a[(cycle[idx+1],cycle[idx])]
            end
        end
    end

    PTDF = -inv(mat1)*mat2

    η = zeros(N_J)
    θ = zeros(N_J)
    for j in J
        if !(j ∈ R)
            η[j] = rand()
        end
        θ[j] = rand()+1
    end

    S = 1:N_S
    pr = rand(N_S)
    pr = pr./sum(pr)

    α = fill(200,(N_S,N_K)).+(rand(Float64,(N_S, N_K)).-0.5)*40
    α_per_β = fill(15000,(N_S,N_K)).+(rand(Float64,(N_S, N_K)).-0.5)*20000
    β = α./α_per_β

    z_max = rand(Float64,(N_F,N_J,N_K,N_S))*(15000/(N_F*N_J))

    ζ = fill(50,(N_F,N_J,N_K)).+(rand(Float64,(N_F,N_J,N_K)).-0.5)*50

    T⁺ = fill(1000,N_A).+(rand(N_A).-0.5)*1000
    T⁻ = T⁺

    Instance(N_F,N_J,N_K,F,J,R,K,A,PTDF,η,θ,N_S,S,α,β,z_max,ζ,T⁺,T⁻,pr)
end
