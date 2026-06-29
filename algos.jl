using Distributions
using Optim
using LinearAlgebra
using Statistics
include("functions.jl")

function algo1(d1_full::Vector{T}, d2_full::Vector{T}, d3_full::Vector{T},
    Li_full::Vector{Float64}, Ri_full::Vector{Float64}, Xp_full::Matrix{Float64},
    n_int::Int, order::Int, g0::Vector{Float64}, b0::Vector{Float64}, q::Float64;
    V_dist::String="gamma", tol::Float64=5e-3, maxit::Int=1000,
    t_seq::Union{Vector{Float64}, Nothing}=nothing, equal::Bool=false,
    rng_seed::Union{Int, Nothing}=nothing) where {T<:Union{Int, Float64}}

    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end

    P, L = length(b0), length(g0)
    N = length(d1_full)

    if t_seq === nothing
        t_seq = range(0, maximum(vcat(Li_full, Ri_full)), length=50)
    end

    ind_sub = rand(Bernoulli(q), N) .== 1
    d1, d2, d3 = d1_full[ind_sub], d2_full[ind_sub], d3_full[ind_sub]
    Li, Ri = Li_full[ind_sub], Ri_full[ind_sub]
    Xp = Xp_full[ind_sub, :]
    n = length(d1)

    Li[d1 .== 1] .= Ri[d1 .== 1]
    Ri[d3 .== 1] .= Li[d3 .== 1]

    ti = vcat(Li[d1 .== 0], Ri[d3 .== 0])
    ti_min, ti_max = minimum(ti)-1e-5, maximum(ti)+1e-5

    knots = equal ?
        range(ti_min, ti_max, length=n_int+2) :
        vcat(ti_min,
             quantile(ti, collect(range(0,1,length=n_int+2))[2:end-1]),
             ti_max)

    bRi = Ispline(Ri, order, knots)
    bLi = Ispline(Li, order, knots)
    bt  = Ispline(collect(t_seq), order, knots)

    V = V_dist == "exp"   ? rand(Exponential(1/q), n) :
        V_dist == "gamma" ? rand(Gamma(1/q, 1.0), n) :
        V_dist == "beta"  ? 3/q .* rand(Beta(1,2), n) :
        V_dist == "Geom"  ? 1.0 .+ rand(Geometric(q), n) :
        error("Unknown V_dist")

    buffers = setup_buffers(n, L, P)

    b1, g1 = copy(b0), copy(g0)
    b0_old, g0_old = similar(b1), similar(g1)

    dif, iter = 1.0, 1

    while dif > tol && iter <= maxit
        b0_old .= b1
        g0_old .= g1

        mul!(buffers.GRi, bRi, g0_old)
        mul!(buffers.GLi, bLi, g0_old)
        mul!(buffers.xb0, Xp, b0_old)

        # === 正确数值计算（修复版）===
        @inbounds @simd for i in 1:n
            xb = buffers.xb0[i]
            exp_xb = exp(xb)
            buffers.exp_xb0[i] = exp_xb

            GR = buffers.GRi[i] * exp_xb
            GL = buffers.GLi[i] * exp_xb

            buffers.GRi_exp[i] = GR
            buffers.GLi_exp[i] = GL

            # dz
            if d1[i] == 1
                buffers.dz[i] = 1 - exp(-GR)
            else
                buffers.dz[i] = 1.0
            end

            # dw（关键修复）
            if d2[i] == 1
                buffers.dw[i] = 1 - exp(-(GR - GL))
            else
                buffers.dw[i] = 1.0
            end
        end

        fill!(buffers.EZil, 0.0)
        fill!(buffers.EWil, 0.0)

        @inbounds for i in 1:n
            if d3[i] != 1
                exp_xb = buffers.exp_xb0[i]

                if d1[i] == 1
                    factor = exp_xb / buffers.dz[i]
                    @simd for l in 1:L
                        buffers.EZil[i,l] = factor * bRi[i,l] * g0_old[l]
                    end
                else
                    factor = exp_xb / buffers.dw[i]
                    @simd for l in 1:L
                        buffers.EWil[i,l] = factor *
                            (bRi[i,l] - bLi[i,l]) * g0_old[l]
                    end
                end
            end
        end

        objective(b) = Q2_V_fast(b, buffers.EZil, buffers.EWil,
                           Xp, bRi, bLi, d1, d2, d3, g0_old, V)

        result = optimize(objective, b0_old, LBFGS())
        b1 = Optim.minimizer(result)

        g1 = Q1_V(b1, buffers.EZil, buffers.EWil,
                  Xp, bRi, bLi, d1, d2, d3, V)

        dif = compute_difference(b0_old, g0_old, b1, g1)
        iter += 1
    end

    return (b=b1, g=g1, iter=iter)
end




function algo2(d1_full::Vector{T}, d2_full::Vector{T}, d3_full::Vector{T},
                         Li_full::Vector{Float64}, Ri_full::Vector{Float64}, Xp_full::Matrix{Float64},
                         n_int::Int, order::Int, g0::Vector{Float64}, b0::Vector{Float64}, q::Float64, M::Int; 
                         V_dist::String="gamma", tol::Float64=5e-3, maxit::Int=1000,
                         t_seq::Union{Vector{Float64}, Nothing}=nothing, equal::Bool=false,
                         rng_seed::Union{Int, Nothing}=nothing) where {T<:Union{Int, Float64}}
    
    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end
    
    P = length(b0)
    
    # if t_seq === nothing
    #     t_seq = range(0, maximum(vcat(Li_full, Ri_full)), length=50)
    # end

    d = nothing
    if V_dist == "exp"
        d = 1 / q
    elseif V_dist == "gamma"
        d = 1 / sqrt(q)
    elseif V_dist == "beta"
        d = 1 / q / sqrt(2)
    elseif V_dist == "Geom"
        d = sqrt(1 - q) / q
    else
        error("Unknown V_dist: $V_dist")
    end

    Bs = zeros(M, P)

    for m in 1:M
        Bs[m, :] .= algo1(d1_full, d2_full, d3_full, Li_full, Ri_full, Xp_full, 
                     n_int, order, g0, b0, q; V_dist = V_dist, maxit = maxit).b
    end

    b_hat = mean(Bs, dims=1)
    
    # cov_Bs = cov(Bs)
    cov_Bs = manual_cov(Bs, b_hat, M)
    cov_cond = cov_Bs ./ M
    cov_uncond = (M / (1/q - 1 + d^2*q) + 1) * cov_cond
    
    return (b = b_hat, cov_cond = cov_cond, cov_uncond = cov_uncond, Bs = Bs)
end
