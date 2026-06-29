using Optim
using LinearAlgebra
using Statistics
using StatsBase

include("/home/wangsong/PSICsurv/code/functions.jl")

# function Ispline(x::Vector{Float64}, order::Int, knots::Vector{Float64})
#     k = order + 1
#     m = length(knots)
#     n = m - 2 + k
    
#     t = vcat(fill(knots[1], k), knots[2:(m-1)], fill(knots[m], k))
    
#     length_x = length(x)
#     yy1 = zeros(n + k - 1, length_x)
    
#     for l in k:n
#         for j in 1:length_x
#             if t[l] <= x[j] < t[l+1]
#                 yy1[l, j] = 1.0 / (t[l+1] - t[l])
#             end
#         end
#     end
    
#     yytem1 = yy1
#     yytem2 = nothing
#     for ii in 1:order
#         yytem2 = zeros(n + k - 1 - ii, length_x)
#         for i in (k - ii):n
#             for j in 1:length_x
#                 denom = t[i + ii + 1] - t[i]
#                 if denom > 0
#                     yytem2[i, j] = (ii + 1) * ((x[j] - t[i]) * yytem1[i, j] + 
#                                                 (t[i + ii + 1] - x[j]) * yytem1[i + 1, j]) / 
#                                                 (denom * ii)
#                 end
#             end
#         end
#         yytem1 = yytem2
#     end
    
#     index = zeros(Int, length_x)
#     for i in 1:length_x
#         idx = findlast(t .<= x[i])
#         index[i] = isnothing(idx) ? 0 : idx
#     end
    
#     yy = zeros(n - 1, length_x)  
    
#     if order == 1
#         for i in 2:n
#             for j in 1:length_x
#                 if i < index[j] - order + 1
#                     yy[i-1, j] = 1.0
#                 elseif i == index[j]
#                     denom = t[i + order + 1] - t[i]
#                     if denom > 0
#                         yy[i-1, j] = denom * yytem2[i, j] / (order + 1)
#                     end
#                 end
#             end
#         end
#     else
#         for j in 1:length_x
#             for i in 2:n
#                 if i < index[j] - order + 1
#                     yy[i-1, j] = 1.0
#                 elseif i <= index[j] && i >= index[j] - order + 1
#                     sum_val = 0.0
#                     for kk in i:index[j]
#                         denom = t[kk + order + 1] - t[kk]
#                         if denom > 0
#                             sum_val += denom * yytem2[kk, j]
#                         end
#                     end
#                     yy[i-1, j] = sum_val / (order + 1)
#                 else
#                     yy[i-1, j] = 0.0
#                 end
#             end
#         end
#     end
    
#     return Matrix(yy')  
# end

function Q1(b1::Vector{Float64}, EZil::Matrix{Float64}, EWil::Matrix{Float64},
                        Xp::Matrix{Float64}, bRi::Matrix{Float64}, bLi::Matrix{Float64},
                        d1::Vector{T}, d2::Vector{T}, d3::Vector{T}) where {T<:Union{Int, Float64}}
    
    N = size(Xp, 1)
    xb1 = Xp * b1
    exp_xb1 = exp.(xb1)
    
    w_d1d2 = (d1 .+ d2) .* exp_xb1
    w_d3 = d3 .* exp_xb1
    
    E_sum = EZil .+ EWil
    num = transpose(E_sum) * ones(N)
    
    den = transpose(bRi) * w_d1d2 .+ transpose(bLi) * w_d3
    
    g1 = num ./ den
    return g1
end

function Q2(b1::Vector{Float64}, EZil::Matrix{Float64}, EWil::Matrix{Float64},
                        Xp::Matrix{Float64}, bRi::Matrix{Float64}, bLi::Matrix{Float64},
                        d1::Vector{T}, d2::Vector{T}, d3::Vector{T}, g1::Vector{Float64}) where {T<:Union{Int, Float64}}
    
    N = size(Xp, 1)
    xb1 = Xp * b1
    exp_xb1 = exp.(xb1)
    log_g1 = log.(g1)
    
    # 使用矩阵运算计算 p1
    E_sum = EZil .+ EWil
    
    # p1 = sum(V .* E_sum .* (log_g1' .+ xb1))
    p1_part1 = dot(log_g1, transpose(E_sum) * ones(N))
    p1_part2 = dot(xb1, vec(sum(E_sum, dims=2)))
    
    p1 = p1_part1 + p1_part2
    
    # 计算 p2
    bRi_g = bRi * g1
    bLi_g = bLi * g1
    
    p2 = dot(exp_xb1 .* (d1 .+ d2), bRi_g) + dot(exp_xb1 .* d3, bLi_g)
    
    return -(p1 - p2)
end



function algo_full(d1::Vector{T}, d2::Vector{T}, d3::Vector{T},
                         Li::Vector{Float64}, Ri::Vector{Float64}, Xp::Matrix{Float64},
                         n_int::Int, order::Int, g0::Vector{Float64}, b0::Vector{Float64}; 
                         tol::Float64=5e-3, maxit::Int=1000,
                         t_seq::Union{Vector{Float64}, Nothing}=nothing, equal::Bool=false
                         ) where {T<:Union{Int, Float64}}
    
    P = length(b0)
    L = length(g0)
    N = length(d1)
    
    if t_seq === nothing
        t_seq = range(0, maximum(vcat(Li, Ri)), length=50)
    end
    
    Li[d1 .== 1] .= Ri[d1 .== 1]
    Ri[d3 .== 1] .= Li[d3 .== 1]
    
    ti = vcat(Li[d1 .== 0], Ri[d3 .== 0])
    
    if equal
        ti_max = maximum(ti) + 1e-5
        ti_min = minimum(ti) - 1e-5
        knots = range(ti_min, ti_max, length=n_int+2)
    else
        ti_max = maximum(ti) + 1e-5
        ti_min = minimum(ti) - 1e-5
        id = range(0, 1, length=n_int+2)
        id = id[2:end-1]
        knots = vcat(ti_min, quantile(ti, collect(id)), ti_max)
    end
    
    bRi = Ispline(Ri, order, knots)
    bLi = Ispline(Li, order, knots)
    bt = Ispline(collect(t_seq), order, knots)
    
    
    buffers = setup_buffers(N, L, P)
    
    dif = 1.0
    iter = 1
    b1 = copy(b0)
    g1 = copy(g0)
    b0_old = copy(b1)
    g0_old = copy(g1)
    
    while dif > tol && iter <= maxit
        b0_old = copy(b1)
        g0_old = copy(g1)
        
        mul!(buffers.GRi, bRi, g0_old)
        mul!(buffers.GLi, bLi, g0_old)
        
        mul!(buffers.xb0, Xp, b0_old)
        buffers.exp_xb0 .= exp.(buffers.xb0)
        
        buffers.GRi_exp .= buffers.GRi .* buffers.exp_xb0
        buffers.GLi_exp .= buffers.GLi .* buffers.exp_xb0
        
        buffers.dz .= 1 .- exp.(-buffers.GRi_exp)
        buffers.dz[d1 .== 0] .= 1.0
        
        buffers.dw .= 1 .- exp.(-(buffers.GRi_exp .- buffers.GLi_exp))
        buffers.dw[d2 .== 0] .= 1.0
        
        fill!(buffers.EZil, 0.0)
        fill!(buffers.EWil, 0.0)
        
        @inbounds for i in 1:N
            d3[i] == 1 && continue
            
            exp_xb0_i = buffers.exp_xb0[i]
            
            if d1[i] == 1
                factor = exp_xb0_i / buffers.dz[i]
                @inbounds for l in 1:L
                    buffers.EZil[i, l] = factor * bRi[i, l] * g0_old[l]
                end
            else
                factor = exp_xb0_i / buffers.dw[i]
                @inbounds for l in 1:L
                    buffers.EWil[i, l] = factor * (bRi[i, l] - bLi[i, l]) * g0_old[l]
                end
            end
        end
        
        objective(b) = Q2(b, buffers.EZil, buffers.EWil, Xp, bRi, bLi, 
                                 d1, d2, d3, g0_old)
        
        result = optimize(objective, b0_old, LBFGS())
        b1 = Optim.minimizer(result)
        
        g1 = Q1(b1, buffers.EZil, buffers.EWil, Xp, bRi, bLi, 
                           d1, d2, d3)
        
        dif = compute_difference(b0_old, g0_old, b1, g1)
        iter += 1
    end
    
    GRi_final = bRi * g1
    GLi_final = bLi * g1
    xb_final = Xp * b1
    exp_xb_final = exp.(xb_final)
    
    GRi_exp = GRi_final .* exp_xb_final
    GLi_exp = GLi_final .* exp_xb_final
    
    ll1 = sum(d1 .* log.(max.(1 .- exp.(-GRi_exp), 1e-10)))
    ll2 = sum(d2 .* log.(max.(exp.(-GLi_exp) - exp.(-GRi_exp), 1e-10)))
    ll3 = sum(d3 .* log.(max.(exp.(-GLi_exp), 1e-10)))
    ll = ll1 + ll2 + ll3
    
    k_params = length(b1) + length(g1)
    AIC = 2 * k_params - 2 * ll
    BIC = k_params * log(N) - 2 * ll
    
    hz = bt * g1
    
    return (b=b1, g=g1, hz=hz, ll=ll, AIC=AIC, BIC=BIC, iter=iter)
end


function algo_unis(d1_full::Vector{T}, d2_full::Vector{T}, d3_full::Vector{T},
                         Li_full::Vector{Float64}, Ri_full::Vector{Float64}, Xp_full::Matrix{Float64},
                         n_int::Int, order::Int, g0::Vector{Float64}, b0::Vector{Float64}, q::Float64, repl::Bool; 
                         tol::Float64=5e-3, maxit::Int=1000,
                         t_seq::Union{Vector{Float64}, Nothing}=nothing, equal::Bool=false,
                         rng_seed::Union{Int, Nothing}=nothing) where {T<:Union{Int, Float64}}
    
    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end
    
    P = length(b0)
    L = length(g0)
    N = length(d1_full)
    
    if t_seq === nothing
        t_seq = range(0, maximum(vcat(Li_full, Ri_full)), length=50)
    end
    
    n = round(Int, q * N)
    ind_sub = sample(1:N, n, replace = repl)

    d1 = d1_full[ind_sub]
    d2 = d2_full[ind_sub]
    d3 = d3_full[ind_sub]
    Li = Li_full[ind_sub]
    Ri = Ri_full[ind_sub]
    Xp = Xp_full[ind_sub, :]
    
    Li[d1 .== 1] .= Ri[d1 .== 1]
    Ri[d3 .== 1] .= Li[d3 .== 1]
    
    ti = vcat(Li[d1 .== 0], Ri[d3 .== 0])
    
    if equal
        ti_max = maximum(ti) + 1e-5
        ti_min = minimum(ti) - 1e-5
        knots = range(ti_min, ti_max, length=n_int+2)
    else
        ti_max = maximum(ti) + 1e-5
        ti_min = minimum(ti) - 1e-5
        id = range(0, 1, length=n_int+2)
        id = id[2:end-1]
        knots = vcat(ti_min, quantile(ti, collect(id)), ti_max)
    end
    
    bRi = Ispline(Ri, order, knots)
    bLi = Ispline(Li, order, knots)
    bt = Ispline(collect(t_seq), order, knots)
    
    
    buffers = setup_buffers(n, L, P)
    
    dif = 1.0
    iter = 1
    b1 = copy(b0)
    g1 = copy(g0)
    b0_old = copy(b1)
    g0_old = copy(g1)
    
    while dif > tol && iter <= maxit
        b0_old = copy(b1)
        g0_old = copy(g1)
        
        mul!(buffers.GRi, bRi, g0_old)
        mul!(buffers.GLi, bLi, g0_old)
        
        mul!(buffers.xb0, Xp, b0_old)
        buffers.exp_xb0 .= exp.(buffers.xb0)
        
        buffers.GRi_exp .= buffers.GRi .* buffers.exp_xb0
        buffers.GLi_exp .= buffers.GLi .* buffers.exp_xb0
        
        buffers.dz .= 1 .- exp.(-buffers.GRi_exp)
        buffers.dz[d1 .== 0] .= 1.0
        
        buffers.dw .= 1 .- exp.(-(buffers.GRi_exp .- buffers.GLi_exp))
        buffers.dw[d2 .== 0] .= 1.0
        
        fill!(buffers.EZil, 0.0)
        fill!(buffers.EWil, 0.0)
        
        @inbounds for i in 1:n
            d3[i] == 1 && continue
            
            exp_xb0_i = buffers.exp_xb0[i]
            
            if d1[i] == 1
                factor = exp_xb0_i / buffers.dz[i]
                @inbounds for l in 1:L
                    buffers.EZil[i, l] = factor * bRi[i, l] * g0_old[l]
                end
            else
                factor = exp_xb0_i / buffers.dw[i]
                @inbounds for l in 1:L
                    buffers.EWil[i, l] = factor * (bRi[i, l] - bLi[i, l]) * g0_old[l]
                end
            end
        end
        
        objective(b) = Q2(b, buffers.EZil, buffers.EWil, Xp, bRi, bLi, 
                                 d1, d2, d3, g0_old)
        
        result = optimize(objective, b0_old, LBFGS())
        b1 = Optim.minimizer(result)
        
        g1 = Q1(b1, buffers.EZil, buffers.EWil, Xp, bRi, bLi, 
                           d1, d2, d3)
        
        dif = compute_difference(b0_old, g0_old, b1, g1)
        iter += 1
    end
    
    GRi_final = bRi * g1
    GLi_final = bLi * g1
    xb_final = Xp * b1
    exp_xb_final = exp.(xb_final)
    
    GRi_exp = GRi_final .* exp_xb_final
    GLi_exp = GLi_final .* exp_xb_final
    
    ll1 = sum(d1 .* log.(max.(1 .- exp.(-GRi_exp), 1e-10)))
    ll2 = sum(d2 .* log.(max.(exp.(-GLi_exp) - exp.(-GRi_exp), 1e-10)))
    ll3 = sum(d3 .* log.(max.(exp.(-GLi_exp), 1e-10)))
    ll = ll1 + ll2 + ll3
    
    k_params = length(b1) + length(g1)
    AIC = 2 * k_params - 2 * ll
    BIC = k_params * log(N) - 2 * ll
    
    hz = bt * g1
    
    return (b=b1, g=g1, hz=hz, ll=ll, AIC=AIC, BIC=BIC, iter=iter)
end

function algo_nopert(d1_full::Vector{T}, d2_full::Vector{T}, d3_full::Vector{T},
                         Li_full::Vector{Float64}, Ri_full::Vector{Float64}, Xp_full::Matrix{Float64},
                         n_int::Int, order::Int, g0::Vector{Float64}, b0::Vector{Float64}, q::Float64; 
                         tol::Float64=5e-3, maxit::Int=1000,
                         t_seq::Union{Vector{Float64}, Nothing}=nothing, equal::Bool=false,
                         rng_seed::Union{Int, Nothing}=nothing) where {T<:Union{Int, Float64}}
    
    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end
    
    P = length(b0)
    L = length(g0)
    N = length(d1_full)
    
    if t_seq === nothing
        t_seq = range(0, maximum(vcat(Li_full, Ri_full)), length=50)
    end
    
    ind_sub = rand(Bernoulli(q), N) .== 1
    n = sum(ind_sub)

    d1 = d1_full[ind_sub]
    d2 = d2_full[ind_sub]
    d3 = d3_full[ind_sub]
    Li = Li_full[ind_sub]
    Ri = Ri_full[ind_sub]
    Xp = Xp_full[ind_sub, :]
    
    Li[d1 .== 1] .= Ri[d1 .== 1]
    Ri[d3 .== 1] .= Li[d3 .== 1]
    
    ti = vcat(Li[d1 .== 0], Ri[d3 .== 0])
    
    if equal
        ti_max = maximum(ti) + 1e-5
        ti_min = minimum(ti) - 1e-5
        knots = range(ti_min, ti_max, length=n_int+2)
    else
        ti_max = maximum(ti) + 1e-5
        ti_min = minimum(ti) - 1e-5
        id = range(0, 1, length=n_int+2)
        id = id[2:end-1]
        knots = vcat(ti_min, quantile(ti, collect(id)), ti_max)
    end
    
    bRi = Ispline(Ri, order, knots)
    bLi = Ispline(Li, order, knots)
    bt = Ispline(collect(t_seq), order, knots)
    
    
    buffers = setup_buffers(n, L, P)
    
    dif = 1.0
    iter = 1
    b1 = copy(b0)
    g1 = copy(g0)
    b0_old = copy(b1)
    g0_old = copy(g1)
    
    while dif > tol && iter <= maxit
        b0_old = copy(b1)
        g0_old = copy(g1)
        
        mul!(buffers.GRi, bRi, g0_old)
        mul!(buffers.GLi, bLi, g0_old)
        
        mul!(buffers.xb0, Xp, b0_old)
        buffers.exp_xb0 .= exp.(buffers.xb0)
        
        buffers.GRi_exp .= buffers.GRi .* buffers.exp_xb0
        buffers.GLi_exp .= buffers.GLi .* buffers.exp_xb0
        
        buffers.dz .= 1 .- exp.(-buffers.GRi_exp)
        buffers.dz[d1 .== 0] .= 1.0
        
        buffers.dw .= 1 .- exp.(-(buffers.GRi_exp .- buffers.GLi_exp))
        buffers.dw[d2 .== 0] .= 1.0
        
        fill!(buffers.EZil, 0.0)
        fill!(buffers.EWil, 0.0)
        
        @inbounds for i in 1:n
            d3[i] == 1 && continue
            
            exp_xb0_i = buffers.exp_xb0[i]
            
            if d1[i] == 1
                factor = exp_xb0_i / buffers.dz[i]
                @inbounds for l in 1:L
                    buffers.EZil[i, l] = factor * bRi[i, l] * g0_old[l]
                end
            else
                factor = exp_xb0_i / buffers.dw[i]
                @inbounds for l in 1:L
                    buffers.EWil[i, l] = factor * (bRi[i, l] - bLi[i, l]) * g0_old[l]
                end
            end
        end
        
        objective(b) = Q2(b, buffers.EZil, buffers.EWil, Xp, bRi, bLi, 
                                 d1, d2, d3, g0_old)
        
        result = optimize(objective, b0_old, LBFGS())
        b1 = Optim.minimizer(result)
        
        g1 = Q1(b1, buffers.EZil, buffers.EWil, Xp, bRi, bLi, 
                           d1, d2, d3)
        
        dif = compute_difference(b0_old, g0_old, b1, g1)
        iter += 1
    end
    
    GRi_final = bRi * g1
    GLi_final = bLi * g1
    xb_final = Xp * b1
    exp_xb_final = exp.(xb_final)
    
    GRi_exp = GRi_final .* exp_xb_final
    GLi_exp = GLi_final .* exp_xb_final
    
    ll1 = sum(d1 .* log.(max.(1 .- exp.(-GRi_exp), 1e-10)))
    ll2 = sum(d2 .* log.(max.(exp.(-GLi_exp) - exp.(-GRi_exp), 1e-10)))
    ll3 = sum(d3 .* log.(max.(exp.(-GLi_exp), 1e-10)))
    ll = ll1 + ll2 + ll3
    
    k_params = length(b1) + length(g1)
    AIC = 2 * k_params - 2 * ll
    BIC = k_params * log(N) - 2 * ll
    
    hz = bt * g1
    
    return (b=b1, g=g1, hz=hz, ll=ll, AIC=AIC, BIC=BIC, iter=iter)
end
