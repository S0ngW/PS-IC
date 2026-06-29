

function Ispline(x::Vector{Float64}, order::Int, knots::Vector{Float64})
    k = order + 1
    m = length(knots)
    n = m - 2 + k
    
    t = vcat(fill(knots[1], k), knots[2:(m-1)], fill(knots[m], k))
    
    length_x = length(x)
    yy1 = zeros(n + k - 1, length_x)
    
    for l in k:n
        for j in 1:length_x
            if t[l] <= x[j] < t[l+1]
                yy1[l, j] = 1.0 / (t[l+1] - t[l])
            end
        end
    end
    
    yytem1 = yy1
    yytem2 = nothing
    for ii in 1:order
        yytem2 = zeros(n + k - 1 - ii, length_x)
        for i in (k - ii):n
            for j in 1:length_x
                denom = t[i + ii + 1] - t[i]
                if denom > 0
                    yytem2[i, j] = (ii + 1) * ((x[j] - t[i]) * yytem1[i, j] + 
                                                (t[i + ii + 1] - x[j]) * yytem1[i + 1, j]) / 
                                                (denom * ii)
                end
            end
        end
        yytem1 = yytem2
    end
    
    index = zeros(Int, length_x)
    for i in 1:length_x
        idx = findlast(t .<= x[i])
        index[i] = isnothing(idx) ? 0 : idx
    end
    
    yy = zeros(n - 1, length_x)  
    
    if order == 1
        for i in 2:n
            for j in 1:length_x
                if i < index[j] - order + 1
                    yy[i-1, j] = 1.0
                elseif i == index[j]
                    denom = t[i + order + 1] - t[i]
                    if denom > 0
                        yy[i-1, j] = denom * yytem2[i, j] / (order + 1)
                    end
                end
            end
        end
    else
        for j in 1:length_x
            for i in 2:n
                if i < index[j] - order + 1
                    yy[i-1, j] = 1.0
                elseif i <= index[j] && i >= index[j] - order + 1
                    sum_val = 0.0
                    for kk in i:index[j]
                        denom = t[kk + order + 1] - t[kk]
                        if denom > 0
                            sum_val += denom * yytem2[kk, j]
                        end
                    end
                    yy[i-1, j] = sum_val / (order + 1)
                else
                    yy[i-1, j] = 0.0
                end
            end
        end
    end
    
    return Matrix(yy')  
end




function Q1_V(b1::Vector{Float64}, EZil::AbstractMatrix{Float64}, EWil::AbstractMatrix{Float64},
                        Xp::AbstractMatrix{Float64}, bRi::AbstractMatrix{Float64}, bLi::AbstractMatrix{Float64},
                        d1::AbstractVector{T}, d2::AbstractVector{T}, d3::AbstractVector{T},
                        V::AbstractVector{Float64}) where {T<:Union{Int, Float64}}
    
    xb1 = Xp * b1
    exp_xb1 = exp.(xb1)
    
    w_d1d2 = V .* (d1 .+ d2) .* exp_xb1
    w_d3 = V .* d3 .* exp_xb1
    
    E_sum = EZil .+ EWil
    num = transpose(E_sum) * V
    
    den = transpose(bRi) * w_d1d2 .+ transpose(bLi) * w_d3
    
    g1 = num ./ den
    return g1
end


function Q2_V_fast(b1::Vector{Float64}, EZil::AbstractMatrix{Float64}, EWil::AbstractMatrix{Float64},
                        Xp::AbstractMatrix{Float64}, bRi::AbstractMatrix{Float64}, bLi::AbstractMatrix{Float64},
                        d1::AbstractVector{T}, d2::AbstractVector{T}, d3::AbstractVector{T}, g1::Vector{Float64},
                        V::AbstractVector{Float64}) where {T<:Union{Int, Float64}}
    
    xb1 = Xp * b1
    exp_xb1 = exp.(xb1)
    log_g1 = log.(g1)
    
    # 使用矩阵运算计算 p1
    E_sum = EZil .+ EWil
    
    # p1 = sum(V .* E_sum .* (log_g1' .+ xb1))
    p1_part1 = dot(log_g1, transpose(E_sum) * V)
    p1_part2 = dot(xb1 .* V, vec(sum(E_sum, dims=2)))
    
    p1 = p1_part1 + p1_part2
    
    # 计算 p2
    bRi_g = bRi * g1
    bLi_g = bLi * g1
    
    w = V .* exp_xb1
    p2 = dot(w .* (d1 .+ d2), bRi_g) + dot(w .* d3, bLi_g)
    
    return -(p1 - p2)
end

function setup_buffers(n, L, P)
    return (
        GRi = zeros(n),
        GLi = zeros(n),
        xb0 = zeros(n),
        exp_xb0 = zeros(n),
        GRi_exp = zeros(n),
        GLi_exp = zeros(n),
        dz = zeros(n),
        dw = zeros(n),
        EZil = zeros(n, L),
        EWil = zeros(n, L)
    )
end

function compute_difference(b_old, g_old, b_new, g_new)
    max_diff = 0.0
    @inbounds for i in 1:length(b_old)
        diff = abs(b_old[i] - b_new[i])
        if diff > max_diff
            max_diff = diff
        end
    end
    @inbounds for i in 1:length(g_old)
        diff = abs(g_old[i] - g_new[i])
        if diff > max_diff
            max_diff = diff
        end
    end
    return max_diff
end

function manual_cov(X::Matrix{Float64}, mean_X::Matrix{Float64}, n::Int)    
    centered = X .- mean_X
    cov_matrix = (centered' * centered) ./ (n - 1)
    
    return cov_matrix
end