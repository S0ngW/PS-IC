# User needs to modify this root directory
const ROOT_DIR = ""

cd(joinpath(ROOT_DIR, "code"))

using Random, Roots, Dates
include(joinpath(ROOT_DIR, "code", "algos.jl"))
include(joinpath(ROOT_DIR, "code", "algos_compare.jl"))
using JLD2

# Define global constants
const b_true = [1.0, 1.0]
const ord = 3
const nint = 3
const b_init = zeros(2)
const g_init = ones(ord + nint)
const h_type = 3

function generate_survival_data(n::Int, beta_true::Vector{Float64}, seed::Union{Int, Nothing} = nothing; hazard_type::Int = 1)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    Lambda_0 = if hazard_type == 1
        t -> log(1 + t) + sqrt(t)
    elseif hazard_type == 2
        t -> 0.5 * sqrt(t)
    elseif hazard_type == 3
        t -> t/10 - log(1 + t/10)
    else
        error("hazard_type must be 1, 2, or 3")
    end
    
    x1 = rand(Binomial(1, 0.5), n)
    sigma = 0.5
    if hazard_type == 1
        sigma = 0.5
    elseif hazard_type == 2
        sigma = 0.5
    elseif hazard_type == 3
        sigma = 0.25
    else
        error("hazard_type must be 1, 2, or 3")
    end
    x2 = rand(Normal(0, sigma), n)
    Xp = hcat(x1, x2)
    
    U = rand(n)
    
    function find_survival_time(i::Int)::Float64
        function f(t::Float64)::Float64
            exponent = exp(dot(Xp[i, :], beta_true))
            return exp(-Lambda_0(t) * exponent) - U[i]
        end
        
        try
            return find_zero(f, (0.0, 1e6), Bisection(), xatol=1e-8)
        catch e
            println("Warning: No root found for i=$i, U[i]=$(U[i])")
            return 0.0
        end
    end
    
    T_true = zeros(n)
    for i in 1:n
        T_true[i] = find_survival_time(i)
    end
    
    Li = zeros(n)
    Ri = zeros(n)
    d1 = zeros(n)
    d2 = zeros(n)
    d3 = zeros(n)
    
    mu_pois = 6
    mu_exp = 0.5
    if hazard_type == 3
        mu_pois = 9
        mu_exp = 0.5
    end

    for i in 1:n
        num_obs = rand(Poisson(mu_pois)) + 1
        intervals = rand(Exponential(mu_exp), num_obs)
        obs_times = cumsum(intervals)
        
        if T_true[i] < minimum(obs_times)
            Li[i] = 0.0
            Ri[i] = minimum(obs_times)
            d1[i] = 1.0
        elseif T_true[i] > maximum(obs_times)
            Li[i] = maximum(obs_times)
            Ri[i] = 0.0
            d3[i] = 1.0
        else
            idx = findfirst(x -> x > T_true[i], obs_times)
            
            if idx === nothing
                Li[i] = maximum(obs_times)
                Ri[i] = 0.0
                d3[i] = 1.0
            elseif idx == 1
                Li[i] = 0.0
                Ri[i] = obs_times[1]
                d1[i] = 1.0
            else
                Li[i] = obs_times[idx-1]
                Ri[i] = obs_times[idx]
                d2[i] = 1.0
            end
        end
    end
    
    return d1, d2, d3, Li, Ri, Xp
end

function simu(sim_id, n::Int; max_iter::Int = 1000)
    t_start_full = time_ns()
    Random.seed!(2025 + 100 * sim_id)
    d1, d2, d3, Li, Ri, Xp = generate_survival_data(n, b_true; hazard_type = h_type)

    t_start_algo = time_ns()
    b_full = algo_full(d1, d2, d3, Li, Ri, Xp, nint, ord, g_init, b_init; maxit = max_iter).b
    t_end_algo = time_ns()
    time_ms_full = (t_end_algo - t_start_algo)/ 1e6
    cMSE_full = 0
    uMSE_full = sum((b_full .- b_true).^2)

    q_ns = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    v_dists = ["gamma","beta","Geom"]
    Ms = [50,100]

    res_unis = zeros(length(q_ns), 3)
    res_unis_norep = zeros(length(q_ns), 3)
    res_unis_nopert = zeros(length(q_ns), 3)
    res_a1 = zeros(length(q_ns), length(v_dists), 3)
    res_a2 = zeros(length(q_ns), length(v_dists), length(Ms), 7)
    n_records = 2 + 7 * 3 + 7 * 3 + 7 * 3 * 2
    beta_values = zeros(Float64, n_records, 6)
    idx = 1
    beta_values[idx, :] = [0.0, 0.0, 0.0, 0.0, b_true[1], b_true[2]]; idx += 1
    beta_values[idx, :] = [1.0, 0.0, 0.0, 0.0, b_full[1], b_full[2]]; idx += 1

    for q_i in 1:length(q_ns)
        q_n = q_ns[q_i]

        t_start_algo = time_ns()
        b_sub = algo_unis(d1, d2, d3, Li, Ri, Xp, nint, ord, g_init, b_init, q_n, true; maxit = max_iter).b
        t_end_algo = time_ns()
        res_unis[q_i, 1] = (t_end_algo - t_start_algo)/ 1e6
        res_unis[q_i, 2] = sum((b_sub .- b_full).^2)
        res_unis[q_i, 3] = sum((b_sub .- b_true).^2)
        beta_values[idx, :] = [q_n, 1.0, 0.0, 0.0, b_sub[1], b_sub[2]]; idx += 1

        t_start_algo = time_ns()
        b_sub = algo_unis(d1, d2, d3, Li, Ri, Xp, nint, ord, g_init, b_init, q_n, false; maxit = max_iter).b
        t_end_algo = time_ns()
        res_unis_norep[q_i, 1] = (t_end_algo - t_start_algo)/ 1e6
        res_unis_norep[q_i, 2] = sum((b_sub .- b_full).^2)
        res_unis_norep[q_i, 3] = sum((b_sub .- b_true).^2)
        beta_values[idx, :] = [q_n, 2.0, 0.0, 0.0, b_sub[1], b_sub[2]]; idx += 1

        t_start_algo = time_ns()
        b_sub = algo_nopert(d1, d2, d3, Li, Ri, Xp, nint, ord, g_init, b_init, q_n; maxit = max_iter).b
        t_end_algo = time_ns()
        res_unis_nopert[q_i, 1] = (t_end_algo - t_start_algo)/ 1e6
        res_unis_nopert[q_i, 2] = sum((b_sub .- b_full).^2)
        res_unis_nopert[q_i, 3] = sum((b_sub .- b_true).^2)
        beta_values[idx, :] = [q_n, 3.0, 0.0, 0.0, b_sub[1], b_sub[2]]; idx += 1

        for v_i in 1:length(v_dists)
            v_dist = v_dists[v_i]

            t_start_algo = time_ns()
            b_sub = algo1(d1, d2, d3, Li, Ri, Xp, nint, ord, g_init, b_init, q_n; V_dist = v_dist, maxit = max_iter).b
            t_end_algo = time_ns()
            res_a1[q_i, v_i, 1] = (t_end_algo - t_start_algo)/ 1e6
            res_a1[q_i, v_i, 2] = sum((b_sub .- b_full).^2)
            res_a1[q_i, v_i, 3] = sum((b_sub .- b_true).^2)
            beta_values[idx, :] = [q_n, 4.0, v_i, 0.0, b_sub[1], b_sub[2]]; idx += 1

            for M_i in 1:length(Ms)
                M = Ms[M_i]
                t_start_algo = time_ns()
                b_sub, cov_cond, cov_uncond = algo2(d1, d2, d3, Li, Ri, Xp, nint, ord, g_init, b_init, q_n, M; V_dist = v_dist, maxit = max_iter)
                t_end_algo = time_ns()
                res_a2[q_i, v_i, M_i, 1] = (t_end_algo - t_start_algo)/ 1e6
                res_a2[q_i, v_i, M_i, 2] = sum((b_sub .- b_full).^2)
                res_a2[q_i, v_i, M_i, 3] = sum((b_sub .- b_true).^2) 
                se_cond = sqrt.(diag(cov_cond))
                res_a2[q_i, v_i, M_i, 4] = abs(b_full[1] - b_sub[1]) < 1.96*se_cond[1]
                res_a2[q_i, v_i, M_i, 5] = abs(b_full[2] - b_sub[2]) < 1.96*se_cond[2]
                se_uncond = sqrt.(diag(cov_uncond))
                res_a2[q_i, v_i, M_i, 6] = abs(b_true[1] - b_sub[1]) < 1.96*se_uncond[1]
                res_a2[q_i, v_i, M_i, 7] = abs(b_true[2] - b_sub[2]) < 1.96*se_uncond[2]
                beta_values[idx, :] = [q_n, 5.0, v_i, M, b_sub[1], b_sub[2]]; idx += 1
            end
        end
    end

    return (
        sim_id = sim_id,
        time_ms_full = time_ms_full,
        cMSE_full = cMSE_full,
        uMSE_full = uMSE_full,
        res_unis = res_unis,
        res_unis_norep = res_unis_norep,
        res_unis_nopert = res_unis_nopert,
        res_a1 = res_a1,
        res_a2 = res_a2,
        beta_values = beta_values,
    )
end

function current_time_str()
    return Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
end

function format_duration(seconds::Float64)
    h = floor(Int, seconds ÷ 3600)
    m = floor(Int, (seconds % 3600) ÷ 60)
    s = round(Int, seconds % 60)

    if h > 0
        return "$(h)h$(m)m$(s)s"
    elseif m > 0
        return "$(m)m$(s)s"
    else
        return "$(s)s"
    end
end


sim_id = parse(Int, ARGS[1])
n_sim = parse(Int, ARGS[2])
n_sims = parse(Int, ARGS[3])

base = (sim_id - 1) * n_sims

global_id = base + n_sim

n = 20000
# ===== Warm-up =====
t0 = time()
global log_str = "Warm-up started at $(current_time_str())\n"
println("[Process $sim_id] Simulation $global_id warm-up started at $(current_time_str())")

try
    simu(sim_id, n; max_iter=1)  # small-scale compilation
    t1 = time()
    msg = "Warm-up completed at $(current_time_str()), duration $(format_duration(t1 - t0))"
    global log_str *= "$msg\n"
    println("[Process $sim_id] $msg")
catch e
    msg = "Warm-up failed at $(current_time_str())"
    global log_str *= "$msg\n"
    println("[Process $sim_id] $msg")
end



t_start = time()
msg_start = "[Process $sim_id] Simulation $global_id started at $(current_time_str())"
global log_str *= "$msg_start\n"
println(msg_start)

result = simu(global_id, n)

t_end = time()

t_ymdhms = current_time_str()
msg_end = "[Process $sim_id] Simulation $global_id completed at $t_ymdhms, duration $(format_duration(t_end - t_start))"
global log_str *= "$msg_end\n"


# ===== Save =====
results_dir = joinpath(ROOT_DIR, "results", "simu_hz$(h_type)_500")
mkpath(results_dir)
filename = joinpath(results_dir, "simulation_results_hz$(h_type)_n$(n)_id$(global_id).jld2")
@save filename result log_str

println("[Process $sim_id] Simulation $global_id completed at $t_ymdhms, duration $(format_duration(t_end - t_start)), results saved")
