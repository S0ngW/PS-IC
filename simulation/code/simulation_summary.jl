
using JLD2, Statistics, DataFrames

# User only needs to modify this root directory
const ROOT_DIR = ""

# Three target subdirectories
const SUB_DIRS = ["simu_hz1_500", "simu_hz2_500", "simu_hz3_500"]

for sub_dir in SUB_DIRS
    local results_dir = joinpath(ROOT_DIR, sub_dir)
    println("\n" * "="^60)
    println("Processing folder: $results_dir")
    println("="^60)

    all_files = readdir(results_dir)
    # Filter jld2 files (start with 's' and end with '.jld2')
    files = filter(f -> startswith(f, "s") && endswith(f, ".jld2"), all_files)

    println("Found $(length(files)) result files")

    if length(files) == 0
        println("No result files found, skip this folder.\n")
        continue
    end

    # Read the first file to obtain structure
    first_file = joinpath(results_dir, files[1])
    result = load(first_file)["result"]

    # Obtain dimension information
    q_len = size(result.res_unis, 1)
    v_len = size(result.res_a1, 2)
    m_len = size(result.res_a2, 3)

    println("Dimension info: q_len=$q_len, v_len=$v_len, m_len=$m_len")

    # Initialize accumulators
    n_files = length(files)
    sum_time_ms_full = 0.0
    sum_uMSE_full = 0.0
    sum_res_unis = zeros(q_len, 3)
    sum_res_unis_norep = zeros(q_len, 3)
    sum_res_unis_nopert = zeros(q_len, 3)
    sum_res_a1 = zeros(q_len, v_len, 3)
    sum_res_a2 = zeros(q_len, v_len, m_len, 7)

    all_times = Float64[]
    all_uMSEs = Float64[]

    for (idx, file) in enumerate(files)
        filepath = joinpath(results_dir, file)
        result = load(filepath)["result"]

        sum_time_ms_full += result.time_ms_full
        sum_uMSE_full += result.uMSE_full

        sum_res_unis .+= result.res_unis
        sum_res_unis_norep .+= result.res_unis_norep
        sum_res_unis_nopert .+= result.res_unis_nopert
        sum_res_a1 .+= result.res_a1
        sum_res_a2 .+= result.res_a2

        push!(all_times, result.time_ms_full)
        push!(all_uMSEs, result.uMSE_full)

        if idx % 50 == 0
            println("Processed $idx/$n_files")
        end
    end

    # Compute averages
    results_avg = (
        n_sims = n_files,
        n = 20000,
        mean_time_ms_full = sum_time_ms_full / n_files,
        mean_uMSE_full = sum_uMSE_full / n_files,
        mean_res_unis = sum_res_unis / n_files,
        mean_res_unis_norep = sum_res_unis_norep / n_files,
        mean_res_unis_nopert = sum_res_unis_nopert / n_files,
        mean_res_a1 = sum_res_a1 / n_files,
        mean_res_a2 = sum_res_a2 / n_files
    )

    # Save aggregated results
    output_file = joinpath(results_dir, "aggregated_results_$(n_files).jld2")
    @save output_file results_avg
    println("Aggregated results saved to: $output_file")

    # ---- Extract coverages and generate table ----
    cov = results_avg.mean_res_a2[:, :, :, 4:7]   # [cCP1, cCP2, uCP1, uCP2]

    q_ns = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    dist_names = ["Gamma", "Beta", "Geom"]
    M_vals = [50, 100]

    n_q = length(q_ns)
    n_dist = 3
    n_M = 2

    n_rows = 2 * n_q
    n_cols = 1 + n_dist * 2 * n_M   # 1 for q_n, then 12 coverage columns

    res = zeros(n_rows, n_cols)
    res[:, 1] = repeat(q_ns, inner=2)          # q_n column

    col_start = 2
    for dist in 1:n_dist
        for M in 1:n_M
            ccp1 = cov[:, dist, M, 1] .* 100
            ccp2 = cov[:, dist, M, 2] .* 100
            ucp1 = cov[:, dist, M, 3] .* 100
            ucp2 = cov[:, dist, M, 4] .* 100

            for (row_q, q_idx) in enumerate(1:n_q)
                row_odd  = 2*row_q - 1
                row_even = 2*row_q
                res[row_odd,  col_start]   = ccp1[q_idx]
                res[row_odd,  col_start+1] = ucp1[q_idx]
                res[row_even, col_start]   = ccp2[q_idx]
                res[row_even, col_start+1] = ucp2[q_idx]
            end
            col_start += 2
        end
    end

    col_names = ["q_n"]
    for dist in dist_names
        for M in M_vals
            push!(col_names, "$(dist)_cCP_$M")
            push!(col_names, "$(dist)_uCP_$M")
        end
    end

    df = DataFrame(res, Symbol.(col_names))
    insertcols!(df, 2, :beta => repeat(["β₁", "β₂"], n_q))

    println("\nCoverage probabilities (%) for $sub_dir:")
    show(df, allcols=true)
    println()
end
