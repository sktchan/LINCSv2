using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using JLD2, CairoMakie, StatsBase

# using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2, SparseArrays, Dates, Printf, Profile
# using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra, MLUtils

# CUDA.device!(0)

all_trues = load("/home/golem/scratch/chans/lincsv2/plots/untrt/rank_tf/2025-12-05_01-33/predstrues.jld2")["all_trues"]
all_preds = load("/home/golem/scratch/chans/lincsv2/plots/untrt/rank_tf/2025-12-05_01-33/predstrues.jld2")["all_preds"]

cs = corspearman(all_trues, all_preds)
cp = cor(all_trues, all_preds)

# exp val
begin
    fig_hex = Figure(size = (800, 700))
    ax_hex = Axis(fig_hex[1, 1],
        # backgroundcolor = to_colormap(:viridis)[1], 
        xlabel="true expression value",
        ylabel="predicted expression value"
        # title="predicted vs. true gene id density"
        # aspect=DataAspect() 
    )
    hexplot = hexbin!(ax_hex, all_trues, all_preds, cellsize = (0.06,0.06), colorscale = log10)
    # text!(ax_hex, 0, 1050, align = (:left, :top), text = "Pearson: $cp")
    Colorbar(fig_hex[1, 2], hexplot, label="point count (log10)")
    display(fig_hex)
end
save_dir = "/home/golem/scratch/chans/lincsv2/plots/untrt/TEST_rank_tf/baseline"
save(joinpath(save_dir, "exp_nn_hbin.png"), fig_hex)
print(cs)



# rank id

# # to sort x axis
sorted_indices_by_mean = load("/home/golem/scratch/chans/lincsv2/plots/untrt/infographs/sorted_gene_indices_by_exp.jld2")["sorted_indices_by_mean"]
gene_id_to_rank_map = invperm(sorted_indices_by_mean);
sorted_trues = gene_id_to_rank_map[all_trues];
sorted_preds = gene_id_to_rank_map[all_preds];

bin_edges = 1:979 
h = fit(Histogram, (sorted_trues, sorted_preds), (bin_edges, bin_edges))
begin
    fig_hm = Figure(size = (500, 400))
    ax_hm = Axis(fig_hm[1, 1],
        xlabel = "true rank",
        ylabel = "predicted rank"
    )

    log10_weights = log10.(h.weights .+ 1)
    hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
    text!(ax_hm, 20, 950, align = (:left, :top), text = "Pearson: $cp", color = :white)
    Colorbar(fig_hm[1, 2], hm, label = "count (log10)")
    display(fig_hm)
end
save_dir = "/home/golem/scratch/chans/lincsv2/plots/untrt/TEST_rank_tf/normalized/2025-12-03_03-22"
save(joinpath(save_dir, "heatmap.png"), fig_hm)