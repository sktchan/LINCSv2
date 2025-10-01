using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2, SparseArrays, Dates, Printf, Profile
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra, MLUtils

CUDA.device!(1)
# GC.gc()
# CUDA.reclaim()

#######################################################################################################################################
### INFO
#######################################################################################################################################

# untrt only
# const data_path = "data/lincs_untrt_data.jld2"
# const save_path_base = "untrt"

# untrt and trt
const data_path = "data/lincs_trt_untrt_data.jld2"
const save_path_base = "trt"

# params
const batch_size = 512
const n_epochs = 30
const embed_dim = 128
const hidden_dim = 256
const mid_dim = 64
const latent_dim = 32
const mask_ratio = 0.1f0
const mask_value = 0.0f0
const lr = 0.001
# const drop_prob = 0.05

# notes
const gpu_info = "this was on kraken"
const dataset_note = "trt"
const additional_notes = "30ep with fixed layernorm"

#######################################################################################################################################
### DATA
#######################################################################################################################################

start_time = now()

data = load(data_path)["filtered_data"]

@time X = data.expr #!# use raw expression values!!!

const n_genes = size(X, 1)
# const n_classes = 1 #!# n_classes is 1 for regression (n_features is not vocabulary size)

#######################################################################################################################################
### MODEL
#######################################################################################################################################

#=
PIPELINE:
FNN: 978-length vector --> embed into embed_dim vector
Encoder: embed_dim vector --> masked -> compressed into latent_dim vector
Decoder: latent_dim vector --> reconstructed into embed_dim vector
evaluate on embed_dim vector similarity
later:
    FNN: embed_dim vector --> downstream task (single value for regression or n_classes vector for classification)
=#

#=
thing is, we have two options for noising and two options for loss calculation:

noising:
A. masking
    - sets a fixed % of rows per sample to set to 0
    - no scaling mean
B. Flux.Dropout
    - randomly sets each element to 0 with a given dropout_probability
    - scales up the remaining elements so that the mean remains the same

loss caluclation:
A. loss only calculated on teh values that are masked (close to transf)
B. loss calcaulted on the entire reconstructed output and entire original input (close to trad DAE)

for now, will use:
    1. masking
        - more specific task; not just a regularizer to prevent overfitting..?
    2. loss only calculated on masked values
        - could get an artifically low loss score by perfectly reconstructing the unmasked portions
        - forces it to get good at the task at hand
        - also less computational effort to caluclate loss on more values?
mostly just becasue this is closer to the transformer task and this is the main goal; comparison!

OR:
- since we only calculate loss on the 10% masked values
- the model has no incentive to maintain meaningful representations for the 90% unmasked values
- therefore this leads to poor overall representations...?
=#

# for use of both CPU and GPU
# const IntMatrix2DType = Union{Array{Int}, CuArray{Int, 2}}
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
# const Float32Matrix3DType = Union{Array{Float32}, CuArray{Float32, 3}}

### masking

struct Mask
    mask_ratio::Float32
    mask_value::Float32
end

# function (m::Mask)(x::AbstractMatrix{Float32})
#     X_masked = copy(x)
#     mask_labels = fill(NaN32, size(x)) #!# NaN to ignore positions in the loss calculation

#     for j in 1:size(x,2) # per column
#         num_masked = ceil(Int, size(x,1) * m.mask_ratio)
#         mask_positions = randperm(size(x,1))[1:num_masked]

#         for pos in mask_positions
#             mask_labels[pos, j] = x[pos, j] 
#             X_masked[pos, j] = m.mask_value
#         end
#     end
#     return X_masked, mask_labels
# end

### changed to this masking function since masking is done on gpu within the model fxn, not cpu beforehand (as prev in tf file)
function (m::Mask)(x::AbstractMatrix{Float32})
    # random boolean mask for entire matrix for what to mask
        # rand_like: all element of the new array will be set to a random value. 
        # .< mask_ratio: if value is < mask ratio, then it is marked as true
    mask_indices = rand_like(x) .< m.mask_ratio

    # place mask on data
        # if true, places mask_value into the new matrix
        # if false, it copies the original value from x
    X_masked = ifelse.(mask_indices, m.mask_value, x)

    # get masked labels
        # if true, gets original value from x
        # if false, it puts NaN32 to skip over in loss
    mask_labels = ifelse.(mask_indices, x, NaN32)

    return X_masked, mask_labels
end

### encoder

struct Encoder
    noise::Mask
    compress::Flux.Chain
end

function Encoder(
    embed_dim::Int,
    mid_dim::Int,
    latent_dim::Int,
    mask_ratio::Float32,
    mask_value::Float32
    )

    noise = Mask(mask_ratio, mask_value)

    compress = Flux.Chain(
        Flux.Dense(embed_dim => mid_dim, relu),
        Flux.Dense(mid_dim => latent_dim)
    )

    return Encoder(noise, compress)
end

Flux.@functor Encoder (compress,)

function (enc::Encoder)(input::Float32Matrix2DType)
    noised, labels = enc.noise(input)
    compressed = enc.compress(noised)
    return compressed, labels
end

### decoder

struct Decoder
    reconstruct::Flux.Chain
end

function Decoder(
    embed_dim::Int,
    mid_dim::Int,
    latent_dim::Int
    )

    reconstruct = Flux.Chain(
        Flux.Dense(latent_dim => mid_dim),
        Flux.Dense(mid_dim => embed_dim)
    )

    return Decoder(reconstruct)
end

Flux.@functor Decoder

function (dec::Decoder)(input::Float32Matrix2DType)
    return dec.reconstruct(input)
end

### full model

struct Model
    mlp::Flux.Chain
    encoder::Encoder
    decoder::Decoder
    # mlp_head::Flux.Chain
end

function Model(;
    num_genes::Int,
    hidden_dim::Int,
    embed_dim::Int,
    mid_dim::Int,
    latent_dim::Int,
    mask_ratio::Float32,
    mask_value::Float32
    # n_classes::Int,
    # dropout_prob::Float64
    )

    mlp = Flux.Chain(
        Dense(num_genes => hidden_dim, relu),
        LayerNorm(hidden_dim),
        relu,
        # Dropout(0.1),
        Dense(hidden_dim => embed_dim) # softplus?
    )

    encoder = Encoder(embed_dim, mid_dim, latent_dim, mask_ratio, mask_value)

    decoder = Decoder(embed_dim, mid_dim, latent_dim)

    # mlp_head = Flux.Chain(
    #     Dense(embed_dim => num_genes)
    # )

    return Model(mlp, encoder, decoder)
end

Flux.@functor Model

function (model::Model)(input::AbstractMatrix{Float32})
    embedding = model.mlp(input)
    latent, labels = model.encoder(embedding)
    recon_embed = model.decoder(latent)
    return recon_embed, labels
end

#######################################################################################################################################
### DATA PREP
#######################################################################################################################################

### splitting data

function split_data(X, test_ratio::Float64, y=nothing)
    n = size(X, 2)
    indices = shuffle(1:n)

    test_size = floor(Int, n * test_ratio)
    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]

    X_train = X[:, train_indices]
    X_test = X[:, test_indices]

    if y === nothing
        return X_train, X_test
    else
        y_train = y[train_indices]
        y_test = y[test_indices]
        return X_train, y_train, X_test, y_test
    end
end

X_train, X_test = split_data(X, 0.2)

#######################################################################################################################################
### TRAINING
#######################################################################################################################################

model = Model(
    num_genes=n_genes,
    hidden_dim=hidden_dim,
    embed_dim=embed_dim,
    mid_dim=mid_dim,
    latent_dim=latent_dim,
    mask_ratio=mask_ratio,
    mask_value=mask_value
) |> gpu

opt = Flux.setup(Adam(lr), model)

function loss(model::Model, x, mode::String)
    preds, trues = model(x)
    preds_flat = vec(preds)
    trues_flat = vec(trues)

    mask = .!isnan.(trues_flat)
    
    preds_masked = preds_flat[mask]
    trues_masked = trues_flat[mask]
    
    error = Flux.mse(preds_masked, trues_masked)

    if mode == "train"
        return error
    end
    if mode == "test"
        return error, preds_masked, trues_masked, trues
    end
end

train_losses = Float32[]
test_losses = Float32[]
all_preds = Float32[]
all_trues = Float32[]
all_gene_indices = Int[]

for epoch in ProgressBar(1:n_epochs)
    train_epoch_losses = Float32[]
    for start_idx in 1:batch_size:size(X_train, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_train, 2))
        x_gpu = gpu(X_train[:, start_idx:end_idx])

        loss_val, grads = Flux.withgradient(model) do m
            loss(m, x_gpu, "train")
        end
        Flux.update!(opt, model, grads[1])
        loss_val = loss(model, x_gpu, "train")
        push!(train_epoch_losses, loss_val)
    end
    push!(train_losses, mean(train_epoch_losses))

    test_epoch_losses = Float32[]
    for start_idx in 1:batch_size:size(X_test, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test, 2))
        x_gpu = gpu(X_test[:, start_idx:end_idx])

        test_loss_val, preds_masked, trues_masked, trues_full = loss(model, x_gpu, "test")
        push!(test_epoch_losses, test_loss_val)

        if epoch == n_epochs
            append!(all_preds, cpu(preds_masked))
            append!(all_trues, cpu(trues_masked))

            trues_cpu = cpu(trues_full)
            masked_indices = findall(!isnan, trues_cpu)
            batch_gene_indices = [idx[1] for idx in masked_indices]
            append!(all_gene_indices, batch_gene_indices)
        end
    end
    push!(test_losses, mean(test_epoch_losses))
end

correlation = cor(all_trues, all_preds)


#######################################################################################################################################
### EVAL/PLOT
#######################################################################################################################################

# mk dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", save_path_base, "exp_nn", timestamp)
mkpath(save_dir)

# loss plot
fig_loss = Figure(size = (800, 600))
ax_loss = Axis(fig_loss[1, 1], 
    xlabel="epoch", 
    ylabel="loss (mse)", 
    title="train vs. test loss"
)
lines!(ax_loss, 1:n_epochs, train_losses, label="train loss", linewidth=2)
lines!(ax_loss, 1:n_epochs, test_losses, label="test loss", linewidth=2)
axislegend(ax_loss, position=:rt)
display(fig_loss)
save(joinpath(save_dir, "loss.png"), fig_loss)

### boxplot and histogram combined

min_val = minimum(all_trues)
max_val = maximum(all_trues)

# define bins
# bin_edges = min_val:0.005:max_val
n_bins = 15
bin_edges = range(min_val, max_val, length=n_bins + 1)
bin_midpts = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2

stats = Dict()
x_outliers = Float32[]
y_outliers = Float32[]
grouped_preds = Float32[]
grouped_trues_midpts = Float64[]

for (i, midpt) in enumerate(bin_midpts) # for each bin
    ind = findall(x -> bin_edges[i] <= x < bin_edges[i+1], all_trues)
    if !isempty(ind)
        bin_preds = all_preds[ind]
        append!(grouped_preds, bin_preds)
        append!(grouped_trues_midpts, fill(midpt, length(bin_preds)))

        q10 = quantile(bin_preds, 0.10)
        q25 = quantile(bin_preds, 0.25)
        q50 = quantile(bin_preds, 0.50)
        q75 = quantile(bin_preds, 0.75)
        q90 = quantile(bin_preds, 0.90)

        stats[midpt] = (q10=q10, q25=q25, q50=q50, q75=q75, q90=q90)
        outlier_ind = findall(y -> y < q10 || y > q90, bin_preds)
        append!(x_outliers, fill(midpt, length(outlier_ind)))
        append!(y_outliers, bin_preds[outlier_ind])
    end
end

midpts_plot = collect(keys(stats))
q10s = [s.q10 for s in values(stats)]
q25s = [s.q25 for s in values(stats)]
q50s = [s.q50 for s in values(stats)]
q75s = [s.q75 for s in values(stats)]
q90s = [s.q90 for s in values(stats)] 

begin
    # setup
    fig_boxhist = Figure(size = (800, 800))
    ax_box = Axis(fig_boxhist[1, 1],
        xlabel="",
        ylabel="predicted embedding value",
        title="predicted vs. true embedding values"
    )
    ax_hist = Axis(fig_boxhist[2, 1],
        xlabel="true embedding value",
        ylabel="count",
        title="distribution of true embedding values",
        # xticks = 0:5:15,
    )
    linkxaxes!(ax_box, ax_hist)

    # boxplot
    scatter!(ax_box, x_outliers, y_outliers, markersize = 5, alpha = 0.5)
    rangebars!(ax_box, midpts_plot, q10s, q25s, color = :black, whiskerwidth = 0.5)
    rangebars!(ax_box, midpts_plot, q75s, q90s, color = :black, whiskerwidth = 0.5)
    boxplot!(ax_box, grouped_trues_midpts, grouped_preds, range = false, whiskerlinewidth = 0, show_outliers = false, width = 0.05)

    # histogram
    hist!(ax_hist, all_trues, bins = bin_edges, strokecolor = :black, strokewidth = 1)
    rowgap!(fig_boxhist.layout, 1, 10)
    # display(fig_boxhist)
    save(joinpath(save_dir, "box_hist.png"), fig_boxhist)
end

# save_dir = "/home/golem/scratch/chans/lincsv2/plots/trt/exp_nn/2025-09-30_11-58"
# save(joinpath(save_dir, "box_hist.png"), fig_boxhist)

### plot hexbin
fig_hex = Figure(size = (800, 600))
ax_hex = Axis(fig_hex[1, 1],
    xlabel="true embedding val",
    ylabel="predicted embedding val",
    title="predicted vs. true embedding density"
    # aspect=DataAspect() 
)
hexplot = hexbin!(ax_hex, all_trues, all_preds)
Colorbar(fig_hex[1, 2], hexplot, label="point count")
# display(fig_hex)
save(joinpath(save_dir, "hexbin.png"), fig_hex)

### error by gene analysis
absolute_errors = abs.(all_trues .- all_preds)
df_gene_errors = DataFrame(gene_index = all_gene_indices, absolute_error = absolute_errors)

# for all pred errors
begin
    fig_gene_error_scatter = Figure(size = (800, 600))
    ax_gene_error_scatter = Axis(fig_gene_error_scatter[1, 1], title = "prediction error by gene", xlabel = "gene index", ylabel = "absolute prediction error")
    scatter!(ax_gene_error_scatter, all_gene_indices, absolute_errors, alpha=0.5)
    display(fig_gene_error_scatter)
    save(joinpath(save_dir, "gene_vs_error_scatter.png"), fig_gene_error_scatter)
end

# for mean pred error
df_mean_error = combine(groupby(df_gene_errors, :gene_index), :absolute_error => mean => :mean_absolute_error)
begin
    fig_gene_meanerror = Figure(size = (800, 600))
    ax_gene_meanerror= Axis(fig_gene_meanerror[1, 1], title = "mean absolute error by gene", xlabel = "gene index", ylabel = "mean error")
    scatter!(ax_gene_meanerror, df_mean_error.gene_index, df_mean_error.mean_absolute_error, alpha=0.5)
    display(fig_gene_meanerror)
    save(joinpath(save_dir, "gene_vs_meanerror.png"), fig_gene_meanerror)
end

# for sorted indices
sorted_indices_by_mean = load("/home/golem/scratch/chans/lincs/plots/trt_and_untrt/infographs/sorted_gene_indices_by_exp.jld2")["sorted_indices_by_mean"]
error_map = Dict(row.gene_index => row.mean_absolute_error for row in eachrow(df_mean_error))
sorted_mean_errors = [get(error_map, idx, 0) for idx in sorted_indices_by_mean]
gene_ranks = 1:length(sorted_indices_by_mean)

begin
    fig_sort_error = Figure(size = (800, 600))
    ax_sort_error = Axis(fig_sort_error[1, 1], xlabel = "sorted gene index", ylabel = "mean error", title = "mean absolute error by sorted gene")
    scatter!(ax_sort_error, gene_ranks, sorted_mean_errors, alpha = 0.5)
    display(fig_sort_error)
    save(joinpath(save_dir, "sorted_gene_vs_meanerror.png"), fig_sort_error)
end

#######################################################################################################################################
### LOG
#######################################################################################################################################

# save model!!!
model_cpu = cpu(model)
jldsave(joinpath(save_dir, "model_object.jld2"); model=model_cpu) # whole model
jldsave(joinpath(save_dir, "model_state.jld2"); model_state=Flux.state(model_cpu)) # need to recreate model + apply state to load back in

jldsave(joinpath(save_dir, "losses.jld2"); 
    epochs = 1:n_epochs, 
    train_losses = train_losses, 
    test_losses = test_losses
)
jldsave(joinpath(save_dir, "predstrues.jld2"); 
    all_preds = all_preds, 
    all_trues = all_trues,
    all_gene_indices = all_gene_indices
)
jldsave(joinpath(save_dir, "test_data.jld2"); X=X_test)

end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

params_txt = joinpath(save_dir, "params.txt")
open(params_txt, "w") do io
    println(io, "PARAMETERS:")
    println(io, "########### $(gpu_info)")
    println(io, "dataset = $(dataset_note)")
    println(io, "masking_ratio = $mask_ratio")
    println(io, "mask_value = $mask_value")
    println(io, "batch_size = $batch_size")
    println(io, "n_epochs = $n_epochs")
    println(io, "embed_dim = $embed_dim")
    println(io, "hidden_dim = $hidden_dim")
    println(io, "mid_dim = $mid_dim")
    println(io, "latent_dim = $latent_dim")
    println(io, "learning_rate = $lr")
    # println(io, "dropout_probability = $drop_prob")
    println(io, "ADDITIONAL NOTES: $(additional_notes)")
    println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
    println(io, "correlation = $correlation")
    # println(io, "mse model = $mse_model")
    # println(io, "mse baseline = $mse_baseline")
end