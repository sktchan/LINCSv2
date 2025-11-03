using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2, SparseArrays, Dates, Printf, Profile
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra, MLUtils

CUDA.device!(0)
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
const batch_size = 128
const n_epochs = 30
const embed_dim = 128
const latent_1 = 734
const latent_2 = 490
const latent_3 = 246
const mask_ratio = 0.1f0
const mask_value = 0.0f0 # should remain as this if we are masking the embedding vals!
const lr = 0.001
# const drop_prob = 0.05

# notes
const gpu_info = "this was on kraken"
const dataset_note = "trt"
const additional_notes = "rerun 30ep with fixed saving"

#######################################################################################################################################
### DATA
#######################################################################################################################################

start_time = now()

data = load(data_path)["filtered_data"]

### converting from genes x samples = exp --> genes x samples = rank (not rank x samples = gene)
function rank_gene(expr::AbstractMatrix)
    n, m = size(expr)
    data_ranked = Matrix{Float32}(undef, size(expr))
    rank_values = Statistics.LinRange(-1.0f0, 1.0f0, n)
    for j in 1:m
        p = sortperm(view(expr, :, j))
        for (i, original_index) in enumerate(p)
            data_ranked[original_index, j] = rank_values[i]
        end
    end
    return data_ranked
end

@time X = rank_gene(data.expr)

# ### tokenization (row 1 is ranks of gene 1 in each sample)
# function sort_gene(expr)
#     n, m = size(expr)
#     data_ranked = Matrix{Int}(undef, size(expr)) # faster than fill(-1, size(expr))
#     sorted_ind_col = Vector{Int}(undef, n)
#     for j in 1:m
#         unsorted_expr_col = view(expr, :, j)
#         sortperm!(sorted_ind_col, unsorted_expr_col, rev=true)
#             # rev=true -> data[1, :] = index (into gene.expr) of highest expression value in experiment/column 1
#         for i in 1:n
#             data_ranked[i, j] = sorted_ind_col[i]
#         end
#     end
#     return data_ranked
# end

# @time X = sort_gene(data.expr)

const n_genes = size(X, 1)
# const n_classes = 1 #!# n_classes is 1 for regression (n_features is not vocabulary size)

#######################################################################################################################################
### MODEL
#######################################################################################################################################

#=
PIPELINE:
Embedding Layer: Flux.Embedding --> embed into embed_dim vector (this is the only difference between rank and exp!)
Encoder: embed_dim vector --> masked -> compressed into latent_dim vector
Decoder: latent_dim vector --> reconstructed into embed_dim vector
evaluate on embed_dim vector similarity
later:
    FNN: embed_dim vector --> downstream task (single value for regression or n_classes vector for classification)
=#

# for use of both CPU and GPU
const IntMatrix2DType = Union{Array{Int}, CuArray{Int, 2}}
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
const Float32Matrix3DType = Union{Array{Float32}, CuArray{Float32, 3}}

### masking

# struct Mask
#     mask_ratio::Float32
#     mask_value::Float32
# end

# function (m::Mask)(x::Float32Matrix2DType)
#     # random boolean mask for entire matrix for what to mask
#     mask_indices = rand_like(x) .< m.mask_ratio

#     # place mask on data
#     X_masked = ifelse.(mask_indices, m.mask_value, x)
#     mask_labels = ifelse.(mask_indices, x, NaN32)

#     return X_masked, mask_labels
# end

# ### encoder

# struct Encoder
#     # noise::Mask
#     compress::Flux.Chain
# end

# function Encoder(
#     num_genes::Int,
#     embed_dim::Int,
#     latent_1::Int,
#     latent_2::Int,
#     latent_3::Int,
#     mask_ratio::Float32,
#     mask_value::Float32
#     )

#     # noise = Mask(mask_ratio, mask_value)

#     compress = Flux.Chain(
#         Flux.Dense(num_genes => latent_1, relu),
#         Flux.Dense(latent_1 => latent_2, relu),
#         Flux.Dense(latent_2 => latent_3, relu),
#         Flux.Dense(latent_3 => embed_dim)
#     )

#     return Encoder(compress)
# end

# Flux.@functor Encoder (compress,)

# function (enc::Encoder)(input::Float32Matrix2DType)
#     # noised, labels = enc.noise(input)
#     compressed = enc.compress(input)
#     return compressed
# end

# ### decoder

# struct Decoder
#     reconstruct::Flux.Chain
# end

# function Decoder(
#     embed_dim::Int,
#     num_genes::Int,
#     latent_1::Int,
#     latent_2::Int,
#     latent_3::Int,
#     )

#     reconstruct = Flux.Chain(
#         Flux.Dense(embed_dim => latent_3, relu),
#         Flux.Dense(latent_3 => latent_2, relu),
#         Flux.Dense(latent_2 => latent_1, relu),
#         Flux.Dense(latent_1 => num_genes)
#     )

#     return Decoder(reconstruct)
# end

# Flux.@functor Decoder

# function (dec::Decoder)(input::Float32Matrix2DType)
#     return dec.reconstruct(input)
# end

# ### full model

# struct Model
#     embedding::Flux.Embedding
#     norm::Flux.LayerNorm
#     encoder::Encoder
#     decoder::Decoder
#     mlp::Flux.Dense
#     num_genes::Int
#     embed_dim::Int
# end

# function Model(;
#     num_genes::Int,
#     embed_dim::Int,
#     latent_1::Int,
#     latent_2::Int,
#     latent_3::Int,
#     mask_ratio::Float32,
#     mask_value::Float32
#     # n_classes::Int,
#     # dropout_prob::Float64
#     )

#     # mlp = Flux.Chain(
#     #     Dense(num_genes => hidden_dim),
#     #     LayerNorm(hidden_dim),
#     #     relu,
#     #     # Dropout(0.1),
#     #     Dense(hidden_dim => embed_dim) # softplus?
#     # )

#     input_size = num_genes + 1 # incl mask token

#     embedding = Flux.Embedding(input_size => embed_dim)

#     norm = Flux.LayerNorm(embed_dim)

#     flat_dim = num_genes * embed_dim
#     encoder = Encoder(flat_dim, embed_dim, latent_1, latent_2, latent_3, mask_ratio, mask_value)

#     decoder = Decoder(embed_dim, flat_dim, latent_1, latent_2, latent_3)

#     mlp = Flux.Dense(embed_dim => num_genes)

#     return Model(embedding, norm, encoder, decoder, mlp, num_genes, embed_dim)
# end

# Flux.@functor Model

# function (model::Model)(input::IntMatrix2DType)
#     # embedded = model.embedding(input) # (embed_dim, num_genes, batch_size)
#     # # embedded_flat = Flux.flatten(embedded) # flatten into (embed_dim * num_genes, batch_size) OR change encoder input to Float32Matrix3DType?
#     # pooled = mean(embedded, dims=2) # mean pool into (128, 1, batch_size) for input into encoder... is this proper?
#     # final_emb = dropdims(pooled, dims=2) # (128, batch_size)

#     # normed = model.norm(final_emb) # normalize here?
    
#     # latent, labels = model.encoder(normed)
#     # recon_embed = model.decoder(latent)
#     # return recon_embed, labels

#     embedded = model.embedding(input) # (embed_dim, num_genes, batch_size)
#     normed = model.norm(embedded) # same
    
#     flattened_input = Flux.flatten(normed) # (embed_dim * num_genes, batch_size) for input into encoder
    
#     latent = model.encoder(flattened_input)
#     recon_flat = model.decoder(latent)

#     recon_embed = reshape(recon_flat, model.embed_dim,  model.num_genes, :) # (embed_dim, num_genes, batch_size)
#     logits = model.mlp(recon_embed) # (num_genes, num_genes, batch_size)
#     return logits
# end

# below is pasted fro exp_nn.jl

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
    num_genes::Int,
    embed_dim::Int,
    latent_1::Int,
    latent_2::Int,
    latent_3::Int,
    mask_ratio::Float32,
    mask_value::Float32
    )

    noise = Mask(mask_ratio, mask_value)

    compress = Flux.Chain(
        Flux.Dense(num_genes => latent_1, relu),
        Flux.Dense(latent_1 => latent_2, relu),
        Flux.Dense(latent_2 => latent_3, relu),
        Flux.Dense(latent_3 => embed_dim)
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
    num_genes::Int,
    latent_1::Int,
    latent_2::Int,
    latent_3::Int,
    )

    reconstruct = Flux.Chain(
        Flux.Dense(embed_dim => latent_3, relu),
        Flux.Dense(latent_3 => latent_2, relu),
        Flux.Dense(latent_2 => latent_1, relu),
        Flux.Dense(latent_1 => num_genes)
    )

    return Decoder(reconstruct)
end

Flux.@functor Decoder

function (dec::Decoder)(input::Float32Matrix2DType)
    return dec.reconstruct(input)
end

### full model

struct Model
    # mlp::Flux.Chain
    encoder::Encoder
    decoder::Decoder
    # mlp_head::Flux.Chain
end

function Model(;
    num_genes::Int,
    embed_dim::Int,
    latent_1::Int,
    latent_2::Int,
    latent_3::Int,
    mask_ratio::Float32,
    mask_value::Float32
    )

    # mlp = Flux.Chain(
    #     Dense(num_genes => hidden_dim, relu),
    #     LayerNorm(hidden_dim),
    #     relu,
    #     # Dropout(0.1),
    #     Dense(hidden_dim => embed_dim) # softplus?
    # )

    encoder = Encoder(num_genes, embed_dim, latent_1, latent_2, latent_3, mask_ratio, mask_value)

    decoder = Decoder(embed_dim, num_genes, latent_1, latent_2, latent_3)

    return Model(encoder, decoder)
end

Flux.@functor Model

function (model::Model)(input::AbstractMatrix{Float32})
    # embedding = model.mlp(input)
    latent, labels = model.encoder(input)
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
        return X_train, X_test, train_indices, test_indices
    else
        y_train = y[train_indices]
        y_test = y[test_indices]
        return X_train, y_train, X_test, y_test, train_indices, test_indices
    end
end

X_train, X_test, train_indices, test_indices = split_data(X, 0.2)

# const MASK_ID = n_genes + 1

# function mask_input(X::Matrix{Int}; mask_ratio=mask_ratio)
#     X_masked = copy(X)
#     mask_labels = fill(-100, size(X))
    
#     for j in 1:size(X, 2)
#         num_masked = ceil(Int, size(X, 1) * mask_ratio)
#         mask_positions = randperm(size(X, 1))[1:num_masked]

#         for pos in mask_positions
#             mask_labels[pos, j] = X[pos, j]
#             X_masked[pos, j] = MASK_ID
#         end
#     end
#     return X_masked, mask_labels
# end

# X_train_masked, y_train_labels = mask_input(X_train; mask_ratio=mask_ratio)
# X_test_masked, y_test_labels = mask_input(X_test; mask_ratio=mask_ratio)


######################################################################################################################################
### TRAINING
#######################################################################################################################################

model = Model(
    num_genes=n_genes,
    embed_dim=embed_dim,
    latent_1=latent_1,
    latent_2=latent_2,
    latent_3=latent_3,
    mask_ratio=mask_ratio,
    mask_value=mask_value
) |> gpu

opt = Flux.setup(Adam(lr), model)

# function loss(model::Model, x, mode::String)
#     preds, trues = model(x)
#     preds_flat = vec(preds)
#     trues_flat = vec(trues)

#     mask = .!isnan.(trues_flat)
    
#     preds_masked = preds_flat[mask]
#     trues_masked = trues_flat[mask]
    
#     error = Flux.mse(preds_masked, trues_masked)

#     if mode == "train"
#         return error
#     end
#     if mode == "test"
#         return error, preds_masked, trues_masked, trues
#     end
# end

# function loss(model::Model, x, y, mode::String)
#     logits = model(x)
#     logits_flat = reshape(logits, size(logits, 1), :)
#     y_flat = vec(y)
    
#     mask = y_flat .!= -100
    
#     logits_masked = logits_flat[:, mask]
#     y_masked = y_flat[mask]
#     y_oh = Flux.onehotbatch(y_masked, 1:n_genes)

#     error = Flux.logitcrossentropy(logits_masked, y_oh)

#     if mode == "train"
#         return error 
#     end
#     if mode == "test"
#         return error, logits_masked, y_masked
#     end
# end

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

# train_losses = Float32[]
# test_losses = Float32[]
# all_preds = Int[]
# all_trues = Int[]
# all_gene_indices = Int[]

# for epoch in ProgressBar(1:n_epochs)
#     train_epoch_losses = Float32[]
#     for start_idx in 1:batch_size:size(X_train, 2)
#         end_idx = min(start_idx + batch_size - 1, size(X_train, 2))
#         # x_gpu = gpu(X_train[:, start_idx:end_idx])
#         x_gpu = gpu(X_train_masked[:, start_idx:end_idx])
#         y_gpu = gpu(y_train_labels[:, start_idx:end_idx])

#         loss_val, grads = Flux.withgradient(model) do m
#             loss(m, x_gpu, y_gpu, "train")
#         end
#         Flux.update!(opt, model, grads[1])
#         loss_val = loss(model, x_gpu, y_gpu, "train")
#         push!(train_epoch_losses, loss_val)
#     end
#     push!(train_losses, mean(train_epoch_losses))

#     test_epoch_losses = Float32[]
#     for start_idx in 1:batch_size:size(X_test, 2)
#         end_idx = min(start_idx + batch_size - 1, size(X_test, 2))
#         # x_gpu = gpu(X_test[:, start_idx:end_idx])
#         x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
#         y_gpu = gpu(y_test_labels[:, start_idx:end_idx])

#         test_loss_val, preds_masked, trues_masked = loss(model, x_gpu, y_gpu, "test")
#         push!(test_epoch_losses, test_loss_val)

#         if epoch == n_epochs
#             final_preds = vec(argmax(preds_masked, dims=1))
#             append!(all_preds, cpu(final_preds))
#             append!(all_trues, cpu(trues_masked))
#             # trues_cpu = cpu(trues_full)
#             # masked_indices = findall(!isnan, trues_cpu)
#             # batch_gene_indices = [idx[1] for idx in masked_indices]

#             masked_indices_in_batch = findall(cpu(y_gpu) .!= -100)
#             batch_gene_indices = [idx[1] for idx in masked_indices_in_batch]
#             append!(all_gene_indices, batch_gene_indices)
#         end
#     end
#     push!(test_losses, mean(test_epoch_losses))
# end

# correlation = cor(all_trues, all_preds)

train_losses = Float32[]
test_losses = Float32[]
all_preds = Float32[]
all_trues = Float32[]
all_gene_indices = Int[]
all_column_indices = Int[]

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

            batch_col_indices = start_idx:end_idx
            pred_col_indices = [batch_col_indices[idx[2]] for idx in masked_indices]
            append!(all_column_indices, pred_col_indices)
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
save_dir = joinpath("plots", save_path_base, "rank_nn", timestamp)
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
    boxplot!(ax_box, grouped_trues_midpts, grouped_preds, range = false, whiskerlinewidth = 0, show_outliers = false, width = 0.2)

    # histogram
    hist!(ax_hist, all_trues, bins = bin_edges, strokecolor = :black, strokewidth = 1)
    rowgap!(fig_boxhist.layout, 1, 10)
    # display(fig_boxhist)
    save(joinpath(save_dir, "box_hist.png"), fig_boxhist)
end

# save_dir = "/home/golem/scratch/chans/lincsv2/plots/trt/rank_nn/2025-09-30_11-58"
# save(joinpath(save_dir, "box_hist.png"), fig_boxhist)

# ### plot hexbin
# fig_hex = Figure(size = (800, 600))
# ax_hex = Axis(fig_hex[1, 1],
#     xlabel="true embedding val",
#     ylabel="predicted embedding val",
#     title="predicted vs. true embedding density"
#     # aspect=DataAspect() 
# )
# hexplot = hexbin!(ax_hex, all_trues, all_preds)
# Colorbar(fig_hex[1, 2], hexplot, label="point count")
# # display(fig_hex)
# save(joinpath(save_dir, "hexbin.png"), fig_hex)

### plot hexbin
begin
    fig_hex = Figure(size = (800, 600))
    ax_hex = Axis(fig_hex[1, 1],
        # backgroundcolor = to_colormap(:viridis)[1], 
        xlabel="true expression val",
        ylabel="predicted expression val",
        title="predicted vs. true expression density"
        # aspect=DataAspect() 
    )
    hexplot = hexbin!(ax_hex, all_trues, all_preds, cellsize = (0.1,0.1), colorscale = log10)
    text!(ax_hex, -1, 1, align = (:left, :top), text = "Pearson: $correlation")
    Colorbar(fig_hex[1, 2], hexplot, label="point count (log10)")
    display(fig_hex)
end
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

### to convert back into ranks for evaluation
function convert_to_rank(values, ref)
    combined = vcat(values, ref)
    p = sortperm(combined, rev=true)
    ranks = invperm(p)
    return ranks[1]
end

reference_matrix = X_test 
ranked_preds = similar(all_preds, Int)
ranked_trues = similar(all_trues, Int)

for i in 1:length(all_preds)
    pred = all_preds[i]
    true_val = all_trues[i]
    col_idx = all_column_indices[i]
    reference_col = reference_matrix[:, col_idx]
    ranked_preds[i] = convert_to_rank(pred, reference_col)
    ranked_trues[i] = convert_to_rank(true_val, reference_col)
end

# heatmap
bin_edges = 1:979 
h = fit(Histogram, (ranked_trues, ranked_preds), (bin_edges, bin_edges))

fig_hm = Figure(size = (800, 700))
ax_hm = Axis(fig_hm[1, 1],
    xlabel = "true rank",
    ylabel = "predicted rank"
)

log10_weights = log10.(h.weights .+ 1)
hm = heatmap!(ax_hm, h.edges[1], h.edges[2], log10_weights)
Colorbar(fig_hm[1, 2], hm, label = "count (log10)")
# display(fig)
save(joinpath(save_dir, "heatmap.png"), fig_hm)

#######################################################################################################################################
### LOG
#######################################################################################################################################

# save model!!!
model_cpu = cpu(model)
model_state = Flux.state(model_cpu)
jldsave(joinpath(save_dir, "model_state.jld2"); 
    model_state=model_state
)
jldsave(joinpath(save_dir, "model_object.jld2"); 
    model=model_cpu
)
jldsave(joinpath(save_dir, "indices.jld2"); 
    train_indices=train_indices, 
    test_indices=test_indices
)
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
jldsave(joinpath(save_dir, "rankedpredstrues.jld2"); 
    ranked_preds = ranked_preds, 
    ranked_trues = ranked_trues
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
    println(io, "latent_1 = $latent_1")
    println(io, "latent_2 = $latent_2")
    println(io, "latent_3 = $latent_3")
    println(io, "learning_rate = $lr")
    # println(io, "dropout_probability = $drop_prob")
    println(io, "ADDITIONAL NOTES: $(additional_notes)")
    println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
    println(io, "correlation = $correlation")
    # println(io, "mse model = $mse_model")
    # println(io, "mse baseline = $mse_baseline")
end