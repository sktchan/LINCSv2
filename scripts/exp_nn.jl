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
const additional_notes = "trying with trt and untrt data, 30ep"

# # untrt only
# data_path = "data/lincs_untrt_data.jld2"
# save_path_base = "untrt"

# # untrt and trt
# # data_path = "data/lincs_trt_untrt_data.jld2"
# # save_path_base = "trt"

# # params
# constbatch_size = 64
# n_epochs = 10
# embed_dim = 128
# hidden_dim = 256
# mid_dim = 64
# latent_dim = 32
# mask_ratio = 0.1f0
# mask_value = 0.0f0
# lr = 0.001
# # drop_prob = 0.05

# # notes
# gpu_info = "this was on kraken"
# dataset_note = "untrt"
# additional_notes = "demo run"

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

function (m::Mask)(x::AbstractMatrix{Float32})
    # random boolean mask for entire matrix for what to mask
    mask_indices = rand_like(x) .< m.mask_ratio

    # place mask on data
    X_masked = ifelse.(mask_indices, m.mask_value, x)
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
        Flux.Dense(latent_dim => mid_dim, relu),
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
        Dense(num_genes => hidden_dim),
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

# do i need to normalize?

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

# X_train_raw, X_test_raw = split_data(X, 0.2)

# train_mean = mean(X_train_raw, dims=2)
# train_std = std(X_train_raw, dims=2)
# epsilon = 1e-8

# X_train = (X_train_raw .- train_mean) ./ (train_std .+ epsilon)
# X_test = (X_test_raw .- train_mean) ./ (train_std .+ epsilon)

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
        return error, preds_masked, trues_masked
    end
end

train_losses = Float32[]
test_losses = Float32[]
all_preds = Float32[]
all_trues = Float32[]

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

        test_loss_val, preds_masked, trues_masked = loss(model, x_gpu, "test")
        push!(test_epoch_losses, test_loss_val)

        if epoch == n_epochs
            append!(all_preds, cpu(preds_masked))
            append!(all_trues, cpu(trues_masked))
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
    all_trues = all_trues
    # all_gene_indices = all_gene_indices
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