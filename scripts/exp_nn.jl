using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2, SparseArrays, Dates, Printf, Profile
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra

CUDA.device!(0)
# GC.gc()
# CUDA.reclaim()

#######################################################################################################################################
### INFO
#######################################################################################################################################

# untrt only
const data_path = "data/lincs_untrt_data.jld2"
const save_path_base = "untrt"

# untrt and trt
# const data_path = "data/lincs_trt_untrt_data.jld2"
# const save_path_base = "trt"

# params
const batch_size = 64
const n_epochs = 1
const embed_dim = 128
const hidden_dim = 256
const mid_dim = 64
const latent_dim = 32
const mask_ratio = 0.1
const mask_value = 0.0
const lr = 0.001
# const drop_prob = 0.05

# notes
const gpu_info = "this was on kraken"
const dataset_note = "untrt"
const additional_notes = "demo run"

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

function (m::Mask)(x::AbstractMatrix{Float32})
    X_masked = copy(x)
    mask_labels = fill(NaN32, size(x)) #!# NaN to ignore positions in the loss calculation

    for j in 1:size(x,2) # per column
        num_masked = ceil(Int, size(x,1) * m.mask_ratio)
        mask_positions = randperm(size(x,1))[1:num_masked]

        for pos in mask_positions
            mask_labels[pos, j] = x[pos, j] 
            X_masked[pos, j] = m.mask_value
        end
    end
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

Flux.@functor Encoder

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
        # layernorm?
        # dropout?
        Dense(hidden_dim => embed_dim) # softplus?
    )

    encoder = Encoder(embed_dim, mid_dim, latent_dim, mask_ratio, mask_value)

    decoder = Decoder(embed_dim, mid_dim, latent_dim)

    return Model(mlp, encoder, decoder)
end

Flux.@functor Model

function (model::Model)(x::AbstractMatrix{Float32})
    embedding = model.mlp(x)
    latent, labels = model.encoder(embedding)
    reconstructed = model.decoder(latent)
    return reconstructed, labels
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

function loss(model::Model, x, y)
    preds = model(x)
end