#=
THIS FILE CALCULATES ERROR INSTEAD OF ACCURACY
=#

using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2, SparseArrays, Dates, Printf, Profile
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra

CUDA.device!(0)

#######################################################################################################################################
### INFO
#######################################################################################################################################

# untrt only/
const data_path = "data/lincs_untrt_data.jld2"
const save_path_base = "untrt"

# untrt and trt
# const data_path = "data/lincs_trt_untrt_data.jld2"
# const save_path_base = "trt"

# params
const batch_size = 128
const n_epochs = 30
const embed_dim = 128
const hidden_dim = 256
const n_heads = 2
const n_layers = 4
const drop_prob = 0.05
const lr = 0.001
const mask_ratio = 0.1

# notes
const gpu_info = "this was on smaug"
const dataset_note = "untrt"
const additional_notes = "testing no posenc"

#######################################################################################################################################
### DATA
#######################################################################################################################################

start_time = now()

data = load(data_path)["filtered_data"]

### tokenization (row 1 is ranks of gene 1 in each sample)
function sort_gene(expr)
    n, m = size(expr)
    data_ranked = Matrix{Int}(undef, size(expr)) # faster than fill(-1, size(expr))
    sorted_ind_col = Vector{Int}(undef, n)
    for j in 1:m
        unsorted_expr_col = view(expr, :, j)
        sortperm!(sorted_ind_col, unsorted_expr_col, rev=true)
            # rev=true -> data[1, :] = index (into gene.expr) of highest expression value in experiment/column 1
        for i in 1:n
            data_ranked[i, j] = sorted_ind_col[i]
        end
    end
    return data_ranked
end

@time X = sort_gene(data.expr) # lookup table of indices from highest rank to lowest rank gene

const n_features = size(X, 1) + 2
const n_classes = size(X, 1)
const n_genes = size(X, 1)
const MASK_ID = (n_classes + 1)
const CLS_ID = n_genes + 2
const CLS_VECTOR = fill(CLS_ID, (1, size(X, 2)))
X = vcat(CLS_VECTOR, X)


#######################################################################################################################################
### MODEL
#######################################################################################################################################

# so we can use GPU or CPU :D
const IntMatrix2DType = Union{Array{Int}, CuArray{Int, 2}}
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
const Float32Matrix3DType = Union{Array{Float32, 3}, CuArray{Float32, 3}}

### positional encoder
struct PosEnc
    pe_matrix::CuArray{Float32,2}
end

# function PosEnc(embed_dim::Int, max_len::Int) # max_len is usually maximum length of sequence but here it is just len(genes)
#     pe_matrix = Matrix{Float32}(undef, embed_dim, max_len)
#     for pos in 1:max_len, i in 1:embed_dim
#         angle = pos / (10000^(2*(div(i-1,2))/embed_dim))
#         if mod(i, 2) == 1
#             pe_matrix[i,pos] = sin(angle) # odd indices
#         else
#             pe_matrix[i,pos] = cos(angle) # even indices
#         end
#     end
#     return PosEnc(cu(pe_matrix))
# end

function PosEnc(embed_dim::Int, max_len::Int)
    pe_matrix = zeros(Float32, embed_dim, max_len) 
    return PosEnc(cu(pe_matrix))
end

Flux.@functor PosEnc

function (pe::PosEnc)(input::Float32Matrix3DType)
    seq_len = size(input,2)
    return input .+ pe.pe_matrix[:,1:seq_len] # adds positional encoding to input embeddings
end

### building transformer section
struct Transf
    mha::Flux.MultiHeadAttention
    att_dropout::Flux.Dropout
    att_norm::Flux.LayerNorm # this is the normalization aspect
    mlp::Flux.Chain
    mlp_norm::Flux.LayerNorm
end

function Transf(
    embed_dim::Int, 
    hidden_dim::Int; 
    n_heads::Int, 
    dropout_prob::Float64
    )

    mha = Flux.MultiHeadAttention((embed_dim, embed_dim, embed_dim) => (embed_dim, embed_dim) => embed_dim, 
                                    nheads=n_heads, 
                                    dropout_prob=dropout_prob
                                    )
    att_dropout = Flux.Dropout(dropout_prob)
    att_norm = Flux.LayerNorm(embed_dim)
    mlp = Flux.Chain(
        Flux.Dense(embed_dim => hidden_dim, gelu),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => embed_dim),
        Flux.Dropout(dropout_prob)
        )
    mlp_norm = Flux.LayerNorm(embed_dim)
    return Transf(mha, att_dropout, att_norm, mlp, mlp_norm)
end

Flux.@functor Transf

function (tf::Transf)(input::Float32Matrix3DType) # input shape: embed_dim × seq_len × batch_size
    normed = tf.att_norm(input)
    atted = tf.mha(normed, normed, normed)[1] # outputs a tuple (a, b)
    att_dropped = tf.att_dropout(atted)
    residualed = input + att_dropped
    res_normed = tf.mlp_norm(residualed)
    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size) # dense layers expect 2D inputs
    mlp_out = tf.mlp(reshaped)
    mlp_out_reshaped = reshape(mlp_out, embed_dim, seq_len, batch_size)
    tf_output = residualed + mlp_out_reshaped
    return tf_output
end

### full model as << ranked data --> token embedding --> position embedding --> transformer --> classifier head >>
struct Model
    embedding::Flux.Embedding
    pos_encoder::PosEnc
    pos_dropout::Flux.Dropout
    transformer::Flux.Chain
    classifier::Flux.Chain
end

function Model(;
    input_size::Int,
    embed_dim::Int,
    n_layers::Int,
    n_classes::Int,
    n_heads::Int,
    hidden_dim::Int,
    dropout_prob::Float64
    )

    embedding = Flux.Embedding(input_size => embed_dim)
    pos_encoder = PosEnc(embed_dim, input_size)
    pos_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain(
        [Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...
        )
    classifier = Flux.Chain(
        Flux.Dense(embed_dim => embed_dim, gelu),
        Flux.LayerNorm(embed_dim),
        Flux.Dense(embed_dim => n_classes)
        )
    return Model(embedding, pos_encoder, pos_dropout, transformer, classifier)
end

Flux.@functor Model

function (model::Model)(input::IntMatrix2DType)
    embedded = model.embedding(input)
    encoded = model.pos_encoder(embedded)
    encoded_dropped = model.pos_dropout(encoded)
    transformed = model.transformer(encoded_dropped)
    logits_output = model.classifier(transformed)
    return logits_output
end

#######################################################################################################################################
### DATA PREP
#######################################################################################################################################

### splitting data
function split_data(X, test_ratio::Float64, y=nothing) # masking doesn't need y!
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

### masking data
# function mask_input(X::Matrix{Int64}; mask_ratio=mask_ratio)
#     X_masked = copy(X)
#     mask_labels = fill((-100), size(X)) # -100 = ignore, this is not masked
#     for j in 1:size(X,2) # per column
#         num_masked = ceil(Int, size(X,1) * mask_ratio)
#         mask_positions = randperm(size(X,1))[1:num_masked]
#         for pos in mask_positions
#             mask_labels[pos, j] = X[pos, j] # original label
#             X_masked[pos, j] = MASK_ID # mask label
#         end
#     end
#     return X_masked, mask_labels
# end

function mask_input(X::Matrix{Int}; mask_ratio=mask_ratio)
    X_masked = copy(X)
    mask_labels = fill((-100), size(X)) # -100 = ignore, this is not masked
    for j in 1:size(X,2) # per column, start at second row so we don't mask CLS token
        num_masked = ceil(Int, (size(X,1) - 1) * mask_ratio)
        mask_positions_local = randperm(size(X,1) - 1)[1:num_masked]
        mask_positions_global = mask_positions_local .+ 1 # also shifted for CLS token
        
        for pos in mask_positions_global
            mask_labels[pos, j] = X[pos, j] # original label
            X_masked[pos, j] = MASK_ID # mask label
        end
    end
    return X_masked, mask_labels
end

X_train_masked, y_train_masked = mask_input(X_train)
X_test_masked, y_test_masked = mask_input(X_test)

#######################################################################################################################################
### TRAINING
#######################################################################################################################################

model = Model(
    input_size=n_features,
    embed_dim=embed_dim,
    n_layers=n_layers,
    n_classes=n_classes,
    n_heads=n_heads,
    hidden_dim=hidden_dim,
    dropout_prob=drop_prob
) |> gpu

opt = Flux.setup(Adam(lr), model)

#=
loss: cross-entropy between the model’s predicted distribution and the true token at each masked position
compute the loss by iterating over masked positions OR by using a mask in the loss function
=#
function loss(model::Model, x, y, mode::String)
    logits = model(x)  # (n_classes, seq_len, batch_size)
    logits_flat = reshape(logits, size(logits, 1), :) # (n_classes, seq_len*batch_size)
    y_flat = vec(y) # (seq_len*batch_size) column vec
    mask = y_flat .!= -100 # bit vec, where sum = n_masked
    logits_masked = logits_flat[:, mask] # (n_classes, n_masked)
    y_masked = y_flat[mask] # (n_masked) column vec
    y_oh = Flux.onehotbatch(y_masked, 1:n_classes) # (n_classes, n_masked)

    if mode == "train"
        return Flux.logitcrossentropy(logits_masked, y_oh) 
    end
    if mode == "test"
        return Flux.logitcrossentropy(logits_masked, y_oh), logits_masked, y_masked
    end
end

train_losses = Float32[]
test_losses = Float32[]
test_rank_errors = Float32[]

# collect stuff
all_preds = Int[]
all_trues = Int[]
all_original_ranks = Int[]
all_prediction_errors = Int[]

for epoch in ProgressBar(1:n_epochs)
    epoch_losses = Float32[]
    for start_idx in 1:batch_size:size(X_train_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_train_masked, 2))
        x_gpu = gpu(X_train_masked[:, start_idx:end_idx])
        y_gpu = gpu(y_train_masked[:, start_idx:end_idx])
        
        loss_val, grads = Flux.withgradient(model) do m
            loss(m, x_gpu, y_gpu, "train")
        end
        Flux.update!(opt, model, grads[1])
        loss_val = loss(model, x_gpu, y_gpu, "train")
        push!(epoch_losses, loss_val)
    end
    push!(train_losses, mean(epoch_losses))

    test_epoch_losses = Float32[]
    epoch_rank_errors = Int[]
    for start_idx in 1:batch_size:size(X_test_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
        x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
        y_gpu = gpu(y_test_masked[:, start_idx:end_idx])

        test_loss_val, logits_masked, y_masked = loss(model, x_gpu, y_gpu, "test")
        push!(test_epoch_losses, test_loss_val)

        if isempty(y_masked) continue end

        logits_cpu = cpu(logits_masked)
        y_cpu = cpu(y_masked)
        
        if epoch == n_epochs
            y_cpu_batch = cpu(y_gpu)
            masked_indices_cartesian = findall(y_cpu_batch .!= -100)
            original_ranks_in_batch = [idx[1] for idx in masked_indices_cartesian]
        end

        for i in 1:length(y_cpu)
            true_gene_id = y_cpu[i]
            prediction_logits = logits_cpu[:, i]
            ranked_gene_ids = sortperm(prediction_logits, rev=true)
            predicted_rank = findfirst(isequal(true_gene_id), ranked_gene_ids)
            
            if !isnothing(predicted_rank)
                error = predicted_rank - 1
                push!(epoch_rank_errors, error)
                
                if epoch == n_epochs
                    original_rank = original_ranks_in_batch[i] - 1
                    push!(all_original_ranks, original_rank)
                    push!(all_prediction_errors, error)
                end
            end
        end

        if epoch == n_epochs
            predicted_ids = Flux.onecold(logits_masked)
            append!(all_preds, cpu(predicted_ids))
            append!(all_trues, y_cpu)
        end
    end
    push!(test_losses, mean(test_epoch_losses))
    if !isempty(epoch_rank_errors)
        push!(test_rank_errors, mean(epoch_rank_errors))
    else
        push!(test_rank_errors, NaN32)
    end
end

#######################################################################################################################################
### EVAL/PLOT
#######################################################################################################################################

# mk dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", save_path_base, "rank_tf", timestamp)
mkpath(save_dir)

# loss plot
fig_loss = Figure(size = (800, 600))
ax_loss = Axis(fig_loss[1, 1], xlabel="epoch", ylabel="loss (logit-ce)", title="train vs. test loss")
lines!(ax_loss, 1:n_epochs, train_losses, label="train loss", linewidth=2)
lines!(ax_loss, 1:n_epochs, test_losses, label="test loss", linewidth=2)
axislegend(ax_loss, position=:rt)
save(joinpath(save_dir, "loss.png"), fig_loss)

# rank error plot
fig_err = Figure(size = (800, 600))
ax_err = Axis(fig_err[1, 1], xlabel="epoch", ylabel="error", title="mean rank errors")
lines!(ax_err, 1:n_epochs, test_rank_errors, label="test error", linewidth=2)
save(joinpath(save_dir, "error.png"), fig_err)

# boxplots
bin_size = 50
bin_edges = collect(1:bin_size:n_classes)
if bin_edges[end] < n_classes
    push!(bin_edges, n_classes + 1)
end
bin_midpts = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2
grouped_preds = Int[]
grouped_trues_midpts = Float64[]
for i in 1:length(bin_edges)-1
    indices = findall(x -> bin_edges[i] <= x < bin_edges[i+1], all_trues)
    if !isempty(indices)
        preds_in_bin = all_preds[indices]
        midpoint = bin_midpts[i]
        append!(grouped_preds, preds_in_bin)
        append!(grouped_trues_midpts, fill(midpoint, length(preds_in_bin)))
    end
end
fig_box = Figure(size = (800, 600))
ax_box = Axis(fig_box[1, 1], xlabel="true gene id", ylabel="predicted gene id", title="predicted vs true gene ids")
boxplot!(ax_box, grouped_trues_midpts, grouped_preds, width=bin_size*0.5)
save(joinpath(save_dir, "boxplot.png"), fig_box)

# hexbin
fig_hex = Figure(size = (800, 700))
ax_hex = Axis(fig_hex[1, 1], xlabel="true gene id", ylabel="predicted gene id", title="predicted vs true gene id density", aspect=DataAspect())
hexplot = hexbin!(ax_hex, all_trues, all_preds)
Colorbar(fig_hex[1, 2], hexplot, label="point count")
save(joinpath(save_dir, "hexbin.png"), fig_hex)

# MASKED PREDICTION ERROR BY RANK (SCATTER)
fig_rank_error_scatter = Figure(size = (800, 600))
ax_rank_error_scatter = Axis(fig_rank_error_scatter[1, 1], xlabel = "rank (1 = highest exp)", ylabel = "prediction error", title = "prediction error by rank")
scatter!(ax_rank_error_scatter, all_original_ranks, all_prediction_errors, markersize=4, alpha=0.3)
save(joinpath(save_dir, "rank_vs_error_scatter.png"), fig_rank_error_scatter)

# MEAN MASKED PREDICTION ERROR BY RANK (SCATTER)
df_rank_errors = DataFrame(original_rank = all_original_ranks, prediction_error = all_prediction_errors)
avg_errors = combine(groupby(df_rank_errors, :original_rank), :prediction_error => mean => :avg_error)
fig_rank_error_line = Figure(size = (800, 600))
ax_rank_error_line = Axis(fig_rank_error_line[1, 1], xlabel = "rank (1 = highest exp)", ylabel = "mean prediction error", title = "mean prediction error by rank")
scatter!(ax_rank_error_line, avg_errors.original_rank, avg_errors.avg_error, alpha = 0.5)
save(joinpath(save_dir, "rank_vs_avgerror_scatter.png"), fig_rank_error_line)

#######################################################################################################################################
### LOG
#######################################################################################################################################

# log model
model_cpu = cpu(model)
jldsave(joinpath(save_dir, "model_object.jld2"); model=model_cpu)
jldsave(joinpath(save_dir, "model_state.jld2"); model_state=Flux.state(model_cpu))

# # get profile embeddings
# function get_profile_embedding(model::Model, input::IntMatrix2DType)
#     embedded = model.embedding(input)
#     encoded = model.pos_encoder(embedded)
#     encoded_dropped = model.pos_dropout(encoded)
#     transformed = model.transformer(encoded_dropped)
#     cls_embedding = transformed[:, 1, :]
#     return cls_embedding
# end

# Flux.testmode!(model)
# input_batch = gpu(X_test[:, 1:batch_size])
# profile_embeddings = get_profile_embedding(model, input_batch) # expected (128, 128) for a batch_size of 128

model_cpu = cpu(model)
model_state = Flux.state(model_cpu)

function get_profile_embedding(m, input)
    transformed = m.transformer(m.pos_dropout(m.pos_encoder(m.embedding(input))))
    return transformed[:, 1, :] # Assuming [CLS] token is at position 1
end

all_embeddings = []
Flux.testmode!(model)
for start_idx in 1:batch_size:size(X, 2)
    end_idx = min(start_idx + batch_size - 1, size(X, 2))
    input_batch = gpu(X[:, start_idx:end_idx])
    batch_embeddings = cpu(get_profile_embedding(model, input_batch))
    push!(all_embeddings, batch_embeddings)
end
final_embeddings = hcat(all_embeddings...) 

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
jldsave(joinpath(save_dir, "profile_embeddings.jld2"); 
    profile_embeddings=final_embeddings
)
jldsave(joinpath(save_dir, "losses.jld2"); 
    epochs = 1:n_epochs, 
    train_losses = train_losses, 
    test_losses = test_losses
)
jldsave(joinpath(save_dir, "predstrues.jld2"); 
    all_preds = all_preds, 
    all_trues = all_trues
)
jldsave(joinpath(save_dir, "rank_vs_error.jld2"); 
    original_rank = all_original_ranks, 
    prediction_error = all_prediction_errors
)
jldsave(joinpath(save_dir, "avg_errors.jld2"); 
    original_rank = avg_errors.original_rank,
    avg_error = avg_errors.avg_error
)
# jldsave(joinpath(save_dir, "data_expr.jld2"); 
#     data_expr = data.expr
# )

# log run info
end_time = now()
run_time = end_time - start_time
total_minutes = div(run_time.value, 60000)
run_hours = div(total_minutes, 60)
run_minutes = rem(total_minutes, 60)

params_txt = joinpath(save_dir, "params.txt")
open(params_txt, "w") do io
    println(io, "PARAMETERS:")
    println(io, "########### $(gpu_info)")
    println(io, "dataset = $dataset_note")
    println(io, "masking_ratio = $mask_ratio")
    println(io, "NO DYNAMIC MASKING")
    println(io, "batch_size = $batch_size")
    println(io, "n_epochs = $n_epochs")
    println(io, "embed_dim = $embed_dim")
    println(io, "hidden_dim = $hidden_dim")
    println(io, "n_heads = $n_heads")
    println(io, "n_layers = $n_layers")
    println(io, "learning_rate = $lr")
    println(io, "dropout_probability = $drop_prob")
    println(io, "ADDITIONAL NOTES: $additional_notes")
    println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
end
