using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using LincsProject, DataFrames, Dates, StatsBase, JLD2
using Flux, Random, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra

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
const hidden_dim = 256
const n_heads = 2
const n_layers = 4
const drop_prob = 0.05
const lr = 0.001
const mask_ratio = 0.1
const MASK_VALUE = -1.0f0

# notes
const gpu_info = "this was on smaug"
const dataset_note = "trt"
const additional_notes = "test for re-run w saving indices n converting to rank"


#######################################################################################################################################
### DATA
#######################################################################################################################################

start_time = now()

data = load(data_path)["filtered_data"]

@time X = data.expr #!# use raw expression values!!!

const n_genes = size(X, 1)
const n_classes = 1 #!# n_classes is 1 for regression (n_features is not vocabulary size)

#######################################################################################################################################
### MODEL
#######################################################################################################################################

# so we can use GPU or CPU :D
const IntMatrix2DType = Union{Array{Int}, CuArray{Int, 2}}
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
const Float32Matrix3DType = Union{Array{Float32}, CuArray{Float32, 3}}

### positional encoder

struct PosEnc
    pe_matrix::Float32Matrix2DType
end

#!# uses n_genes as max_len directly
function PosEnc(embed_dim::Int, max_len::Int) # max_len is number of genes
    pe_matrix = Matrix{Float32}(undef, embed_dim, max_len)
    for pos in 1:max_len, i in 1:embed_dim
        angle = pos / (10000^(2*(div(i-1,2))/embed_dim))
        if mod(i, 2) == 1
            pe_matrix[i,pos] = sin(angle) # odd indices
        else
            pe_matrix[i,pos] = cos(angle) # even indices
        end
    end
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

#!# full model for raw value regression

struct Model
    projection::Flux.Dense #!# replace embedding w/ dense layer for cont's input
    pos_encoder::PosEnc
    pos_dropout::Flux.Dropout
    transformer::Flux.Chain
    classifier::Flux.Chain
end

function Model(;
    seq_len::Int, #!# changed from input_size
    embed_dim::Int,
    n_layers::Int,
    n_classes::Int, #!# 1 for regression
    n_heads::Int,
    hidden_dim::Int,
    dropout_prob::Float64
    )

    #!# project the single raw expression value to the embedding dimension
    projection = Flux.Dense(1 => embed_dim)

    pos_encoder = PosEnc(embed_dim, seq_len)

    pos_dropout = Flux.Dropout(dropout_prob)

    transformer = Flux.Chain(
        [Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...
        )

    #!# classifier preds a singular cont's val
    classifier = Flux.Chain(
        Flux.Dense(embed_dim => embed_dim, gelu),
        Flux.LayerNorm(embed_dim),
        Flux.Dense(embed_dim => 1, softplus) #!# 1 value returned
        )

    return Model(projection, pos_encoder, pos_dropout, transformer, classifier)
end

Flux.@functor Model

function (model::Model)(input::Float32Matrix2DType)
    seq_len, batch_size = size(input)

    #!# reshape for dense projection: (seq_len, batch_size) -> (1, seq_len * batch_size)
    input_reshaped = reshape(input, 1, :)
    #!# output is (embed_dim, seq_len * batch_size) -> (embed_dim, seq_len, batch_size)
    embedded = reshape(model.projection(input_reshaped), :, seq_len, batch_size)
    
    encoded = model.pos_encoder(embedded)
    encoded_dropped = model.pos_dropout(encoded)
    transformed = model.transformer(encoded_dropped)
    
    regression_output = model.classifier(transformed)
    return regression_output
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
        return X_train, X_test, test_indices, train_indices
    else
        y_train = y[train_indices]
        y_test = y[test_indices]
        return X_train, y_train, X_test, y_test
    end
end

X_train, X_test, test_indices, train_indices = split_data(X, 0.2)

### masking for raw expression values

function mask_input(X::Matrix{Float32}; mask_ratio=mask_ratio)
    X_masked = copy(X)
    mask_labels = fill(NaN32, size(X)) #!# NaN to ignore positions in the loss calculation

    for j in 1:size(X,2) # per column
        num_masked = ceil(Int, size(X,1) * mask_ratio)
        mask_positions = randperm(size(X,1))[1:num_masked]

        for pos in mask_positions
            mask_labels[pos, j] = X[pos, j] 
            X_masked[pos, j] = MASK_VALUE  
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
    seq_len=n_genes,
    embed_dim=embed_dim,
    n_layers=n_layers,
    n_classes=n_classes, # n_classes is 1
    n_heads=n_heads,
    hidden_dim=hidden_dim,
    dropout_prob=drop_prob
) |> gpu

opt = Flux.setup(Adam(lr), model)

#!# loss is now mse for regression on masked values

function loss(model::Model, x, y, mode::String)
    preds = model(x)  # (1, seq_len, batch_size)
    preds_flat = vec(preds)
    y_flat = vec(y)

    mask = .!isnan.(y_flat)

    if sum(mask) == 0
        return 0.0f0
    end
    
    preds_masked = preds_flat[mask]
    y_masked = y_flat[mask]
    
    regression_loss = Flux.mse(preds_masked, y_masked)

    if mode == "train"
        return regression_loss
    end
    if mode == "test"
        return regression_loss, preds_masked, y_masked
    end
end

train_losses = Float32[]
test_losses = Float32[]

# Profile.Allocs.@profile sample_rate=1 begin
for epoch in ProgressBar(1:n_epochs)

    epoch_losses = Float32[]

    # # dynamic masking here (optional, kept as is)
    # X_train_masked = copy(X_train)
    # y_train_masked = mask_input_dyn(X_train_masked)

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
    
    for start_idx in 1:batch_size:size(X_test_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
        x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
        y_gpu = gpu(y_test_masked[:, start_idx:end_idx])

        test_loss_val, _, _ = loss(model, x_gpu, y_gpu, "test")
        push!(test_epoch_losses, test_loss_val)

    end

    push!(test_losses, mean(test_epoch_losses))
end
# end


#######################################################################################################################################
### EVAL/PLOT
#######################################################################################################################################

# mk dir
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
save_dir = joinpath("plots", save_path_base, "exp_tf", timestamp)
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
save(joinpath(save_dir, "loss.png"), fig_loss)

#!# collect all predictions and true values from the test set
all_preds = Float32[]
all_trues = Float32[]
all_gene_indices = Int[]
all_column_indices = Int[]

for start_idx in 1:batch_size:size(X_test_masked, 2)
    end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
    x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
    y_gpu = gpu(y_test_masked[:, start_idx:end_idx])
    _, preds_masked, y_masked = loss(model, x_gpu, y_gpu, "test")

    # for calculating predicted error per gene
    y_cpu = cpu(y_gpu)
    masked_indices = findall(!isnan, y_cpu)
    batch_gene_indices = [idx[1] for idx in masked_indices]
    append!(all_gene_indices, batch_gene_indices)

    append!(all_preds, cpu(preds_masked))
    append!(all_trues, cpu(y_masked))

    batch_col_indices = start_idx:end_idx
    pred_col_indices = [batch_col_indices[idx[2]] for idx in masked_indices]
    append!(all_column_indices, pred_col_indices)
end

correlation = cor(all_trues, all_preds)

### boxplot and histogram combined

min_val = minimum(all_trues)
max_val = maximum(all_trues)

# define bins
bin_edges = min_val:1.0:max_val
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
        ylabel="predicted expression value",
        title="predicted vs. true expression values"
    )
    ax_hist = Axis(fig_boxhist[2, 1],
        xlabel="true expression value",
        ylabel="count",
        title="distribution of true expression values",
        xticks = 0:5:15,
    )
    linkxaxes!(ax_box, ax_hist)

    # boxplot
    scatter!(ax_box, x_outliers, y_outliers, markersize = 5, alpha = 0.5)
    rangebars!(ax_box, midpts_plot, q10s, q25s, color = :black, whiskerwidth = 0.5)
    rangebars!(ax_box, midpts_plot, q75s, q90s, color = :black, whiskerwidth = 0.5)
    boxplot!(ax_box, grouped_trues_midpts, grouped_preds, range = false, whiskerlinewidth = 0, show_outliers = false)

    # histogram
    hist!(ax_hist, all_trues, bins = bin_edges, strokecolor = :black, strokewidth = 1)
    rowgap!(fig_boxhist.layout, 1, 10)
    display(fig_boxhist)
    save(joinpath(save_dir, "box_hist.png"), fig_boxhist)
end

### plot hexbin
fig_hex = Figure(size = (800, 600))
ax_hex = Axis(fig_hex[1, 1],
    xlabel="true expression val",
    ylabel="predicted expression val",
    title="predicted vs. true expression density",
    aspect=DataAspect() 
)
hexplot = hexbin!(ax_hex, all_trues, all_preds)
Colorbar(fig_hex[1, 2], hexplot, label="point count")
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


### checking if predicting average
gene_averages_train = vec(mean(X_train, dims=2)) |> cpu
masked_indices = findall(!isnan, y_test_masked)
gene_indices_for_masked_values = getindex.(masked_indices, 1)
baseline_preds = gene_averages_train[gene_indices_for_masked_values]
mse_model = mean((all_trues .- all_preds).^2)
mse_baseline = mean((all_trues .- baseline_preds).^2)

fig_baseline_hex = Figure(size = (800, 600))
ax_baseline_hex = Axis(fig_baseline_hex[1, 1], xlabel="true expression val", ylabel="gene average val", title="predicting the average vs. true expression density", aspect=DataAspect())
hexplot_baseline = hexbin!(ax_baseline_hex, all_trues, baseline_preds)
Colorbar(fig_baseline_hex[1, 2], hexplot_baseline, label="point count")
save(joinpath(save_dir, "avg_hexbin.png"), fig_baseline_hex)

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
jldsave(joinpath(save_dir, "indices.jld2"); 
    test_indices = test_indices, 
    train_indices = train_indices
)
jldsave(joinpath(save_dir, "masked_test_data.jld2"); X=X_test_masked, y=y_test_masked)
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
    println(io, "mask_value = $MASK_VALUE")
    println(io, "NO DYNAMIC MASKING")
    println(io, "batch_size = $batch_size")
    println(io, "n_epochs = $n_epochs")
    println(io, "embed_dim = $embed_dim")
    println(io, "hidden_dim = $hidden_dim")
    println(io, "n_heads = $n_heads")
    println(io, "n_layers = $n_layers")
    println(io, "learning_rate = $lr")
    println(io, "dropout_probability = $drop_prob")
    println(io, "ADDITIONAL NOTES: $(additional_notes)")
    println(io, "run_time = $(run_hours) hours and $(run_minutes) minutes")
    println(io, "correlation = $correlation")
    println(io, "mse model = $mse_model")
    println(io, "mse baseline = $mse_baseline")
end