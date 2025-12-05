using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase, JLD2, SparseArrays, Dates, Printf, Profile
using Flux, Random, OneHotArrays, CategoricalArrays, ProgressBars, CUDA, Statistics, CairoMakie, LinearAlgebra, MLUtils

CUDA.device!(3)

# run_timestamp = "2025-11-26_00-21" 
# save_dir = "plots/untrt/rank_tf/$(run_timestamp)" 
save_dir = "/home/golem/scratch/chans/lincsv2/plots/untrt/TEST_rank_tf/baseline/2025-11-26_00-21"
data_path = "data/lincs_untrt_data.jld2"

struct PosEnc
    pe_matrix::AbstractMatrix{Float32} 
end
Flux.@functor PosEnc
function (pe::PosEnc)(input)
    seq_len = size(input,2)
    dev = Flux.get_device(input) 
    return input .+ (pe.pe_matrix[:,1:seq_len] |> dev)
end

function PosEnc(embed_dim::Int, max_len::Int)
    pe_matrix = Matrix{Float32}(undef, embed_dim, max_len)
    for pos in 1:max_len, i in 1:embed_dim
        angle = pos / (10000^(2*(div(i-1,2))/embed_dim))
        if mod(i, 2) == 1
            pe_matrix[i,pos] = sin(angle)
        else
            pe_matrix[i,pos] = cos(angle)
        end
    end
    return PosEnc(pe_matrix)
end

struct Transf
    mha::Flux.MultiHeadAttention
    att_dropout::Flux.Dropout
    att_norm::Flux.LayerNorm
    mlp::Flux.Chain
    mlp_norm::Flux.LayerNorm
end
Flux.@functor Transf
function (tf::Transf)(input)
    normed = tf.att_norm(input)
    atted = tf.mha(normed, normed, normed)[1]
    att_dropped = tf.att_dropout(atted)
    residualed = input + att_dropped
    res_normed = tf.mlp_norm(residualed)
    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size)
    mlp_out = tf.mlp(reshaped)
    mlp_out_reshaped = reshape(mlp_out, embed_dim, seq_len, batch_size)
    return residualed + mlp_out_reshaped
end

struct Model
    embedding::Flux.Embedding
    pos_encoder::PosEnc
    pos_dropout::Flux.Dropout
    transformer::Flux.Chain
    classifier::Flux.Chain
end
Flux.@functor Model
function (model::Model)(input)
    embedded = model.embedding(input)
    encoded = model.pos_encoder(embedded)
    encoded_dropped = model.pos_dropout(encoded)
    transformed = model.transformer(encoded_dropped)
    return model.classifier(transformed)
end

data = load(data_path)["filtered_data"]

function sort_gene(expr)
    n, m = size(expr)
    data_ranked = Matrix{Int}(undef, size(expr))
    sorted_ind_col = Vector{Int}(undef, n)
    for j in 1:m
        unsorted_expr_col = view(expr, :, j)
        sortperm!(sorted_ind_col, unsorted_expr_col, rev=true)
        for i in 1:n
            data_ranked[i, j] = sorted_ind_col[i]
        end
    end
    return data_ranked
end
X_raw = sort_gene(data.expr)

const n_classes = size(X_raw, 1)
const CLS_ID = n_classes + 2
const MASK_ID = n_classes + 1
X = vcat(fill(CLS_ID, (1, size(X_raw, 2))), X_raw)

idx = load(joinpath(save_dir, "indices.jld2"))
test_idx = idx["test_indices"]
X_test = X[:, test_idx]

# because accidentally defined posenc as a cuarray in the struct when saving the model; thus need to re-make posenc w/o being a cuarray
loaded_object = load(joinpath(save_dir, "model_object.jld2"));
raw_model = loaded_object["model"];

saved_embedding = raw_model.embedding
saved_transformer = raw_model.transformer
saved_classifier = raw_model.classifier
saved_pos_dropout = raw_model.pos_dropout

input_size_loaded = size(saved_embedding.weight, 2)
embed_dim_loaded = size(saved_embedding.weight, 1)

new_pos_encoder = PosEnc(embed_dim_loaded, input_size_loaded)

model = Model(
    saved_embedding,
    new_pos_encoder,
    saved_pos_dropout,
    saved_transformer,
    saved_classifier
) |> gpu 

ranks_list = Int[]
diff_expr_list = Float32[]
ce_loss_list = Float32[]

pred_rank_list = Int[]
pred_expr_list = Float32[]

mask_ratio = 0.1
batch_size = 128

function mask(X::Matrix{Int}; mask_ratio=0.1)
    X_masked = copy(X)
    mask_labels = fill((-100), size(X))
    for j in 1:size(X,2)
        num_masked = ceil(Int, (size(X,1) - 1) * mask_ratio)
        mask_positions_local = randperm(size(X,1) - 1)[1:num_masked]
        mask_positions_global = mask_positions_local .+ 1 
        for pos in mask_positions_global
            mask_labels[pos, j] = X[pos, j]
            X_masked[pos, j] = MASK_ID
        end
    end
    return X_masked, mask_labels
end

X_test_masked, y_test_masked = mask(X_test, mask_ratio=mask_ratio)

for start_idx in 1:batch_size:size(X_test_masked, 2)
    end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
    
    x_batch = X_test_masked[:, start_idx:end_idx]
    y_batch = y_test_masked[:, start_idx:end_idx]
    batch_test_idx = test_idx[start_idx:end_idx] 
    
    x_gpu = gpu(x_batch)
    
    logits = model(x_gpu) # (n_classes, seq_len, batch_size)
    logits_cpu = cpu(logits)
    
    for b in 1:size(x_batch, 2)
        sample_idx_in_data = batch_test_idx[b]
        
        masked_positions = findall(y_batch[:, b] .!= -100)
        
        for pos in masked_positions
            true_gene_id = y_batch[pos, b]
            
            logit_vec = logits_cpu[:, pos, b]
            pred_gene_id = argmax(logit_vec)
            
            true_rank = pos - 1 
            
            pred_rank_loc = findfirst(==(pred_gene_id), view(X_raw, :, sample_idx_in_data))
            
            expr_true = data.expr[true_gene_id, sample_idx_in_data]
            expr_pred = data.expr[pred_gene_id, sample_idx_in_data]
            
            diff_expr = expr_pred - expr_true
            
            y_oh = Flux.onehotbatch([true_gene_id], 1:n_classes)
            ce = Flux.logitcrossentropy(reshape(logit_vec, :, 1), y_oh)
            
            push!(ranks_list, true_rank)
            push!(diff_expr_list, diff_expr)
            push!(ce_loss_list, ce)
            
            push!(pred_rank_list, pred_rank_loc)
            push!(pred_expr_list, expr_pred)
        end
    end
end

# PLOT 1: pred vs true diff in exp vs rank
df_diff = DataFrame(rank = ranks_list, diff = diff_expr_list)
gdf_diff = combine(groupby(df_diff, :rank), :diff => mean => :avg_diff) 
sort!(gdf_diff, :rank)
begin
    fig1 = Figure(size = (800, 600))
    ax1 = Axis(fig1[1, 1], 
        xlabel = "true rank (1 = highest expression)", 
        ylabel = "predicted expression - true expression",
        title = "prediction error vs rank")

    scatter!(ax1, ranks_list, diff_expr_list , alpha=0.5, markersize=3)
    lines!(ax1, gdf_diff.rank, gdf_diff.avg_diff, color=:red, label="mean")
    axislegend(ax1)
    display(fig1)
end
save(joinpath(save_dir, "plot1B_expr_diff_scatter.png"), fig1)


# PLOT 2: variance in exp vs rank
n_ranks = size(X_raw, 1)
true_rank_vars = Float32[]
true_rank_idx = 1:n_ranks

for r in 1:n_ranks
    exprs_at_rank = [data.expr[X_raw[r, i], i] for i in test_idx]
    push!(true_rank_vars, var(exprs_at_rank))
end

fig2 = Figure(size = (800, 600))
ax2 = Axis(fig2[1, 1], 
    xlabel = "true rank", 
    ylabel = "expression variance",
    title = "variance of expression at true rank")

scatter!(ax2, true_rank_idx, true_rank_vars, alpha=0.5)
save(joinpath(save_dir, "plot2_var_vs_true_rank_scatter.png"), fig2)

# PLOT 3: variance in exp vs pred rank
df_pred = DataFrame(rank = pred_rank_list, expr = pred_expr_list)
gdf_pred = combine(groupby(df_pred, :rank), :expr => var => :expr_var)

fig3 = Figure(size = (800, 600))
ax3 = Axis(fig3[1, 1], 
    xlabel = "predicted rank", 
    ylabel = "expression variance",
    title = "variance of expression at predicted rank")

scatter!(ax3, gdf_pred.rank, gdf_pred.expr_var, alpha=0.5)
save(joinpath(save_dir, "plot3_var_vs_pred_rank_scatter.png"), fig3)

# PLOT 2/3: overlapping 2 and 3
n_ranks = size(X_raw, 1)
true_rank_vars = Float32[]
true_rank_idx = 1:n_ranks

for r in 1:n_ranks
    exprs_at_rank = [data.expr[X_raw[r, i], i] for i in test_idx]
    push!(true_rank_vars, var(exprs_at_rank))
end

df_pred = DataFrame(rank = pred_rank_list, expr = pred_expr_list)
gdf_pred = combine(groupby(df_pred, :rank), :expr => var => :expr_var)
begin
    fig3 = Figure(size = (800, 600))
    ax3 = Axis(fig3[1, 1], 
        xlabel = "rank", 
        ylabel = "expression varaince",
        title = "expression variance at true/predicted rank")

    scatter!(ax3, gdf_pred.rank, gdf_pred.expr_var, alpha=0.5, label="predicted")
    lines!(ax3, true_rank_idx, true_rank_vars, color=:red, label="true")
    axislegend(ax3)
    display(fig3)
end
save(joinpath(save_dir, "plot2n3_var_overlay.png"), fig3)


# PLOT 4: CE loss vs rank
df_ce = DataFrame(rank = ranks_list, loss = ce_loss_list)
gdf_ce = combine(groupby(df_ce, :rank), :loss => mean => :mean_loss)
sort!(gdf_ce, :rank)
begin
    fig4 = Figure(size = (800, 600))
    ax4 = Axis(fig4[1, 1], 
        xlabel = "true rank", 
        ylabel = "cross entropy loss",
        title = "cross entropy loss vs rank")

    scatter!(ax4, ranks_list, ce_loss_list, alpha=0.5, markersize=3)
    display(fig4)
end
save(joinpath(save_dir, "plot4_ce_loss_scatter.png"), fig4)

# PLOT 5: matrix of sum of squared differences between positional encodings
pe = cpu(model.pos_encoder.pe_matrix)
dims, seq_len = size(pe)
dist_matrix = Matrix{Float32}(undef, seq_len, seq_len)

for i in 1:seq_len
    for j in 1:seq_len
        d = 0.0f0
        for k in 1:dims
            diff = pe[k, i] - pe[k, j]
            d += diff * diff
        end
        dist_matrix[i, j] = d
    end
end

fig5 = Figure(size = (800, 700))
ax5 = Axis(fig5[1, 1], 
    xlabel = "rank position index", 
    ylabel = "rank position index",
    title = "posenc matrix",
    aspect = 1)

hm = heatmap!(ax5, 1:seq_len, 1:seq_len, dist_matrix, colormap=:viridis)

Colorbar(fig5[1, 2], hm, label="sum of squared differences")
save(joinpath(save_dir, "plot5_pos_encoding_matrix.png"), fig5)