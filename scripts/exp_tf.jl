using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using JLD2, Flux, CUDA, Random

include("../src/model.jl")
using .ModelDefinition
include("../src/process_data.jl")
using .DataProcessing
include("../src/train.jl")
using .Training
include("../src/eval.jl")
using .Evaluation

CUDA.device!(0)

start_time = now()
params = (
    mask_ratio = 0.1,
    test_ratio = 0.2,
    batch_size = 64,
    n_epochs = 100,
    embed_dim = 64,
    hidden_dim = 128,
    n_heads = 1,
    n_layers = 4,
    drop_prob = 0.05,
    lr = 0.001,
    start_time = start_time,
    end_time = now()
)

data = load("data/lincs_untrt_data.jld2")["filtered_data"]
X = data.expr
n_genes = size(X, 1)

X_train, X_test = split_data(X, params.test_ratio)
X_train_masked, y_train_masked = mask_input(X_train; mask_ratio=params.mask_ratio)
X_test_masked, y_test_masked = mask_input(X_test; mask_ratio=params.mask_ratio)

model = Model(
    seq_len=n_genes,
    embed_dim=params.embed_dim,
    n_layers=params.n_layers,
    n_heads=params.n_heads,
    hidden_dim=params.hidden_dim,
    dropout_prob=params.drop_prob
) |> gpu

opt = Flux.setup(Adam(params.lr), model)

train_losses, test_losses = train_model!(
    model, opt, 
    X_train_masked, y_train_masked, 
    X_test_masked, y_test_masked,
    params.n_epochs, params.batch_size
)

final_params = merge(params, (end_time = now(),)) # update end time

evaluate_and_save(
    model, 
    X_train, X_test_masked, y_test_masked, 
    train_losses, test_losses,
    final_params
)