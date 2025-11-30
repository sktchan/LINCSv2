using Pkg
Pkg.activate("/home/golem/scratch/chans/lincs")

using Flux, CUDA, Random, Statistics, ProgressBars, JLD2
using DataFrames, CSV, MLUtils, StatsBase

CUDA.device!(0)

#######################################################################################################################################
### MODEL DEFINITIONS (REUSED FROM YOUR PRE-TRAINING SCRIPT)
#######################################################################################################################################

# We'll need these types for the model definition
const IntMatrix2DType = Union{Array{Int}, CuArray{Int, 2}}
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
const Float32Matrix3DType = Union{Array{Float32, 3}, CuArray{Float32, 3}}
const IntVectorType = Union{Vector{Int}, CuArray{Int, 1}}

### Positional Encoder
struct PosEnc
    pe_matrix::CuArray{Float32,2}
end

function PosEnc(embed_dim::Int, max_len::Int)
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

### Transformer Block
struct Transf
    mha::Flux.MultiHeadAttention
    att_dropout::Flux.Dropout
    att_norm::Flux.LayerNorm
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
    atted = tf.mha(normed, normed, normed)[1] 
    att_dropped = tf.att_dropout(atted)
    residualed = input + att_dropped
    res_normed = tf.mlp_norm(residualed)
    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size)
    mlp_out = tf.mlp(reshaped)
    mlp_out_reshaped = reshape(mlp_out, embed_dim, seq_len, batch_size)
    tf_output = residualed + mlp_out_reshaped
    return tf_output
end

#######################################################################################################################################
### NEW DRP MODEL
#######################################################################################################################################

struct DRPModel
    # Re-used pre-trained components
    token_embedding::Flux.Embedding
    pos_encoder::PosEnc
    pos_dropout::Flux.Dropout
    transformer::Flux.Chain
    
    # New components for DRP
    drug_embedding::Flux.Embedding
    regression_head::Flux.Chain
end

function DRPModel(
    pretrained_model_path::String,
    n_drugs::Int,
    gene_vocab_size::Int,          # Your n_features (e.g., 978 genes + 2 tokens)
    embed_dim::Int,                # 128
    n_layers::Int,                 # 1
    n_heads::Int,                  # 2
    hidden_dim::Int,               # 256
    dropout_prob::Float64 = 0.05
    )

    # 1. Load the pre-trained model state
    # We assume the JLD2 file contains the `model_state` from your pre-training
    pretrain_state = JLD2.load(pretrained_model_path)["model_state"]

    # 2. Build the Transformer body components
    token_embedding = Flux.Embedding(gene_vocab_size => embed_dim)
    pos_encoder = PosEnc(embed_dim, gene_vocab_size)
    pos_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain(
        [Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...
    )

    # 3. Load pre-trained weights into the components
    # We match state keys (e.g., "embedding.weight", "transformer.layers.1.mha.q_proj.weight")
    # Note: `classifier` head from pre-training is NOT loaded.
    Flux.loadmodel!(token_embedding, pretrain_state; prefix="embedding")
    Flux.loadmodel!(pos_encoder, pretrain_state; prefix="pos_encoder")
    Flux.loadmodel!(transformer, pretrain_state; prefix="transformer")
    
    # 4. Define new components for DRP
    # Drug embedding layer
    drug_embed_dim = embed_dim # Let's use the same dim for simplicity
    drug_embedding = Flux.Embedding(n_drugs => drug_embed_dim)

    # Regression head
    # It takes the concatenated [CLS] token and drug embedding
    regression_head = Flux.Chain(
        Flux.Dense((embed_dim + drug_embed_dim) => hidden_dim, gelu),
        Flux.LayerNorm(hidden_dim),
        Flux.Dropout(dropout_prob),
        Flux.Dense(hidden_dim => 1) # Output a single AUDRC value
    )

    return DRPModel(
        token_embedding, pos_encoder, pos_dropout, transformer,
        drug_embedding, regression_head
    )
end

Flux.@functor DRPModel

# Define the forward pass for the DRPModel
function (m::DRPModel)(gene_sequence_batch::IntMatrix2DType, drug_id_batch::IntVectorType)
    # gene_sequence_batch shape: (seq_len, batch_size)
    # drug_id_batch shape: (batch_size,)

    # 1. Process Gene Expression through Transformer
    embedded = m.token_embedding(gene_sequence_batch)
    encoded = m.pos_encoder(embedded)
    encoded_dropped = m.pos_dropout(encoded)
    transformed = m.transformer(encoded_dropped) # (embed_dim, seq_len, batch_size)

    # 2. Extract [CLS] token (assuming it's at position 1)
    cls_embedding = transformed[:, 1, :] # (embed_dim, batch_size)

    # 3. Process Drug ID
    drug_embed = m.drug_embedding(drug_id_batch) # (embed_dim, batch_size)

    # 4. Concatenate and predict
    combined_embedding = vcat(cls_embedding, drug_embed) # (embed_dim + drug_embed_dim, batch_size)
    
    # Reshape for Flux.Dense (which expects features as first dim)
    # This is already in the correct (features, batch_size) format.
    prediction = m.regression_head(combined_embedding) # (1, batch_size)

    return prediction
end

#######################################################################################################################################
### DATA LOADING & PREPARATION (LCO)
#######################################################################################################################################

# Gene expression rank sorting (reused from your script)
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

# --- MOCK DATA LOADING FUNCTIONS ---
# In your real application, you would replace these with actual file loading.
function load_mock_ccle_expression()
    println("Loading MOCK CCLE expression...")
    n_genes = 978
    cell_lines = ["Cell_$(lpad(i, 3, '0'))" for i in 1:100]
    # Dict mapping Cell_ID to a random expression vector
    ccle_data = Dict(id => rand(Float32, n_genes) for id in cell_lines)
    return ccle_data, n_genes
end

function load_mock_gdsc_responses()
    println("Loading MOCK GDSC responses...")
    cell_lines = ["Cell_$(lpad(i, 3, '0'))" for i in 1:100]
    drugs = ["Drug_$(lpad(i, 2, '0'))" for i in 1:50]
    
    records = []
    for cell in cell_lines
        for drug in drugs
            # Simulate some drug/cell variance
            drug_effect = findfirst(==(drug), drugs) / 50.0
            cell_effect = findfirst(==(cell), cell_lines) / 100.0
            noise = randn() * 0.1
            # We use AUDRC (Area Under Dose-Response Curve), 0-1 range
            audrc = clamp(0.5 - drug_effect + cell_effect + noise, 0.0, 1.0)
            push!(records, (Cell_ID = cell, Drug_ID = drug, AUDRC = Float32(audrc)))
        end
    end
    return DataFrame(records)
end
# --- END MOCK FUNCTIONS ---


function create_master_table(CLS_ID::Int, n_genes::Int)
    # 1. Load features and labels
    # (Using MOCK functions here)
    ccle_expr_dict, n_genes = load_mock_ccle_expression()
    gdsc_df = load_mock_gdsc_responses()

    # 2. Tokenize Drugs
    # Create a mapping from Drug_ID string to an integer index
    unique_drugs = sort(unique(gdsc_df.Drug_ID))
    drug_to_int = Dict(drug => i for (i, drug) in enumerate(unique_drugs))
    n_drugs = length(unique_drugs)

    # 3. Tokenize Cell Lines (for expression)
    # We will process expressions as we build the table
    n_master_rows = nrow(gdsc_df)
    
    # Pre-allocate columns
    col_cell_id = Vector{String}(undef, n_master_rows)
    col_drug_id_int = Vector{Int}(undef, n_master_rows)
    col_audrc = Vector{Float32}(undef, n_master_rows)
    # This is a matrix where each column is a ranked gene sequence
    col_ranked_genes = Matrix{Int}(undef, n_genes + 1, n_master_rows) # +1 for CLS token

    print("Processing master table... ")
    for (i, row) in enumerate(eachrow(gdsc_df))
        cell_id = row.Cell_ID
        
        # Get expression vector
        # NOTE: In a real case, some CCLE IDs might not be in GDSC or vice-versa
        # We assume all IDs match for this example.
        expr_vector = ccle_expr_dict[cell_id] 
        
        # Rank the genes (this is slow, but matches your function)
        # `sort_gene` expects a matrix, so we pass a 1-column matrix
        ranked_genes_col = sort_gene(reshape(expr_vector, :, 1))
        
        # Add CLS token
        cls_token_id = CLS_ID
        ranked_sequence = vcat([cls_token_id], ranked_genes_col)

        # Store in master table
        col_cell_id[i] = cell_id
        col_drug_id_int[i] = drug_to_int[row.Drug_ID]
        col_audrc[i] = row.AUDRC
        col_ranked_genes[:, i] = ranked_sequence
    end
    println("Done.")

    # We store the main table as a DataFrame, but keep the large gene matrix separate
    master_df = DataFrame(
        Cell_ID = col_cell_id,
        Drug_ID_Int = col_drug_id_int,
        AUDRC = col_audrc
    )
    
    return master_df, col_ranked_genes, n_drugs, n_genes
end


### 2. ✂️ The LCO Split (k-Fold Cross-Validation)
function get_lco_folds(df::DataFrame, k::Int)
    println("Creating $k-fold LCO splits...")
    unique_cell_lines = unique(df.Cell_ID)
    shuffled_cell_lines = shuffle(unique_cell_lines)
    
    # Partition the unique cell line IDs into k folds
    # `partition` is from MLUtils
    cell_id_folds = partition(shuffled_cell_lines, k)
    
    fold_indices = [] # To store (train_indices, test_indices)

    all_row_indices = 1:nrow(df)

    for i in 1:k
        # Test IDs are the cell lines in the i-th partition
        test_cell_ids = Set(cell_id_folds[i])
        
        # Find all row indices in the master DataFrame that match these cell IDs
        test_indices = findall(id -> id in test_cell_ids, df.Cell_ID)
        
        # Train indices are all indices NOT in test_indices
        train_indices = setdiff(all_row_indices, test_indices)
        
        push!(fold_indices, (collect(train_indices), test_indices))
        println("Fold $i: $(length(train_indices)) train samples, $(length(test_indices)) test samples.")
    end
    
    return fold_indices
end


### 3. ⚙️ Evaluation (Fixed-Drug Aggregation)
function fixed_drug_aggregation(y_true::Vector{Float32}, y_pred::Vector{Float32}, drug_ids::Vector{Int})
    
    df = DataFrame(true_val = y_true, pred_val = y_pred, drug = drug_ids)
    
    drug_scores = Float32[]
    
    # Group by drug and calculate Pearson correlation for each
    for drug_group in groupby(df, :drug)
        # Need at least 2 samples to calculate correlation
        if nrow(drug_group) < 2
            continue
        end
        
        # `cor` is from Statistics
        pcc = cor(drug_group.true_val, drug_group.pred_val)
        
        # Add to list if correlation is valid (not NaN)
        if !isnan(pcc)
            push!(drug_scores, pcc)
        end
    end
    
    if isempty(drug_scores)
        println("Warning: No valid drug correlations found in this fold.")
        return 0.0f0
    end
    
    # Final score for the fold is the average of all per-drug scores
    return mean(drug_scores)
end


#######################################################################################################################################
### MAIN FINE-TUNING AND EVALUATION SCRIPT
#######################################################################################################################################
function main_dpr_benchmark()
    
    # --- 1. Parameters ---
    println("Starting DRP Benchmark...")
    
    # Model params (should match your pre-training)
    const EMBED_DIM = 128
    const HIDDEN_DIM = 256
    const N_HEADS = 2
    const N_LAYERS = 1
    const DROP_PROB = 0.05
    const PRETRAINED_MODEL_PATH = "untrt/rank_tf/2024-01-01_12-00/model_state.jld2" # ! UPDATE THIS PATH !

    # Training params
    const N_FOLDS = 5
    const BATCH_SIZE = 64
    const N_EPOCHS = 10 # Fine-tuning usually requires fewer epochs
    const LR = 1e-4

    # --- 2. Data Preparation ---
    # We need to know the vocab size for the pre-trained model
    # (Assuming 978 genes + MASK + CLS)
    const N_GENES_PRETRAIN = 978 
    const MASK_ID = N_GENES_PRETRAIN + 1
    const CLS_ID = N_GENES_PRETRAIN + 2
    const GENE_VOCAB_SIZE = N_GENES_PRETRAIN + 2 

    # Load and process all data
    # (This function uses the MOCK data loaders)
    master_df, gene_sequences_matrix, n_drugs, n_genes_data = create_master_table(CLS_ID, N_GENES_PRETRAIN)

    # Sanity check
    @assert n_genes_data == N_GENES_PRETRAIN "Data gene count ($n_genes_data) does not match pre-train param ($N_GENES_PRETRAIN)"
    @assert size(gene_sequences_matrix, 1) == (N_GENES_PRETRAIN + 1) "Gene sequence length is incorrect"

    # Get LCO Folds
    lco_folds = get_lco_folds(master_df, N_FOLDS)
    
    fold_scores = Float32[]

    # --- 3. ⚙️ Fine-Tuning and Evaluation (The k-Fold Loop) ---
    for (i, (train_indices, test_indices)) in enumerate(lco_folds)
        println("\n--- Starting Fold $i/$N_FOLDS ---")

        # --- A. Generate Datasets for this Fold ---
        
        # Training data
        train_df = master_df[train_indices, :]
        train_X_genes = gene_sequences_matrix[:, train_indices]
        train_X_drugs = train_df.Drug_ID_Int
        train_Y = train_df.AUDRC

        # Test data
        test_df = master_df[test_indices, :]
        test_X_genes = gene_sequences_matrix[:, test_indices]
        test_X_drugs = test_df.Drug_ID_Int
        test_Y = test_df.AUDRC
        
        # Create DataLoaders
        # We need to bundle the two inputs (genes, drugs) together
        train_loader = DataLoader((train_X_genes, train_X_drugs), train_Y, 
                                  batchsize=BATCH_SIZE, shuffle=true, collate=true)
        
        test_loader = DataLoader((test_X_genes, test_X_drugs), test_Y, 
                                 batchsize=BATCH_SIZE, shuffle=false, collate=true)

        # --- B. Instantiate Model ---
        println("Instantiating model and loading pre-trained weights...")
        
        # ! UPDATE PATH: This needs to be the correct path to your .jld2 file
        local model # Use local to avoid global scope issues in loop
        try
            model = DRPModel(
                PRETRAINED_MODEL_PATH,
                n_drugs,
                GENE_VOCAB_SIZE,
                EMBED_DIM,
                N_LAYERS,
                N_HEADS,
                HIDDEN_DIM,
                DROP_PROB
            ) |> gpu
        catch e
            println("Error loading pre-trained model: $e")
            println("!!! Using randomly initialized weights. Make sure PRETRAINED_MODEL_PATH is correct: $(PRETRAINED_MODEL_PATH) !!!")
            # This is a fallback if the load fails, so the script can still run
            # It creates the same architecture but without loading weights
            pretrain_state = Dict() # Empty state
            token_embedding = Flux.Embedding(GENE_VOCAB_SIZE => EMBED_DIM)
            pos_encoder = PosEnc(EMBED_DIM, GENE_VOCAB_SIZE)
            pos_dropout = Flux.Dropout(DROP_PROB)
            transformer = Flux.Chain([Transf(EMBED_DIM, HIDDEN_DIM; n_heads=N_HEADS, dropout_prob=DROP_PROB) for _ in 1:N_LAYERS]...)
            drug_embedding = Flux.Embedding(n_drugs => EMBED_DIM)
            regression_head = Flux.Chain(
                Flux.Dense((EMBED_DIM + EMBED_DIM) => HIDDEN_DIM, gelu),
                Flux.LayerNorm(HIDDEN_DIM),
                Flux.Dropout(DROP_PROB),
                Flux.Dense(hidden_dim => 1)
            )
            model = DRPModel(token_embedding, pos_encoder, pos_dropout, transformer, drug_embedding, regression_head) |> gpu
        end

        # Define loss and optimizer
        # As per the prompt, we use Mean Squared Error for regression
        loss_fn(x_genes, x_drugs, y) = Flux.mse(vec(model(x_genes, x_drugs)), y)
        
        # We fine-tune all parameters, including the pre-trained ones
        opt = Flux.setup(Adam(LR), model)

        # --- C. Fine-Tune ---
        println("Fine-tuning...")
        Flux.trainmode!(model)
        for epoch in 1:N_EPOCHS
            epoch_loss = 0.0
            pbar = ProgressBar(train_loader)
            for (x_batch, y_batch) in pbar
                x_genes_gpu = x_batch[1] |> gpu
                x_drugs_gpu = x_batch[2] |> gpu
                y_gpu = y_batch |> gpu
                
                loss_val, grads = Flux.withgradient(model) do m
                    loss_fn(x_genes_gpu, x_drugs_gpu, y_gpu)
                end
                
                Flux.update!(opt, model, grads[1])
                epoch_loss += loss_val * length(y_batch)
                ProgressBars.set_description(pbar, "Epoch $epoch/$N_EPOCHS, Loss: $(round(loss_val, digits=5))")
            end
            println("Epoch $epoch/$N_EPOCHS, Avg. Train Loss: $(round(epoch_loss / length(train_loader.data[2]), digits=5))")
        end

        # --- D. Evaluate (Fixed-Drug Aggregation) ---
        println("Evaluating on test set...")
        Flux.testmode!(model)
        
        all_y_true = Float32[]
        all_y_pred = Float32[]
        all_drug_ids = Int[]

        for (x_batch, y_batch) in test_loader
            x_genes_gpu = x_batch[1] |> gpu
            x_drugs_gpu = x_batch[2] |> gpu
            
            y_pred_batch = model(x_genes_gpu, x_drugs_gpu) |> cpu
            
            append!(all_y_true, y_batch)
            append!(all_y_pred, vec(y_pred_batch))
            append!(all_drug_ids, x_batch[2]) # Drug IDs are the second part of x_batch
        end

        fold_score = fixed_drug_aggregation(all_y_true, all_y_pred, all_drug_ids)
        println("Fold $i Score (Mean Per-Drug Pearson): $(round(fold_score, digits=4))")
        push!(fold_scores, fold_score)
    end

    # --- 4. Final Result ---
    final_avg_score = mean(fold_scores)
    final_std_dev = std(fold_scores)
    println("\n" * "="^30)
    println("Benchmark Complete")
    println("Task: LCO (Leave-Cell-Line-Out) Generalization")
    println("Metric: Fixed-Drug Aggregation (Mean Per-Drug Pearson)")
    println("="^30)
    println("Final Score ($N_FOLDS-Fold CV): $(round(final_avg_score, digits=4)) ± $(round(final_std_dev, digits=4))")
    println("="^30)

    return final_avg_score
end

# Run the benchmark
if abspath(PROGRAM_FILE) == @__FILE__
    main_dpr_benchmark()
end