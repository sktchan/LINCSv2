module ModelDefinition
export Model
using Flux, CUDA

# so we can use GPU or CPU :D
const Float32Matrix2DType = Union{Array{Float32}, CuArray{Float32, 2}}
const Float32Matrix3DType = Union{Array{Float32}, CuArray{Float32, 3}}

### positional encoder
struct PosEnc
    pe_matrix::CuArray{Float32,2}
end

function PosEnc(embed_dim::Int, max_len::Int)
    pe_matrix = Matrix{Float32}(undef, embed_dim, max_len)
    for pos in 1:max_len, i in 1:embed_dim
        angle = pos / (10000^(2*(div(i-1,2))/embed_dim))
        pe_matrix[i,pos] = isodd(i) ? sin(angle) : cos(angle)
    end
    return PosEnc(cu(pe_matrix))
end

Flux.@functor PosEnc

(pe::PosEnc)(input::Float32Matrix3DType) = input .+ pe.pe_matrix[:, 1:size(input,2)]

### transformer
struct Transf
    mha::Flux.MultiHeadAttention
    att_dropout::Flux.Dropout
    att_norm::Flux.LayerNorm
    mlp::Flux.Chain
    mlp_norm::Flux.LayerNorm
end

function Transf(embed_dim::Int, hidden_dim::Int; n_heads::Int, dropout_prob::Float64)
    mha = Flux.MultiHeadAttention(embed_dim => embed_dim; nheads=n_heads, dropout_prob=dropout_prob)
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

function (tf::Transf)(input::Float32Matrix3DType)
    normed = tf.att_norm(input)
    atted = tf.mha(normed, normed, normed)
    att_dropped = tf.att_dropout(atted)
    residualed = input + att_dropped
    res_normed = tf.mlp_norm(residualed)

    embed_dim, seq_len, batch_size = size(res_normed)
    reshaped = reshape(res_normed, embed_dim, seq_len * batch_size)
    mlp_out = tf.mlp(reshaped)
    mlp_out_reshaped = reshape(mlp_out, embed_dim, seq_len, batch_size)
    
    return residualed + mlp_out_reshaped
end

### full model
struct Model
    projection::Flux.Dense
    pos_encoder::PosEnc
    pos_dropout::Flux.Dropout
    transformer::Flux.Chain
    classifier::Flux.Chain
end

function Model(; seq_len::Int, embed_dim::Int, n_layers::Int, n_heads::Int, hidden_dim::Int, dropout_prob::Float64)
    projection = Flux.Dense(1 => embed_dim)
    pos_encoder = PosEnc(embed_dim, seq_len)
    pos_dropout = Flux.Dropout(dropout_prob)
    transformer = Flux.Chain([Transf(embed_dim, hidden_dim; n_heads, dropout_prob) for _ in 1:n_layers]...)
    classifier = Flux.Chain(
        Flux.Dense(embed_dim => embed_dim, gelu),
        Flux.LayerNorm(embed_dim),
        Flux.Dense(embed_dim => 1)
    )
    return Model(projection, pos_encoder, pos_dropout, transformer, classifier)
end

Flux.@functor Model

function (model::Model)(input::Float32Matrix2DType)
    seq_len, batch_size = size(input)
    input_reshaped = reshape(input, 1, :)
    embedded = reshape(model.projection(input_reshaped), :, seq_len, batch_size)
    
    encoded = model.pos_encoder(embedded)
    encoded_dropped = model.pos_dropout(encoded)
    transformed = model.transformer(encoded_dropped)
    
    return model.classifier(transformed)
end

end