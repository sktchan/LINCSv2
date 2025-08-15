module DataProcessing
export split_data, mask_input
using Random

function split_data(X, test_ratio::Float64)
    n = size(X, 2)
    indices = shuffle(1:n)
    test_size = floor(Int, n * test_ratio)
    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]
    return X[:, train_indices], X[:, test_indices]
end

const MASK_VALUE = -1.0f0

function mask_input(X::Matrix{Float32}; mask_ratio::Float64)
    X_masked = copy(X)
    mask_labels = fill(NaN32, size(X))

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

end