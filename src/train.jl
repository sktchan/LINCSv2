module Training
export loss, train_model!
using Flux, CUDA, Statistics, ProgressBars

function loss(model::ModelDefinition.Model, x, y, mode::String)
    preds = model(x)
    preds_flat = vec(preds)
    y_flat = vec(y)
    mask = .!isnan.(y_flat)
    
    sum(mask) == 0 && return 0.0f0 # Return zero loss if no masked values
    
    preds_masked = preds_flat[mask]
    y_masked = y_flat[mask]
    regression_loss = Flux.mse(preds_masked, y_masked)
    
    return mode == "train" ? regression_loss : (regression_loss, preds_masked, y_masked)
end

function train_model!(model, opt, X_train_masked, y_train_masked, X_test_masked, y_test_masked, n_epochs, batch_size)
    train_losses = Float32[]
    test_losses = Float32[]

    for epoch in ProgressBar(1:n_epochs)
        # train
        epoch_losses = Float32[]
        for start_idx in 1:batch_size:size(X_train_masked, 2)
            end_idx = min(start_idx + batch_size - 1, size(X_train_masked, 2))
            x_gpu = gpu(X_train_masked[:, start_idx:end_idx])
            y_gpu = gpu(y_train_masked[:, start_idx:end_idx])
            
            loss_val, grads = Flux.withgradient(m -> loss(m, x_gpu, y_gpu, "train"), model)
            Flux.update!(opt, model, grads[1])
            push!(epoch_losses, loss_val)
        end
        push!(train_losses, mean(epoch_losses))

        # test
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
    return train_losses, test_losses
end

end