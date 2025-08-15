module Evaluation
export evaluate_and_save
using Flux, CUDA, DataFrames, CSV, CairoMakie, Dates, Statistics, LinearAlgebra

function evaluate_and_save(model, X_train, X_test_masked, y_test_masked, train_losses, test_losses, params)
    # mk dir
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
    save_dir = joinpath("plots", "untrt", "masked_expression", timestamp)
    mkpath(save_dir)

    # loss plot
    fig_loss = Figure(size = (800, 600))
    ax_loss = Axis(fig_loss[1, 1], xlabel="Epoch", ylabel="Loss (MSE)", title="Train vs. Test Loss")
    lines!(ax_loss, 1:params.n_epochs, train_losses, label="Train Loss", linewidth=2)
    lines!(ax_loss, 1:params.n_epochs, test_losses, label="Test Loss", linewidth=2)
    axislegend(ax_loss, position=:rt)
    save(joinpath(save_dir, "loss.png"), fig_loss)

    # get preds and trues for compariosn
    all_preds, all_trues = Float32[], Float32[]
    for start_idx in 1:params.batch_size:size(X_test_masked, 2)
        end_idx = min(start_idx + batch_size - 1, size(X_test_masked, 2))
        x_gpu = gpu(X_test_masked[:, start_idx:end_idx])
        y_gpu = gpu(y_test_masked[:, start_idx:end_idx])
        _, preds_masked, y_masked = Training.loss(model, x_gpu, y_gpu, "test")
        append!(all_preds, cpu(preds_masked))
        append!(all_trues, cpu(y_masked))
    end
    
    # hexbin plpot
    fig_hex = Figure(size = (800, 600))
    ax_hex = Axis(fig_hex[1, 1], xlabel="True Expression", ylabel="Predicted Expression", title="Predicted vs. True Density", aspect=DataAspect())
    hexplot = hexbin!(ax_hex, all_trues, all_preds)
    Colorbar(fig_hex[1, 2], hexplot, label="Count")
    save(joinpath(save_dir, "hexbin.png"), fig_hex)

    # average comparison
    gene_averages_train = vec(mean(X_train, dims=2)) |> cpu
    masked_indices = findall(!isnan, y_test_masked)
    gene_indices = getindex.(masked_indices, 1)
    baseline_preds = gene_averages_train[gene_indices]
    mse_model = mean((all_trues .- all_preds).^2)
    mse_baseline = mean((all_trues .- baseline_preds).^2)
    correlation = cor(all_trues, all_preds)

    # log data
    CSV.write(joinpath(save_dir, "losses.csv"), DataFrame(epoch=1:params.n_epochs, train_loss=train_losses, test_loss=test_losses))
    CSV.write(joinpath(save_dir, "predstrues.csv"), DataFrame(all_preds=all_preds, all_trues=all_trues))
    
    run_time_str = Dates.format(params.end_time - params.start_time, "H:M:S")
    open(joinpath(save_dir, "params.txt"), "w") do io
        println(io, "PARAMETERS:")
        for (key, value) in pairs(params)
            println(io, "$key = $value")
        end
        println(io, "\nRESULTS:")
        println(io, "run_time = $run_time_str")
        println(io, "correlation = $correlation")
        println(io, "mse_model = $mse_model")
        println(io, "mse_baseline = $mse_baseline")
    end
end

end