using LincsProject, DataFrames, CSV, Dates, JSON, StatsBase

# load in beta data via terminal if not present
#=
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/level3_beta_all_n3026460x12328.gctx
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/instinfo_beta.txt
=#

### load in lincs dataset (this is the one filtered for lea's project)
@time lincs_data = LincsProject.Lincs("data/lincs_loading_files/",
                                      "level3_beta_all_n3026460x12328.gctx",
                                      "data/lincs_data.jld2")

# into jld2                                      
function jld2_lincs(lincs_data, filter::Dict, out_file::String)
    filter_idx = LincsProject.create_filter(lincs_data, filter)
    filtered_lincs = lincs_data[filter_idx]
    filtered_expr = lincs_data.expr[:, filter_idx]
    filtered_data = Lincs(filtered_expr, lincs_data.gene, lincs_data.compound, filtered_lincs)
    jldsave(out_file; filtered_data)
    return filtered_data
end

function load_lincs(in_file::String)
    @time return load(in_file)["filtered_data"]
end

# into df for CSV
function csv_lincs(lincs_data, filtered_expr, filtered_lincs, csv_file::String)
    df = DataFrame(transpose(filtered_expr), :auto)
    insertcols!(df, 1, :cell_line => filtered_lincs.cell_iname)
    col_names = ["cell_line"; lincs_data.gene.gene_symbol]
    rename!(df, col_names)
    CSV.write(csv_file, df)
end

####################################################################################################################

### filter for untreated cell lines, 978 x 100425
untreated_filter = Dict(
    :qc_pass => [Symbol("1")],
    :pert_type => [:ctl_untrt, :ctl_vehicle]
)
untreated_file = "data/lincs_untrt_data.jld2"
untreated_data = jld2_lincs(lincs_data, untreated_filter, untreated_file)
loaded_untreated_data = load_lincs(untreated_file)


### on all cell line profiles (trt + untrt), 978 x 1412595
all_filter = Dict(
    :qc_pass => [Symbol("1")],
    :pert_type => [:ctl_untrt, :ctl_vehicle, :trt_cp]
)
all_file = "data/lincs_trt_untrt_data.jld2"
all_data = jld2_lincs(lincs_data, all_filter, all_file)
loaded_all_data = load_lincs(all_file)


# csv_lincs(lincs_data, untreated_data.expr, untreated_data, "data/cellline_geneexpr.csv")
# csv_lincs(lincs_data, all_data.expr, all_data, "data/all_cellline_geneexpr.csv")