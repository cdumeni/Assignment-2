### A Pluto.jl notebook ###
# v0.20.1

using Markdown
using InteractiveUtils

# ╔═╡ e8e33042-2777-11f0-0762-4fb5f5e8095f
begin
	using Pkg
	
	Pkg.add("ARFFFiles")
	Pkg.add("DataFrames")
	Pkg.precompile()
	Pkg.rm("MLJBase")
	Pkg.add("MLJBase")
end


# ╔═╡ ecbd7b18-74fd-43c2-8a81-84384f7f0d6c
using ARFFFiles, DataFrames

# ╔═╡ 0fccab31-8e14-4b68-8ae0-6c495a8e293c
begin
	data = ARFFFiles.load("C://Users//Salatiel Johannes//Downloads//ag_soybean.arff")
	df = DataFrame(data)
	
end

# ╔═╡ 04fc539c-ea75-4e38-aece-fc956d3e4ab6
df_clean = dropmissing(df)

# ╔═╡ a9848d5e-861b-4703-9310-0b65313fc38b
begin
	using MLJ
	
	X = select(df_clean, Not(:class))
	y = df_clean.class
	
	# Coerce features to Multiclass explicitly
	X_encoded = coerce(X, autotype(X, :string_to_multiclass))
	
end

# ╔═╡ 689a76e6-2f8f-4dc4-aff4-fb971bd09763
begin
	using MLJBase
	
	train_inds, test_inds = partition(eachindex(y), 0.7, shuffle=true)
	X_train, X_test = X_encoded[train_inds, :], X_encoded[test_inds, :]
	y_train, y_test = y[train_inds], y[test_inds]
	
end

# ╔═╡ Cell order:
# ╠═e8e33042-2777-11f0-0762-4fb5f5e8095f
# ╠═ecbd7b18-74fd-43c2-8a81-84384f7f0d6c
# ╠═0fccab31-8e14-4b68-8ae0-6c495a8e293c
# ╠═04fc539c-ea75-4e38-aece-fc956d3e4ab6
# ╠═a9848d5e-861b-4703-9310-0b65313fc38b
# ╠═689a76e6-2f8f-4dc4-aff4-fb971bd09763
