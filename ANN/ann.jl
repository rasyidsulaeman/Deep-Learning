using DataFrames, CSV
using Random
using Flux
using StatsBase
using Base.Iterators: repeated

Random.seed!(123)

# Import the data
dataset = DataFrame(CSV.File("Churn_Modelling.csv"))
xdata = Matrix(dataset[:,4:end-1])
ydata = dataset[:,end]

# Encoding categorical data
xdata[:,2] = map(x -> x == "France" ? 1 : x == "Spain" ? 2 : 3, xdata[:,2])
xdata[:,3] = map(x -> x == "Female" ? 0 : 1, xdata[:,3])

xdata = hcat(Flux.onehotbatch(xdata[:,2],1:3)',xdata)
xdata = convert(Array{Float64,2},xdata[:,2:end]) # avoid dummy variable trap

# Splitting the data into training and test sets
function partition(xdat::Array{<:AbstractFloat, 2}, ydat::Array{<:Int, 1}; ratio::AbstractFloat = 0.3)
    scnt = size(xdat, 1) / length(unique(ydat));
    ntst = Int(ceil((size(xdat, 1) * ratio) / length(unique(ydat))));
    idx  = Int.(sample(1:(length(ydat) / length(unique(ydat))), ntst, replace = false));
    for i in 2:length(unique(ydat))
        idx = vcat(idx, Int.(sample(((scnt * (i - 1)) + 1):(scnt * i), ntst, replace = false)));
    end
    xtrn = xdat[.!in.(1:length(ydat), Ref(Set(idx))), :];
    ytrn = ydat[.!in.(1:length(ydat), Ref(Set(idx)))];
    xtst = xdat[idx, :];
    ytst = ydat[idx];

    return (xtrn, ytrn, xtst, ytst);
end

xtrain, ytrain, xtest, ytest = partition(xdata,ydata,ratio=0.2)

# Features scaling
x_features = fit(ZScoreTransform,xtrain,dims=1)
x_train = StatsBase.transform(x_features,xtrain)

# Spefify the model
minibatches = Tuple{typeof(xtrain),typeof(ytrain)}[]
batch_size = 32; n_batch = 250
for i in 1:n_batch
    random_idcs = sample(1:size(xtrain)[1], batch_size)
    push!(minibatches, (xtrain[random_idcs,:], ytrain[random_idcs]))
end

model = Flux.Chain(
    Flux.Dense(size(xtrain)[2],6,Flux.relu),
    Flux.Dense(6,6,Flux.relu),
    Flux.Dense(6,1),
    Flux.sigmoid
)

# Define loss function
loss(x,y) = sum(Flux.binarycrossentropy.(model(x'),Float32.(y')))/size(y)[1]

callback = () -> @show(loss(xtrain,ytrain))

Flux.@epochs 100 Flux.train!(loss,Flux.params(model),minibatches,Flux.ADAM(), cb = Flux.throttle(callback,1))
