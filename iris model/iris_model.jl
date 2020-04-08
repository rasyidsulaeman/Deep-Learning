using Base.Iterators: repeated
using CSV: write
using DataFrames: DataFrame
using Flux
using Random
using RDatasets
using StatsBase: sample

Random.seed!(123)

# Import the iris datasets
iris = dataset("datasets","iris")
xdat = Matrix(iris[1:4])
ydat = iris[:,5]
ydat = map(x -> x == "setosa" ? 1 : x == "versicolor" ? 2 : 3, ydat);

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

xtrn, ytrn, xtst, ytst = partition(xdat,ydat,ratio=0.3)

ytrn = Flux.onehotbatch(ytrn,1:3)
ytst = Flux.onehotbatch(ytst,1:3)

minibatches = Tuple{typeof(xtrn),typeof(ytrn)}[]; batch_size = 15; n_batch = 10
for i in 1:n_batch
    random_idcs = sample(1:size(xtrn)[1], batch_size)
    push!(minibatches, (xtrn[random_idcs,:], ytrn[:,random_idcs]))
end

#Specify the model
model = Flux.Chain(
    Flux.Dense(size(xtrn)[2],10,Flux.relu),
    Flux.Dense(10,1, Flux.sigmoid)
)

#Define loss function
loss(x,y) = Flux.crossentropy(model(x'),Float32.(y))

#Define the callback function that prints the loss every epoch
callback = () -> @show(loss(xtrn,ytrn))

# Train the model
function accuracy(x,y)
    return sum(Flux.onecold(model(x')) .== Flux.onecold(y)) / size(y, 2)
end

err = hcat(Flux.Tracker.data(loss(xtrn,ytrn)), Flux.Tracker.data(loss(xtst,ytst)))
acc = hcat(accuracy(xtrn,ytrn),accuracy(xtst,ytst))

for i in 1:100
    Flux.train!(loss, Flux.params(model), minibatches, Flux.ADAM(), cb = Flux.throttle(callback, 1));
    global err = vcat(err, hcat(Flux.Tracker.data(loss(xtrn, ytrn)), Flux.Tracker.data(loss(xtst, ytst))))
    global acc = vcat(acc, hcat(accuracy(xtrn, ytrn), accuracy(xtst, ytst)))
end

# Save loss and to csv for visualization
write("error-flux.csv", DataFrame(err, [:training, :testing]))
write("accuracy-flux.csv", DataFrame(acc, [:training, :testing]))

Flux.@epochs 100 Flux.train!(loss,Flux.params(model),minibatches,Flux.ADAM(), cb = Flux.throttle(callback,1))

#Evaluate the model
trn_yidx = Flux.onecold(model(xtrn'))
tst_yidx = Flux.onecold(model(xtrn'))

accuracy(xtrn,ytrn)
accuracy(xtst,ytst)
