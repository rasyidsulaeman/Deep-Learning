using DataFrames, CSV
using StatsPlots
using Flux
using Statistics
using Base.Iterators: repeated

# Importing the dataset
dataset = DataFrame(CSV.File("data.txt", header=false))
newnames = ["Score_1", "Score_2", "Result"]
names!(dataset, Symbol.(newnames))

# Visualize the original data
@df dataset scatter(:Score_1, :Score_2,
                zcolor = :Result, label=false)
xaxis!("Score in Test 1")
yaxis!("Score in Test 2")
#savefig("original_data.png")

# Convert the dataset into array
xdata = Matrix(dataset[:,1:2])
ydata = dataset[:,end]

# Applying the standard score, in order to standardize the raw data point
xdata = (xdata .- mean(xdata, dims=1)) ./ std(xdata, dims=1)

# Data iterators
data = repeated((xdata, ydata), 200)

# Initialized parameters
W = Flux.param(zeros(Float64,2))
b = Flux.param([0.0])

# Tracked parameters
parameters = Flux.Params([W,b])

# Appyling the activation function / predicting function
ŷ(x) = NNlib.sigmoid.(x * W .+ b)

# Define loss function a.k.a cost function
loss(x,y) = sum(Flux.binarycrossentropy.(ŷ(x),y)) / size(x)[1]

# Define the callback function
callback() = @show(loss(xdata,ydata))

err = Flux.Tracker.data(loss(xdata,ydata))

# Train the model
for i in 1:25
    Flux.train!(loss, parameters, data, Flux.ADAM(), cb = Flux.throttle(callback,1))
    global err = vcat(err, Flux.Tracker.data(loss(xdata,ydata)))
end

# Visualize the cost function
plot(err, lw=2, label=false)
xaxis!("Number of interations")
yaxis!("Cost function")
savefig("cost.png")


# Test the accuracy
function accuracy(x,y)
    predict = ŷ(x) .> 0.5
    return sum(predict .== y) / size(y)[1]
end

acc = Flux.Tracker.data(accuracy(xdata,ydata))

# Visualize the result
xrange = (-2,2)
xs = LinRange(xrange...,25)
Z = [Flux.Tracker.data(ŷ(hcat(x,y))[1]) for x=xs, y=xs]

contour(xs,xs,Z, levels=1)
scatter!(xdata[:,1],xdata[:,2], zcolor = ydata, label=false)
xaxis!("Score 1 (Z-score)")
yaxis!("Score 2 (Z-score)")
savefig("result.png")
