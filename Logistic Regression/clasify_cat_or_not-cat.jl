# Importing the packages
using HDF5
using Statistics: mean
using PyPlot

#Create a type names parameters and gradiens
struct parameters{T<:Float64}
    w::Array{T,2}
    b::T
end

struct gradiens
    dw
    db
end

"""
load dataset :
contains a data which clasify between a non-cat picture and a cat picture
"""
function load_dataset()

    labels = ["train", "test"]
    dataset = Dict()
    for i in labels
        h5open("dataset/$(i)_catvnoncat.h5", "r") do file
            dataset["$i"] = read(file)
        end
    end

    xtrain = permutedims(Int.(dataset["train"]["train_set_x"]), (4,3,2,1))
    ytrain = Int.(dataset["train"]["train_set_y"])'

    xtest = permutedims(Int.(dataset["test"]["test_set_x"]), (4,3,2,1))
    ytest = Int.(dataset["test"]["test_set_y"])'

    classes = dataset["train"]["list_classes"]

    return (xtrain, ytrain, xtest, ytest, classes)
end

xtrain, ytrain, xtest, ytest, classes = load_dataset()

# Let's look our dataset
index = 1
imshow(xtrain[index,:,:,:])
println("y = [$index], It's a $(classes[ytrain[:,index][1] + 1]) picture")

x_train_flatten = reshape(xtrain,size(xtrain)[1], :)'
x_test_flatten = reshape(xtest,size(xtest)[1], :)'

x_train_norm = x_train_flatten/255.
x_test_norm = x_test_flatten/255.

sigmoid(z) = 1 / (1 + exp(-z))

loss(x,y) = y .* log.(x) .+ (1 .- y) .* log.(1 .- x)

function propagate(w,b,x,y)

    m = size(x)[2]

    #Forward propagation
    a = sigmoid.(w' * x .+ b)

    #Cost function
    cost = - sum(loss(a,y)) / m

    #Backward propagation
    dw = x * (a .- y)' / m
    db = sum(a .- y) / m

    grads = gradiens(dw,db)
    return grads, cost
end

function optimize(p::parameters, x, y, num_iterations, α)

    w = p.w
    b = p.b

    costs = []
    for i in 1:num_iterations
        grads, cost = propagate(w,b,x,y)

        dw = grads.dw
        db = grads.db

        w = w - α * dw
        b = b - α * db

        if i % 100 == 0
            push!(costs, cost)
        elseif i == 1 || i % 500 == 0
            println("Cost after iterations $i = $(cost)")
        end
    end

    params = parameters(w,b)

    return params, costs
end

accuracy(y,ŷ) = 100 - mean(abs.(ŷ .- y)) * 100

function predict(w, b, x)

    y_hat = zeros(eltype(x),1,size(x)[2])

    a = sigmoid.(w' * x .+ b)

    [a[1,i] >= 0.5 ? y_hat[1,i] = 1.0 : y_hat[1,i] = 0.0 for i in 1:size(a)[2]]

    return y_hat
end

function model(xtrain,ytrain,xtest,ytest; num_iterations = 5000, learning_rate = 0.005)

    #Initialized parameters
    w = zeros(eltype(xtrain),size(xtrain)[1],1)
    b = 0.0
    p = parameters(w,b)

    params, costs = optimize(p, xtrain, ytrain, num_iterations, learning_rate)

    #Calculate the accuracy and prediction test
    ŷtrain = predict(params.w, params.b, xtrain)
    ŷtest  = predict(params.w, params.b, xtest)

    println("Train accuracy : $(accuracy(ytrain, ŷtrain)) %")
    println("Test accuracy  : $(accuracy(ytest, ŷtest)) %")

    compile = Dict( "parameters" => params,
                    "costs" => costs,
                    "y_predict_train" => ŷtrain,
                    "y_predict_test" => ŷtest,
                    "num_iters" => num_iterations,
                    "α" => learning_rate)
    return compile
end

#Test the model
compile = model(x_train_norm, ytrain, x_test_norm, ytest)

# Check whether the prediction gives a non-cat or cat picture
index = 1
imshow(xtest[index,:,:,:])
println("At y = $(ytest[1,index]),you predicted that it is
        $(classes[Int.(compile["y_predict_test"][1,index] + 1)]) picture")

# Visualize the costs function
costs = compile["costs"]
plot(costs)
xlabel("Number of iterations")
ylabel("Costs")
show()
