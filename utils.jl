###########################################################################
# This contains TensorFlow utils for building the example networks.
# The user should have NO interactions with this file directly. Changing
# this file will crash the examples if the user is not careful.
###########################################################################

using Distributions

function weight_variable(shape)
    initial = map(Float32, rand(Normal(0, 0.01), shape...))
    return Variable(initial)
end

function bias_variable(shape)
    initial = fill(Float32(.01), shape...)
    return Variable(initial)
end

function conv2d(x, W, stride=1)
    nn.conv2d(x, W, [1, stride, stride, 1], "SAME")
end

function max_pool_2x2(x)
    nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
end
