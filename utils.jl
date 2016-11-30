function weight_variable(shape)
    initial = map(Float32, rand(Normal(0, .001), shape...))
    return Variable(initial)
end

function bias_variable(shape)
    initial = fill(Float32(.1), shape...)
    return Variable(initial)
end

function conv2d(x, W, stride=1)
    nn.conv2d(x, W, [1, stride, stride, 1], "SAME")
end

function max_pool_2x2(x)
    nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
end
