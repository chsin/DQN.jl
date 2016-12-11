using TensorFlow
include("utils.jl");

function createNetwork(hyper_params, prefix, hidden_dim=512)
    @assert length(hyper_params.state_shape) == 1
    input_dim = hyper_params.state_shape[1]
    # network weights
    W1 = weight_variable([input_dim, hidden_dim])
    b1 = bias_variable([hidden_dim])

    W2 = weight_variable([hidden_dim, hyper_params.actions])
    b2 = bias_variable([hyper_params.actions])

    histogram_summary("$prefix/weight/W1", W1)
    histogram_summary("$prefix/weight/b1", b1)
    histogram_summary("$prefix/weight/W2", W2)
    histogram_summary("$prefix/weight/b2", b2)

    # input layer
    s = placeholder(Float32, shape=[nothing, input_dim], name="input")

    # hidden layer
    h1 = nn.relu(s*W1 + b1)

    # readout layer
    readout = h1*W2 + b2

    return s, readout, [W1, b1, W2, b2]
end
