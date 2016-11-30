using TensorFlow
include("utils.jl");

# assumes imput dim is 80x80x4
function createNetwork(ACTIONS)
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = placeholder(Float32, shape=[nothing, 80, 80, 4], name="input")

    # hidden layers
    h_conv1 = nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = reshape(h_conv3, [-1, 1600])
    h_fc1 = nn.relu(h_conv3_flat*W_fc1 + b_fc1)

    # readout layer
    readout = h_fc1*W_fc2 + b_fc2

    return s, readout, [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
end
