using TensorFlow
include("utils.jl");

# assumes input dim is (80, 80, 4)
function createNetwork(hyper_params, prefix, sess=nothing)
    @assert hyper_params.state_shape == (80, 80, 4)
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, hyper_params.actions])
    b_fc2 = bias_variable([hyper_params.actions])

    # image_summary("$prefix/weight/W_conv1", W_conv1)
    # image_summary("$prefix/weight/b_conv1", b_conv1)
    # image_summary("$prefix/weight/W_conv2", W_conv2)
    # image_summary("$prefix/weight/b_conv2", b_conv2)
    # image_summary("$prefix/weight/W_conv3", W_conv3)
    # image_summary("$prefix/weight/b_conv3", b_conv3)
    # histogram_summary("$prefix/weight/W_fc1", W_fc1)
    # histogram_summary("$prefix/weight/b_fc1", b_fc1)
    # histogram_summary("$prefix/weight/W_fc2", W_fc2)
    # histogram_summary("$prefix/weight/b_fc2", b_fc2)

    # input layer
    s = placeholder(Float32, shape=[nothing, 80, 80, 4], name="input")

    # resized_s = image.resize_images(pad(s, [0 0; 0 0; 0 0; 0 1]), 80, 80)

    # hidden layers
    h_conv1 = nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = reshape(h_conv3, [-1, 1600])
    h_fc1 = nn.relu(h_conv3_flat*W_fc1 + b_fc1)

    # readout layer
    readout = h_fc1*W_fc2 + b_fc2

    # run(sess, initialize_all_variables())
    # println("1 ", size(run(sess, resized_s, Dict(s => zeros(Float32, 1, 210, 160, 3)))))
    # println("2 ", size(run(sess, h_conv1, Dict(s => zeros(Float32, 1, 210, 160, 3)))))
    # println("3 ", size(run(sess, h_pool1, Dict(s => zeros(Float32, 1, 210, 160, 3)))))
    # println("4 ", size(run(sess, h_conv2, Dict(s => zeros(Float32, 1, 210, 160, 3)))))
    # println("5 ", size(run(sess, h_conv3, Dict(s => zeros(Float32, 1, 210, 160, 3)))))
    # println("6 ", size(run(sess, h_conv3_flat, Dict(s => zeros(Float32, 1, 210, 160, 3)))))
    # println("7 ", size(run(sess, h_fc1, Dict(s => zeros(Float32, 1, 210, 160, 3)))))
    # println("8 ", size(run(sess, readout, Dict(s => zeros(Float32, 1, 210, 160, 3)))))

    return s, readout, [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
end
