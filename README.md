# DQN Solver for POMDPS.jl

### Requires the user to provide the following:
1. `s, readout, wgts = createNetwork(ACTIONS)`
  * your Tensorflow graph is created in `createNetwork`
  * `s` is input layer, e.x. `s = placeholder(Float32, shape=[nothing, input_dim], name="input")`
  * `readout` is the output layer, e.x. `readout = h1*W2 + b2`
  * `wgts` is a list of all the network weights, e.x. `[W1, b1, W2, b2]`
2. `s_t, r_0, is_terminal, s_0 = frame_step(action, prev_state)`
  * `frame_step` is the only function that interacts with the environment, e.x. OpenAI envs
  * `s_t` is the state after executing `action` from the previous state
  * `r_0` is the reward after executing `action` from previous state and entering into `s_t`
  * `is_terminal` is a boolean if `s_t` is a terminal state
  * `s_0` is the initial state or state returned after reseting of the environment if `is_terminal`, otherwise it is `nothing`
3. `s_t = preprocess(x, prev_state)`
  * if your network requires preprocessing, e.g. if `x` is an image, your `preprocess` function should grayscale and downsample it

### For Tensorboard, navigate to `dqn_julia` directory, in cmdline

`tensorboard --logdir=logs`


