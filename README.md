# DQN Solver for POMDPS.jl

### Dependencies

Assuming TensorFlow and OpenAi Gym are already installed, extra dependencies include the following libraries (and their dependencies)

- Gym.jl
- DeepRL.jl
- Tensorflow.jl

### Instructions

Detailed tutorials are provided in the jupyter notebooks and examples are provided in `*_main.jl` files. But the code is repeated here for convenience.

```julia
using Gym
include("net1.jl") # createNetwork defined here
include("dqn.jl")
using DQN


env = GymEnvironment("CartPole-v0")
@show ACTIONS = n_actions(env)         # number of valid actions
@show STATE_SHAPE = obs_dimensions(env)

preprocess(x, prev_state=nothing) = x

function frame_step(action, prev_state)
    x_t, r_t, is_terminal = step!(env, action)
    # render(env)
    s_t = preprocess(x_t, prev_state)
    s_0 = is_terminal ? preprocess(reset(env), nothing) : nothing
    s_t, r_t / 200.0, is_terminal, s_0
end

hyper_params = HyperParameters(ACTIONS, STATE_SHAPE)

#trainDQN(env, frame_step, createNetwork, hyper_params)

simulateDQN(env, frame_step, createNetwork, "CartPole-v0/saved_wgts/weights-2000", 2, hyper_params)
```


### Requires the user to provide the following (assuming env given):
1. `s, readout, wgts = createNetwork(hyper_params, prefix)`
  * your Tensorflow graph is created in `createNetwork`
  * `s` is input layer, e.x. `s = placeholder(Float32, shape=[nothing, input_dim], name="input")`
  * `readout` is the output layer, e.x. `readout = h1*W2 + b2`
  * `wgts` is a list of all the network weights, e.x. `[W1, b1, W2, b2]`
  * `prefix` is just a string for Tensorboard logs
2. `s_t, r_t, is_terminal, s_0 = frame_step(action, prev_state)`
  * `frame_step` is the only function that interacts with the environment, e.x. OpenAI envs
  * `s_t` is the state after executing `action` from the previous state
  * `r_t` is the reward after executing `action` from previous state and entering into `s_t`
  * `is_terminal` is a boolean if `s_t` is a terminal state
  * `s_0` is the initial state or state returned after reseting of the environment if `is_terminal`, otherwise it is `nothing`
3. `s_t = preprocess(x, prev_state)`
  * if your network requires preprocessing, e.g. if `x` is an image, your `preprocess` function should grayscale and downsample it
3. `hyper_params = HyperParameters(ACTIONS, STATE_SHAPE)` where `ACTIONS` is the number of possible actions and `STATE_SHAPE` is a tuple, e.g. `STATE_SHAPE = (4,)` in Cartpole-v0.


### Tensorboard:

To use Tensorboard, navigate to the relevant experiment directory and enter

`tensorboard --logdir=logs`

on the comand line.

### Video examples of training sessions in progress

1. [CartPole-v0 training sesson clips](https://youtu.be/fDY96bwKw3M)

2. [Pong-v0 training sesson clips](https://youtu.be/_toBTIcEUpo)
