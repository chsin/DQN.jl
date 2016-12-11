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
