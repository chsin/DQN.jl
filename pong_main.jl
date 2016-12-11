using Gym
using Images
using Colors
include("convnet1.jl") # createNetwork defined here
include("dqn.jl")
using DQN

TIME_FRAMES = 4 # number of time frames in the state

env = GymEnvironment("Pong-v0")
@show ACTIONS = n_actions(env)         # number of valid actions
# @show STATE_SHAPE = obs_dimensions(env)[0]
@show STATE_SHAPE = (80, 80, TIME_FRAMES)

function preprocess(x, prev_state)
    grayscale = convert(Image{Gray}, colorim(x))
    resized = Images.imresize(grayscale, (80, 80))
    x = convert(Array{Float32}, data(resized))
    if prev_state == nothing
        cat(3, x, x, x, x)
    else
        cat(3, x, prev_state[:, :, 1:end-1])
    end
end

function frame_step(action, prev_state)
    x_t, r_t, is_terminal = step!(env, action)
    # render(env)
    s_t = preprocess(x_t, prev_state)
    s_0 = is_terminal ? preprocess(reset(env), nothing) : nothing
    s_t, r_t / 200.0, is_terminal, s_0
end

hyper_params = HyperParameters(ACTIONS, STATE_SHAPE)

trainDQN(env, frame_step, createNetwork, hyper_params)

#simulateDQN(env, frame_step, createNetwork, "/Pong-v0/saved_wgts/weights-300", 2, hyper_params)
