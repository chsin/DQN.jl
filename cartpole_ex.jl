env = GymEnvironment("CartPole-v0")
STATE_DIMS = obs_dimensions(env)[1]

preprocess(x, prev_state=nothing) = x

function frame_step(action)
    x_t, r_t, is_terminal = step!(env, action)
    s_t = preprocess(x_t)
    s_0 = is_terminal ? reset(env) : nothing
    s_t, r_t / 200.0, is_terminal, s_0
end

