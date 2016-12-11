###########################################################################
# This codes the core code for the DQN. 
# The user should have NO interactions with this file directly. Changing
# this file will crash the rest of the program if the user is not careful.
###########################################################################

module DQN

using Gym
using TensorFlow

include("ReplayMemory.jl")

export trainDQN, simulateDQN, HyperParameters

###########################################
# HyperParameters contains all the fields
# the user needs to tune to improve
# DQN performance
###########################################
type HyperParameters
    alpha::Float32 # learning rate
    gamma::Float32 # decay rate of past observations
    observe::Int # frames to observe before training
    explore::Float32 # frames over which to anneal epsilon
    final_epsilon::Float32 # final value of epsilon
    initial_epsilon::Float32 # starting value of epsilon
    replay_mem_size::Int # number of previous transitions to remember
    batchsize::Int # size of minibatch
    k::Int # only select an action every Kth frame, repeat prev for others
    target_update_freq::Int # update frequency for weights of target network
    max_num_episodes::Int

    actions::Int
    state_shape

    function HyperParameters(actions, state_shape)
        alpha = 3e-5
        gamma = 0.99
        observe = 1000
        explore = 4000.
        final_epsilon = 0.05
        initial_epsilon = 1.0
        replay_mem_size = 20000
        batchsize = 32
        k = 1
        target_update_freq = 7500
        max_num_episodes = 100000
        new(alpha, gamma, observe, explore, final_epsilon, initial_epsilon, replay_mem_size, batchsize, k, target_update_freq, max_num_episodes, actions, state_shape)
    end
end


###########################################
# ExperienceBatch: 
#     s = current state
#     a = action 
#     r = reward
#     s1 = next state
#     is_terminal
###########################################
type ExperienceBatch
    s::Array{Float32}
    a::Array{Int32}
    r::Array{Float32}
    s1::Array{Float32}
    is_terminal::Array{Bool}

    function ExperienceBatch(BATCHSIZE, STATE_SHAPE)
        new(zeros(Float32, BATCHSIZE, STATE_SHAPE...),
            zeros(Int32, BATCHSIZE),
            zeros(Float32, BATCHSIZE),
            zeros(Float32, BATCHSIZE, STATE_SHAPE...),
            zeros(Bool, BATCHSIZE))
    end
end

###########################################
# trainNetwork: user should not be
# interacting with this function, the
# core of DQN training happens here, as in
# this is where Algorithm 1 from the 2015
# DeepMind paper is implemented
###########################################
function trainNetwork(frame_step, s, readout, wgts, s_target, readout_target, wgts_target, sess, hyper_params, wgts_dir, logs_dir)
    # one hot vector of the action taken
    a = placeholder(Int32, shape=[nothing], name="action")
    # scalar for r + gamma max_a' Q(s',a';theta_i^') from target
    y = placeholder(Float32, shape=[nothing], name="expected_reward")
    # dot product to get Q(s,a;theta_i) from main
    readout_action = reduce_sum(readout.*one_hot(a + 1, hyper_params.actions), reduction_indices=[2])
    #reg = 0.001*sum([reduce_sum(w.^2) for w=wgts])
    # [ (r + gamma max_a' Q(s',a';theta_i^'))  -  Q(s,a;theta_i) ]^2
    loss = reduce_mean((y - readout_action).^2) # + reg
    # use adam update rule
    train_step = train.minimize(train.AdamOptimizer(hyper_params.alpha), loss, var_list=wgts)

    histogram_summary("inputs/action", a)
    histogram_summary("qvalues/action_1", slice(readout, [0, 0], [-1, 1]))
    histogram_summary("qvalues/action_2", slice(readout, [0, 1], [-1, 1]))

    total_reward_op = placeholder(Float32, shape=[])
    scalar_summary("TotalReward", total_reward_op)

    per_episode_summaries = merge_all_summaries()

    loss_summary = scalar_summary("Loss", loss)

    summary_writer = train.SummaryWriter(logs_dir)

    saver = train.Saver()
    saver.max_to_keep = 50

    # store the previous observations in replay memory
    D = ReplayMemory{Array{Float32, length(hyper_params.state_shape)}}(hyper_params.replay_mem_size)

    # initialize state
    s_t, _, is_terminal, _ = frame_step(0, nothing)

    # must initialize tf vars before accessing
    run(sess, initialize_all_variables())

    update_target_weights = [assign(vars[1], vars[2]) for vars=zip(wgts_target, wgts)]

    minibatch = ExperienceBatch(hyper_params.batchsize, hyper_params.state_shape)

    # start training
    epsilon = hyper_params.initial_epsilon
    t = 0
    episode = 0
    # debugging
    while episode < hyper_params.max_num_episodes
        total_reward = 0
        is_terminal = false
        while !is_terminal
            # update target weights to match main weights
            if t % hyper_params.target_update_freq == 0
                run(sess, update_target_weights)
            end

            ## choose an action epsilon greedily
            a_t = 0
            if rand() <= epsilon || t <= hyper_params.observe
                a_t = rand(UInt) % hyper_params.actions
            else
                # readout_t = [Q(s,a;theta_i) for all a in hyper_params.actions]
                readout_t = run(sess, readout,  Dict(s=>reshape(s_t, 1, size(s_t)...)))
                a_t = indmax(readout_t) - 1
            end

            if epsilon > hyper_params.final_epsilon && t > hyper_params.observe
                epsilon -= (hyper_params.initial_epsilon - hyper_params.final_epsilon) / hyper_params.explore
            end

            # run same action K=1 times
            for _=1:hyper_params.k
                # run the selected action and observe next state and reward
                s_t1, r_t, is_terminal, s_0 = frame_step(a_t, s_t)
                total_reward += r_t

                # store the transition in D
                push_memory!(D, s_t, convert(Int32, a_t), r_t, s_t1, is_terminal)

                s_t = s_t1
                if is_terminal
                    s_t = s_0
                    break
                end
            end

            if t == hyper_params.observe
                println("starting training")
            end
            # only train if done observing
            if t > hyper_params.observe
                # sample a minibatch to train on
                sample!(D, hyper_params.batchsize, minibatch.s, minibatch.a, minibatch.r, minibatch.s1, minibatch.is_terminal)

                y_batch = Float32[]
                # readout_j1_batch = [Q(s',a'; theta_i^') for all a in hyper_params.actions]
                readout_j1_batch = run(sess, readout_target, Dict(s_target=>minibatch.s1))
                for i=1:hyper_params.batchsize
                    # if is_terminal, only expect reward
                    if minibatch.is_terminal[i]
                        push!(y_batch, minibatch.r[i])
                    # otherwise, need future reward from best action from current state
                    else
                        push!(y_batch, minibatch.r[i] + hyper_params.gamma * maximum(readout_j1_batch[i,:]))
                    end
                end

                ## perform gradient step
                dic = Dict(y=>y_batch,
                          a=>minibatch.a,
                          s=>minibatch.s,
                          total_reward_op=>total_reward)

                # run is where we compute [y - Q(s,a;theta_i)]^2 and do the update
                _, summaries = run(sess, [train_step, loss_summary], dic)
                if t % 100 == 0
                    write(summary_writer, summaries, t)
                end

                if is_terminal
                    # logs to TensorFlow every episode
                    write(summary_writer, run(sess, per_episode_summaries, dic), t)
                end
            end

            t += 1
        end
        @printf("Finished episode %5d. Reward=%8.3f\n", episode, total_reward)
        if episode % 100 == 0
            train.save(saver, sess, wgts_dir, global_step=episode)
        end
        episode += 1
    end
end

###########################################
# trainDQN: trains the DQN while saving
# video, wgts and tensorboard logs 
###########################################
function trainDQN(env, frame_step, createNetwork, hyper_params::HyperParameters, save_path=nothing)
    if save_path == nothing
        save_path = env.name
    end
    mkpath(save_path)
    wgts_dir = joinpath(save_path, "saved_wgts")
    mkpath(wgts_dir)
    wgts_dir = joinpath(wgts_dir, "weights") # for naming conventions in Tensorflow
    logs_dir = joinpath(save_path, "logs")
    mkpath(logs_dir)

    start_monitor(env, joinpath(save_path, "videos"))
    reset(env) # reset the environment
    # create tf session
    sess = Session()
    # training DQN
    s, readout, wgts = createNetwork(hyper_params, "main_network")
    # check point DQN, only gets updated occassionally to preserve stability
    s_target, readout_target, wgts_target = createNetwork(hyper_params, "target_network")
    trainNetwork(frame_step, s, readout, wgts, s_target, readout_target, wgts_target, sess, hyper_params, wgts_dir, logs_dir)
    close_monitor(env)
end

###########################################
# simulateDQN: simmulates the DQN for set
# number of episodes
###########################################
function simulateDQN(env, frame_step, createNetwork, saved_wgts_path, num_sim_episode, hyper_params::HyperParameters)
    start_monitor(env, "/tmp/dqn/monitor/exp_$(env.name)_$(now())")
    reset(env) # reset the environment
    # create tf session
    sess = Session()
    s, readout, wgts = createNetwork(hyper_params, "main_network")
    _, _, _ = createNetwork(hyper_params, "target_network")

    saver = train.Saver()
    train.restore(saver, sess, saved_wgts_path)

    # initialize state
    s_t, _, _, _ = frame_step(0, nothing)

    t = 0
    episode = 0
    while episode < num_sim_episode
        total_reward = 0
        is_terminal = false
        while !is_terminal
            # readout_t = [Q(s,a;theta_i) for all a in hyper_params.actions]
            readout_t = run(sess, readout,  Dict(s=>reshape(s_t, 1, size(s_t)...)))
            a_t = indmax(readout_t) - 1

            s_t1, r_t, is_terminal, s_0 = frame_step(a_t, s_t)
            render(env)
            total_reward += r_t
            s_t = s_t1
            if is_terminal
                s_t = s_0
                break
            end
        end
        @printf("Finished episode %5d. Reward=%8.3f\n", episode, total_reward)
        episode += 1
    end
    close_monitor(env)
end

end
