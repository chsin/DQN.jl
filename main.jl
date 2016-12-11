using Gym
using TensorFlow
using Distributions
using Images
using Colors
include("utils.jl")
include("ReplayMemory.jl")
include("convnet1.jl")

ALPHA = 3e-5 # learning rate
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1000 # frames to observe before training
EXPLORE = 4000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEM_SIZE = 20000 # number of previous transitions to remember
BATCHSIZE = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others
TARGET_UPDATE_FREQ = 7500 # update frequency for weights of target network
MAX_NUM_EPISODES = 100000

TIME_FRAMES = 4 # number of time frames in the state


env = GymEnvironment("Pong-v0")
@show ACTIONS = n_actions(env)         # number of valid actions
# @show STATE_SHAPE = obs_dimensions(env)[0]
@show STATE_SHAPE = (80, 80, TIME_FRAMES)
STATE_SHAPE_DIMS = length(STATE_SHAPE)

typealias State Array{Float32,  3}

type ExperienceBatch
    s::Array{Float32, STATE_SHAPE_DIMS + 1}
    a::Array{Int32, 1}
    r::Array{Float32, 1}
    s1::Array{Float32, STATE_SHAPE_DIMS + 1}
    is_terminal::Array{Bool, 1}

    function ExperienceBatch(BATCHSIZE)
        new(zeros(Float32, BATCHSIZE, STATE_SHAPE...),
            zeros(Int32, BATCHSIZE),
            zeros(Float32, BATCHSIZE),
            zeros(Float32, BATCHSIZE, STATE_SHAPE...),
            zeros(Bool, BATCHSIZE))
    end
end

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

function trainNetwork(frame_step, s, readout, wgts, s_target, readout_target, wgts_target, sess)
    # one hot vector of the action taken
    a = placeholder(Int32, shape=[nothing], name="action")
    # scalar for r + gamma max_a' Q(s',a';theta_i^') from target
    y = placeholder(Float32, shape=[nothing], name="expected_reward")
    # dot product to get Q(s,a;theta_i) from main
    readout_action = reduce_sum(readout.*one_hot(a + 1, ACTIONS), reduction_indices=[2])
    #reg = 0.001*sum([reduce_sum(w.^2) for w=wgts])
    # [ (r + gamma max_a' Q(s',a';theta_i^'))  -  Q(s,a;theta_i) ]^2
    loss = reduce_mean((y - readout_action).^2) # + reg
    # use adam update rule
    train_step = train.minimize(train.AdamOptimizer(ALPHA), loss, var_list=wgts)

    histogram_summary("inputs/action", a)
    histogram_summary("qvalues/action_1", slice(readout, [0, 0], [-1, 1]))
    histogram_summary("qvalues/action_2", slice(readout, [0, 1], [-1, 1]))

    total_reward_op = placeholder(Float32, shape=[])
    scalar_summary("TotalReward", total_reward_op)

    per_episode_summaries = merge_all_summaries()

    loss_summary = scalar_summary("Loss", loss)

    log_dir = "/tmp/dqn/logs/$(now())"
    mkpath(log_dir)
    summary_writer = train.SummaryWriter(log_dir)

    saver = train.Saver()

    # store the previous observations in replay memory
    D = ReplayMemory{State}(REPLAY_MEM_SIZE)

    # initialize state
    s_t, _, is_terminal, _ = frame_step(0, nothing)

    # must initialize tf vars before accessing
    run(sess, initialize_all_variables())

    update_target_weights = [assign(vars[1], vars[2]) for vars=zip(wgts_target, wgts)]

    minibatch = ExperienceBatch(BATCHSIZE)

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    episode = 0
    # debugging
    while episode < MAX_NUM_EPISODES
        total_reward = 0
        is_terminal = false
        while !is_terminal
            # update target weights to match main weights
            if t % TARGET_UPDATE_FREQ == 0
                run(sess, update_target_weights)
            end

            ## choose an action epsilon greedily
            a_t = 0
            if rand() <= epsilon || t <= OBSERVE
                a_t = rand(UInt) % ACTIONS
            else
                # readout_t = [Q(s,a;theta_i) for all a in ACTIONS]
                readout_t = run(sess, readout,  Dict(s=>reshape(s_t, 1, STATE_SHAPE...)))
                a_t = indmax(readout_t) - 1
            end

            if epsilon > FINAL_EPSILON && t > OBSERVE
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            end

            # run same action K=1 times
            for _=1:K
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

            if t == OBSERVE
                println("starting training")
            end
            # only train if done observing
            if t > OBSERVE
                # sample a minibatch to train on
                sample!(D, BATCHSIZE, minibatch.s, minibatch.a, minibatch.r, minibatch.s1, minibatch.is_terminal)

                y_batch = Float32[]
                # readout_j1_batch = [Q(s',a'; theta_i^') for all a in ACTIONS]
                readout_j1_batch = run(sess, readout_target, Dict(s_target=>minibatch.s1))
                for i=1:BATCHSIZE
                    # if is_terminal, only expect reward
                    if minibatch.is_terminal[i]
                        push!(y_batch, minibatch.r[i])
                    # otherwise, need future reward from best action from current state
                    else
                        push!(y_batch, minibatch.r[i] + GAMMA * maximum(readout_j1_batch[i,:]))
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
            train.save(saver, sess, "./saved_weights/weights", global_step=episode)
        end
        episode += 1
    end
end

function runDQN(frame_step)
    start_monitor(env, "/tmp/dqn/monitor/exp_$(env.name)_$(now())")
    reset(env) # reset the environment
    # create tf session
    sess = Session()
    # training DQN
    s, readout, wgts = createNetwork(ACTIONS, "main_network", sess)
    # check point DQN, only gets updated occassionally to preserve stability
    s_target, readout_target, wgts_target = createNetwork(ACTIONS, "target_network", sess)
    trainNetwork(frame_step, s, readout, wgts, s_target, readout_target, wgts_target, sess)
    close_monitor(env)
end

function simulateDQN(env, frame_step, createNetwork, saved_wgts_path, num_sim_episode)
    start_monitor(env, "/tmp/dqn/monitor/exp_$(env.name)_$(now())")
    reset(env) # reset the environment
    # create tf session
    sess = Session()
    s, readout, wgts = createNetwork(ACTIONS, "main_network", sess)
    _, _, _ = createNetwork(ACTIONS, "target_network", sess)

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
            # readout_t = [Q(s,a;theta_i) for all a in ACTIONS]
            readout_t = run(sess, readout,  Dict(s=>reshape(s_t, 1, STATE_SHAPE...)))
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

# runDQN(frame_step)

simulateDQN(env, frame_step, createNetwork, "/tmp/saved_weights/weights-300", 2)
