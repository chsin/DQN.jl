using Gym
using TensorFlow
using Distributions
include("utils.jl")
include("ReplayMemory.jl")

ALPHA = 3e-5 # learning rate
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1000 # frames to observe before training
EXPLORE = 4000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEM_SIZE = 590000 # number of previous transitions to remember
BATCHSIZE = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others
TARGET_UPDATE_FREQ = 150 # 7500 # update frequency for weights of target network
MAX_NUM_EPISODES = 100000


env = GymEnvironment("CartPole-v0")
@show ACTIONS = n_actions(env)         # number of valid actions
@assert length(obs_dimensions(env)) == 1
@show STATE_DIMS = obs_dimensions(env)[1]

function createNetwork(ACTIONS, input_dim, prefix, hidden_dim=512)
    # network weights
    W1 = weight_variable([input_dim, hidden_dim])
    b1 = bias_variable([hidden_dim])

    W2 = weight_variable([hidden_dim, ACTIONS])
    b2 = bias_variable([ACTIONS])

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


function frame_step(action)
    x_t, r_t, is_terminal = step!(env, action)
    s_t = x_t
    s_0 = is_terminal ? reset(env) : nothing
    s_t, r_t / 200.0, is_terminal, s_0
end

function unpack_memory(minibatch, BATCHSIZE)
    s_j_batch = zeros(Float64, BATCHSIZE, STATE_DIMS)
    a_batch = zeros(Int, BATCHSIZE)
    r_batch = zeros(Float64, BATCHSIZE)
    s_j1_batch = zeros(Float64, BATCHSIZE, STATE_DIMS)

    for i=1:BATCHSIZE
        s_j_batch[i,:] = minibatch[i][1]
        a_batch[i] = minibatch[i][2]
        r_batch[i] = minibatch[i][3]
        s_j1_batch[i,:] = minibatch[i][4]
    end
    s_j_batch, a_batch, r_batch, s_j1_batch
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

    scalar_summary("Loss", loss)
    histogram_summary("inputs/action", a)
    histogram_summary("qvalues/action_1", slice(readout, [0, 0], [-1, 1]))
    histogram_summary("qvalues/action_2", slice(readout, [0, 1], [-1, 1]))

    episode_length_op = placeholder(Float64, shape=[])
    episode_length_summary = scalar_summary("AvgEpisodeLength", episode_length_op)

    all_summaries = merge_all_summaries()
    log_dir = "/tmp/dqn/logs/$(now())"
    mkpath(log_dir)
    summary_writer = train.SummaryWriter(log_dir)

    # store the previous observations in replay memory
    D = ReplayMemory(REPLAY_MEM_SIZE)

    # initialize state
    s_t, _, is_terminal, _ = frame_step(0)

    # must initialize tf vars before accessing
    run(sess, initialize_all_variables())

    update_target_weights = [assign(vars[1], vars[2]) for vars=zip(wgts_target, wgts)]

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    episode = 0
    # debugging
    while episode < MAX_NUM_EPISODES
        # update target weights to match main weights
        if episode % TARGET_UPDATE_FREQ == 0
            run(sess, update_target_weights)
        end

        episode_length = 0
        is_terminal = false
        while !is_terminal
            ## choose an action epsilon greedily
            a_t = 0
            if rand() <= epsilon || t <= OBSERVE
                a_t = rand(UInt) % ACTIONS
            else
                # readout_t = [Q(s,a;theta_i) for all a in ACTIONS]
                readout_t = run(sess, readout,  Dict(s=>s_t'))
                a_t = indmax(readout_t) - 1
            end

            if epsilon > FINAL_EPSILON && t > OBSERVE
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            end

            # run same action K=1 times
            for _=1:K
                # run the selected action and observe next state and reward
                s_t1, r_t, is_terminal, s_0 = frame_step(a_t)
                episode_length += 1

                # store the transition in D
                push_memory!(D, [s_t, a_t, r_t, s_t1, is_terminal])

                s_t = s_t1
                if is_terminal
                    s_t = s_0
                    break
                end
            end

            # only train if done observing
            if t > OBSERVE
                # sample a minibatch to train on
                minibatch = sample(D, BATCHSIZE)
                s_j_batch, a_batch, r_batch, s_j1_batch = unpack_memory(minibatch, BATCHSIZE)

                y_batch = Float64[]
                # readout_j1_batch = [Q(s',a'; theta_i^') for all a in ACTIONS]
                readout_j1_batch = run(sess, readout_target, Dict(s_target=>s_j1_batch))
                # readout_j1_batch = run(sess, readout, Dict(s=>s_j1_batch))
                for i=1:BATCHSIZE
                    # minibatch[i][5] = is_terminal, if is_terminal, only expect reward
                    if minibatch[i][5]
                        push!(y_batch, r_batch[i])
                    # otherwise, need future reward from best action from current state
                    else
                        push!(y_batch, r_batch[i] + GAMMA * maximum(readout_j1_batch[i,:]))
                    end
                end

                ## perform gradient step
                dic = Dict(y=>y_batch,
                           a=>a_batch,
                           s=>s_j_batch,
                           episode_length_op=>episode_length)

                # run is where we compute [y - Q(s,a;theta_i)]^2 and do the update
                if is_terminal
                    # logs to TensorFlow every episode
                    _, summaries = run(sess, [train_step, all_summaries], dic)
                    write(summary_writer, summaries, episode)
                else
                    run(sess, train_step, dic)
                end
            end

            t += 1
        end
        @printf("Finished episode %5d. Length=%3d\n", episode, episode_length)
        episode += 1
    end
end

function runDQN(frame_step)
    start_monitor(env, "/tmp/dqn/monitor/exp_$(env.name)_$(now())")
    reset(env) # reset the environment
    # create tf session
    sess = Session()
    # training DQN
    s, readout, wgts = createNetwork(ACTIONS, STATE_DIMS, "main_network")
    # check point DQN, only gets updated occassionally to preserve stability
    s_target, readout_target, wgts_target = createNetwork(ACTIONS, STATE_DIMS, "target_network")
    trainNetwork(frame_step, s, readout, wgts, s_target, readout_target, wgts_target, sess)
    close_monitor(env)
end

runDQN(frame_step)
