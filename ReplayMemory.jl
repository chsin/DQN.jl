###########################################################################
# This codes the replay memory custom type for storing exeriences.
# The user should have NO interactions with this file directly. Changing
# this file will crash the rest of the program if the user is not careful.
###########################################################################

type ReplayMemory{State}
    # implemented as a circular deque
    cur_ind::Int
    max_len::Int
    full::Bool

    s::Array{State, 1}
    a::Array{Int32, 1}
    r::Array{Float32, 1}
    s1::Array{State, 1}
    is_terminal::Array{Bool, 1}

    function ReplayMemory(max_len::Int)
        s = Array(State, max_len)
        a = Array(Int32, max_len)
        r = Array(Float32, max_len)
        s1 = Array(State, max_len)
        is_terminal = Array(Bool, max_len)

        cur_ind = 1
        full = cur_ind == max_len
        new(cur_ind, max_len, full, s, a, r, s1, is_terminal)
    end
end

function push_memory!{State}(D::ReplayMemory{State}, s, a, r, s1, is_terminal)
    if D.cur_ind > D.max_len
        D.full = true
        D.cur_ind = 1
    end
    D.s[D.cur_ind] = s
    D.a[D.cur_ind] = a
    D.r[D.cur_ind] = r
    D.s1[D.cur_ind] = s1
    D.is_terminal[D.cur_ind] = is_terminal
    D.cur_ind+=1;
end

function sample!{State}(D::ReplayMemory{State}, batch_size::Int, s_batch, a_batch, r_batch, s1_batch, is_terminal_batch)
    deque_end = D.full ? D.max_len : D.cur_ind-1
    indices = rand(1:deque_end, batch_size)
    for i=1:length(indices)
        s_batch[i,:] = D.s[indices[i]]
    end
    for i=1:length(indices)
        a_batch[i] = D.a[indices[i]]
    end
    for i=1:length(indices)
        r_batch[i] = D.r[indices[i]]
    end
    for i=1:length(indices)
        s1_batch[i,:] = D.s1[indices[i]]
    end
    for i=1:length(indices)
        is_terminal_batch[i] = D.is_terminal[indices[i]]
    end
end
