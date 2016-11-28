type ReplayMemory
    # implemented as a circular deque
    cur_ind::Int
    max_len::Int
    full::Bool
    deque

    function ReplayMemory(max_len::Int)
        deque = Array(Any, max_len)
        cur_ind = 1
        full = cur_ind == max_len
        new(cur_ind, max_len, full, deque)
    end
end

function push_memory!(D::ReplayMemory, item)
    if D.cur_ind > D.max_len
        D.full = true
        D.cur_ind = 1
    end
    D.deque[D.cur_ind] = item
    D.cur_ind+=1;
end

function sample(D::ReplayMemory, batch_size::Int)
    deque_end = D.full ? D.max_len : D.cur_ind-1
    rand(D.deque[1:deque_end], batch_size)
end