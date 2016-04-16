--[[
    Example 6.5: Windy GridWorld
    Reinforcement Learning: An Antroduction by Sutton 2015
]]--

nActions = 4
actions = {       left=1, right=2, up=3, down=4}
actionStrings = {'left', 'right', 'up', 'down'}

--nCols = 10
--nRows = 7
--nStates = nCols*nRows;

-- external dynamics.
-- The state of the agent is forced to move when it goes to a location of cloumn c
wind = {}
wind.move = {0,0,0,-1,-1,-1,-2,-2,-1,0} -- up direction
wind.action = function(c) -- column
    if c<1 or c>map.nCols then return 0 end
    return wind.move[c]
end


map={} -- 2D map of nRows x nCols
map.Start = {4,1} -- Start position
map.Goal = {4,8} -- Goal position
map.nCols = 10
map.nRows = 7
map.stateId = function(coord)
    local r=coord[1] c=coord[2]
    return (r-1)*map.nCols + c
end
map.coord = function(sid)
    local rm1 = math.floor(sid/map.nCols)
    return { rm1+1, sid%map.nCols}
end
map.reward = function(coord)
    local s = coord
    --local coord = map.coord(s)
    if s[1]==map.Goal[1] and s[2]==map.Goal[2] then return 0 else return -1 end
end

-- Q function of the agent,
-- size: nStates x nActions
Q={}
Q.nStates = map.nRows*map.nCols
-- 2D array
Q.array = {}
Q.init = function ()
    for s=1, Q.nStates do
        local arr = {}
        for a=1, nActions do
            arr[a] = 0
        end
        Q.array[s] = arr;
    end
end


Q.eps = 0.01

Q.chooseAction = function (coord) -- e-Greedy
    local p = math.random() -- uniform dist [0,1)
    local achoice;
    if p<Q.eps then
        -- choose any action randonly among actions
        achoice = math.random(1, nActions)
        --[[
        print (string.format('Q.chooseAction(): %s by eps %f', actionStrings[achoice], p))
        --]]
    else
        local s = map.stateId (coord)
        achoice = 1
        for a=1, nActions do -- find max
            if Q.array[s][achoice] < Q.array[s][a] then achoice = a end
        end
        --[[
        print (string.format('Q.chooseAction(): %8s by Greedy Q=[%.2f %.2f %.2f %.2f]',
            actionStrings[achoice], Q.f(s,1), Q.f(s,2), Q.f(s,3), Q.f(s,4)))
        --]]
    end
    return actionStrings[achoice]
end

Q.alpha = 0.5
Q.discount = 1
Q.update = function (S,A,R,Snext,Anext)
    local s1 = map.stateId(S)
    local a1 = actions[A]
    local s2 = map.stateId(Snext)
    local a2 = actions[Anext]
    Q.array[s1][a1] = Q.array[s1][a1]
        + Q.alpha * (R + Q.discount * Q.array[s2][a2] - Q.array[s1][a1])
    --[[
    print (string.format('%s %s',
                string.format('Q.update: %d %s(%d) %d %d %s(%d)',
                    s1, A, a1, R, s2, Anext, a2),
                string.format('  ->  Q(%d)= %.2f %.2f %.2f %.2f',
                    s1, Q.f(s1,1), Q.f(s1,2), Q.f(s1,3), Q.f(s1,4))
            ))
    --]]
end

Q.f = function (s,a) return Q.value(s,a) end

Q.value = function (s,a)
    return Q.array[s][a]
end

Q.print = function()
    for s=1,nStates do
        local coord = map.coord (s)
        print (string.format('(%d,%d) %d %d %d %d',
            coord[1],coord[2],
            Q.value(s,1),Q.value(s,2),Q.value(s,3),Q.value(s,4)))
    end
end
 --
agent={}
agent.pos={-1,-1}
agent.init = function()
    agent.pos[2] = map.Start[2];
    agent.pos[1] = map.Start[1];
end
agent.isGoal = function()
    return agent.pos[1]==map.Goal[1] and agent.pos[2]==map.Goal[2]
end
agent.move = function (a)
    if a == 'left' then
        agent.pos[2] = agent.pos[2] - 1;
    elseif a == 'right' then
        agent.pos[2] = agent.pos[2] + 1;
    elseif a == 'up' then
        agent.pos[1] = agent.pos[1] - 1;
    elseif a == 'down' then
        agent.pos[1] = agent.pos[1] + 1;
    else
        print ('stupid action command:'); print (a)
    end
    -- wind effect
    agent.pos[1] = agent.pos[1] + wind.action (agent.pos[2])

    -- correction, confine the agent insde the map
    if agent.pos[1] < 1 then
        agent.pos[1] = 1
    elseif agent.pos[1] > map.nRows then
        agent.pos[1] = map.nRows
    end

    if agent.pos[2] < 1 then
        agent.pos[2] = 1
    elseif agent.pos[2] > map.nCols then
        agent.pos[2] = map.nCols
    end

    rwd = map.reward (agent.pos)
    return rwd, agent.pos
end

-- Sarsa: On-policy TD control algorithm (Figure 6.9)

function TDlearning()
    -- place the agent at the initial location
    agent.init()
    local str = string.format('! TDLearning starts at (%d,%d)',
                    agent.pos[1], agent.pos[2])
    local iter = 1
    local S = {agent.pos[1], agent.pos[2]}
    local A = Q.chooseAction (S)
    local locus = {{agent.pos[1], agent.pos[2]},A}
    local R, Snext, Anext
    repeat
        --print (string.format('Q action: %s', A))
        R, Snext = agent.move (A)

        Anext = Q.chooseAction (Snext)

        Q.update(S,A,R,Snext,Anext)

        --Q.print()
        S = {Snext[1], Snext[2]} -- state update, must use copy asignment!
        A = Anext

        locus[#locus+1] = {{agent.pos[1], agent.pos[2]},A}
        --print(locus[#locus])
        --[[
        print (string.format('[%d] agent now at (%d,%d) Goal(%d,%d)',
                iter, agent.pos[1], agent.pos[2], map.Goal[1], map.Goal[2]))
        --]]
        --if iter == 40 then break end
        iter = iter + 1
    until agent.isGoal()==true

    print(string.format('%s -- finished with time steps: %d', str, iter))
    return iter, locus
end

function run(ntimes)
    for i=1,ntimes do
        n,locus = TDlearning() -- move around and learn for one episode
        nTable[#nTable+1] = n
        --print (n)
    end
end

Q.init();
nTable = {}
run (100)
-- EOF --
