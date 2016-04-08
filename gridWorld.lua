-- yndk@sogang.ac.kr
-- Chapter 4. Grid world
-- Reinforcement Learning: An Introduction by Sutton & Barto
--
-- Not fully implimented but partially the result complies with the final
-- optimal policy table in the book.
-- So, at least I can check the performance of the Value-Iteration Algorithm
--
--[[
   Target: (1,1) (4,4)
   other states: any grid in 4x4 matrix
   state number in Map:
   1   2  3  4
   5   6  7  8
   9  10 11 12
   13 14 15 16
]]--
nGrids = 4;
nStates=nGrids^2
gamma = 1

reward = function (s,a,s2)
    if s2==1 or s2==16 then return 0 end
    return -1
end
-- Policy iteration for v*
-- 1. initialization
nActions=4 -- left, right, bottom, top
ActionSet = {1,2,3,4} --
--A = {left=1, right=2, top=3, bottom=4}
Astr = {'left', 'right', 'up', 'down'}
actionPolicy = {} -- random init
for i=1,nStates do actionPolicy[i] = math.random(1,4) end

function idx (i,j)
    return (i-1)*nGrids+j
end

function print2d(a)
    for i=1,nGrids do
        for j=1,nGrids do
            io.write (string.format("%d ", actionPolicy[idx(i,j)]))
        end
        io.write ('\n')
    end
end

value = {} -- real numbers
for i=1,nStates do value[i]=0 end

function state2coord(s)
    local i = math.floor((s-1)/nGrids)
    local j = s - i*nGrids -1
    return {i+1,j+1}
end
function action2coord(a)
    a2cTable = {{0,-1}, {0,1}, {-1,0}, {1,0} }
    return a2cTable[a]
end

function probability(s2,s,a)
    local c2 = state2coord(s2)
    local  c = state2coord(s)
    local  d = action2coord(a)
    local  t = {c[1]+d[1], c[2]+d[2]}
    for i=1,2 do -- boundary condition. stay the same
        if t[i]<1 or 4<t[i] then t[i]=c[i] end
    end
    if t[1]==c2[1] and t[2]==c2[2] then return 1 end

    return 0
end

valueSum = function(s)
    local sum = 0
    for s2=1,nStates do
        local action_choice = actionPolicy[s]
        sum = sum
            + probability(s2, s, action_choice) -- probabilty of s2 given s,a=pi(s)
            * (reward(s,action_choice, s2)  --
                + gamma*value[s2])
    end
    return sum;
end

function maxValue(s)
    local maxv = -1E10
    for action_choice=1,#ActionSet do
        local sum = 0
        for s2=1,nStates do
            sum = sum +
                probability(s2, s, action_choice) -- probabilty of s2 given s,a=pi(s)
                    * (reward(s, action_choice, s2)  --
                        + gamma*value[s2])
        end
        if maxv < sum then maxv = sum end
    end
    return maxv;
end

ii=0
repeat
-- 2. Policy Evaluation
do
    local iteration=1
    repeat
        local delta = 0
        for s=1,nStates do
            local temp = value[s]
            --value[s] = valueSum(s)
            value[s] = maxValue(s)
            --print ('value ' .. s .. ' = ' .. value[s])
            if delta < math.abs(value[s]-temp) then delta=math.abs(value[s]-temp) end
        end
        print2d (value)
        print ('finished ' .. iteration .. ' -- ' .. 'delta= ' .. delta)
        iteration = iteration + 1
    until delta<1E-3
end

-- Output a deterministic policy, policy, such that
-- p(s) = argmax_a sum_s2 prob * (r + g*v(s2))
function argmaxPolicy(s)
    local sums={}
    for a=1,#ActionSet do
        sums[a]=0
        for s2=1,nStates do
            sums[a] = sums[a] + probability(s2,s,a)*(reward(s,a,s2) + gamma*value[s2])
        end
    end
    -- returns the id corresp. to the maximum of input table
    local maxid = 1
    for i=1,#sums do
        if sums[maxid] < sums[i] then maxid = i end
    end
    --print (sums)
    --print ('maxid = ' .. maxid)
    return maxid
end
for s=1,nStates do
    actionPolicy[s] = argmaxPolicy (s)
end
print('updated actionPolicy')
print2d(actionPolicy)

print ('ii done ' .. ii .. ' ---------------- ')
ii = ii + 1

until ii == 10
