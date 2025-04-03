import os
import argparse
import pdb
import numpy as np
import torch # For the model and 
import csv # For saving the results
from collections import deque, defaultdict # For the cs-pibt and lacam
import cProfile # For profiling
import pstats # For profiling
from tqdm import tqdm # For progress bar
import time

from main_pys.model import GNNStack, CustomConv # Required for using the model even if not explictly called
from main_pys.model_inputs import create_data_object, normalize_graph_data
from main_pys.custom_timer import CustomTimer

def str2bool(v: str) -> bool:
    """Converts a string to a boolean value. Used for argparse."""
    return v.lower() in ("yes", "true", "t", "1")

####################################################
#### Helper functions
def parse_scene(scen_file):
    """Input: scenfile
    Output: start_locations, goal_locations
    """
    start_locations = []
    goal_locations = []

    with open(scen_file) as f:
        line = f.readline().strip()
        # if line[0] == 'v':  # Nathan's benchmark
        start_locations = list()
        goal_locations = list()
        # sep = "\t"
        for line in f:
            line = line.rstrip() #remove the new line
            # line = line.replace("\t", " ") # Most instances have tabs, but some have spaces
            # tokens = line.split(" ")
            tokens = line.split("\t") # Everything is tab separated
            assert(len(tokens) == 9) 
            # num_of_cols = int(tokens[2])
            # num_of_rows = int(tokens[3])
            tokens = tokens[4:]  # Skip the first four elements
            col = int(tokens[0])
            row = int(tokens[1])
            start_locations.append((row,col)) # This is consistent with usage
            col = int(tokens[2])
            row = int(tokens[3])
            goal_locations.append((row,col)) # This is consistant with usage
    return np.array(start_locations, dtype=int), np.array(goal_locations, dtype=int)

def createScenFile(locs, goal_locs, map_name, scenFilepath):
    """Input: 
        locs: (N,2)
        goal_locs: (N,2)
        map_name: name of the map
        scenFilepath: filepath to save scen
    """
    assert(locs.min() >= 0 and goal_locs.min() >= 0)

    ### Write scen file with the locs and goal_locs
    # Note we need to swap row:[0],col:[1] and save it as col,row
    with open(scenFilepath, 'w') as f:
        f.write(f"version {len(locs)} \n")
        for i in range(locs.shape[0]):
            f.write(f"0\t{map_name}\t{0}\t{0}\t{locs[i,1]}\t{locs[i,0]}\t{goal_locs[i,1]}\t{goal_locs[i,0]}\t0 \n")
    # print("Scen file created at: {}".format(scenFilepath))

def getCosts(solution_path, goal_locs):
    """
    Inputs:
        solution_path: (T,N,2)
        goal_locs: (N,2)
    Outputs:
        total_cost_true: sum of true costs (intermediate waiting at goal incurs costs)
        total_cost_not_resting_at_goal: sum of costs if not resting at goal
    """
    print(solution_path.shape)
    at_goal = np.all(np.equal(solution_path, np.expand_dims(goal_locs, 0)), axis=2) # (T,N,2), (1,N,2) -> (T,N)

    # Find the last timestep each agent is not at the goal
    not_at_goal = 1 - at_goal # (T,N)
    last_timestep_at_goal = len(at_goal) - np.argmax(not_at_goal[::-1], axis=0) # (N), argmax returns first occurence, so reverse
    last_timestep_at_goal = np.minimum(last_timestep_at_goal, len(at_goal)-1) # Agents that never reach goal will be fixed here
    total_cost_true = last_timestep_at_goal.sum()

    resting_at_goal = np.logical_and(at_goal[:-1], at_goal[1:]) # (T-1,N)
    total_cost_not_resting_at_goal = (1-resting_at_goal).sum() # int

    num_agents_at_goal = np.sum(at_goal[-1]) # int
    assert(total_cost_true >= total_cost_not_resting_at_goal)
    assert(total_cost_true <= (solution_path.shape[0]-1)*solution_path.shape[1])
    return total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal

def testGetCosts():
    goal_locs = np.array([[1,10], [2,20], [3,30]]) # (N=3,2)
    solution_path = np.array([
        [[1,9],  [1,10], [1,10], [1,9],  [1,10]], # TC=4, TNRAG=3
        [[0,20], [1,20], [2,20], [2,20], [2,20]], # TC=2, TNRAG=2
        [[5,4],  [5,4],  [5,4],  [5,4],  [5,4]]   # TC=4, TNRAG=4
    ]) # (N=3,T=5,2)
    solution_path = np.swapaxes(solution_path, 0, 1) # (T,N,2)

    total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal = getCosts(solution_path, goal_locs)
    assert(total_cost_true == 4+2+4)
    assert(total_cost_not_resting_at_goal == 3+2+4)
    assert(num_agents_at_goal == 2)
    print("getCosts test passed!")

def convertProbsToPreferences(probs, conversion_type):
    """Converts probabilities to preferences
    Inputs:
        probs: (N,5) probabilities
        conversion_type: sorted or sampled
    Outputs:
        preferences: (N,5) preferences with each row containing 0,1,2,3,4
    """
    if conversion_type == "sorted":
        preferences = np.argsort(-probs, axis=1)
    elif conversion_type == "sampled":
        ### This naive version with for loop becomes a bottleneck
        # preferences = np.zeros_like(probs, dtype=int)
        # for i in range(probs.shape[0]):
        #     preferences[i] = np.random.choice(5, size=5, replace=False, p=probs[i])

        ### Faster version using torch
        probs = torch.tensor(probs, dtype=torch.float32)
        preferences = torch.zeros_like(probs, dtype=torch.int64)
        for i in range(5):
            cur_sample = torch.multinomial(probs, num_samples=1, replacement=False) # (N,1)
            probs.scatter_(1, cur_sample, 0) # Set the sampled index to 0
            preferences[:,i] = cur_sample[:,0]
        preferences = preferences.numpy()
        assert(np.all(preferences.sum(axis=1) == 10)) # Make sure 0,1,2,3,4 are all present per row
    else:
        raise ValueError('Invalid conversion type: {}'.format(conversion_type))
    return preferences

####################################################
#### LaCAM and PIBT func

LABEL_TO_MOVES = np.array([[0,0], [0,1], [1,0], [-1,0], [0,-1]]) #  Stop, Right, Down, Up, Left
 # This needs to match Pipeline's action ordering

def pibtRecursive(grid_map, agent_id, action_preferences, planned_agents, move_matrix, 
         occupied_nodes, occupied_edges, current_locs, current_locs_to_agent,
         constrained_agents_to_action, start_time,timeLimit):
    """Inputs:
        grid_map: (H,W)
        agent_id: int
        action_preferences: (N,5)
        planned_agents: list of agent_ids
        move_matrix: (N,2)
        occupied_nodes: set (row, col)
        occupied_edges: set (row_from, col_from, row_to, col_to)
        current_locs: (N,2)
        current_locs_to_agent: (H,W)  # dict: (row, col) -> agent_id
        constrained_agents_to_action: dict: agent_id -> action_index
    """
    moves_ordered = LABEL_TO_MOVES[action_preferences[agent_id]]
    if agent_id in constrained_agents_to_action: # Force agent to only pick that action if constrained
        action_index = constrained_agents_to_action[agent_id]
        moves_ordered = moves_ordered[action_index:action_index+1] # Only consider that action
    
    cur_time=time.time()
    if cur_time-start_time>timeLimit:
        return False

    current_pos = current_locs[agent_id] # (2)
    for aMove in moves_ordered:
        next_loc = current_pos + aMove # (2)
        
        # Skip if would leave map bounds
        if next_loc[0] < 0 or next_loc[0] >= grid_map.shape[0] or next_loc[1] < 0 or next_loc[1] >= grid_map.shape[1]:
            continue
        # Skip if obstacle
        if grid_map[next_loc[0], next_loc[1]] == 1:
            continue
        # Skip if vertex occupied by higher agent
        if occupied_nodes[next_loc[0], next_loc[1]]:
            continue
        # Skip if reverse edge occupied by higher agent
        rev_edge_key = tuple([*next_loc, *current_pos])
        if occupied_edges[rev_edge_key]:
            continue
        
        ### Pretend we move there
        move_matrix[agent_id] = aMove
        planned_agents[agent_id] = True
        occupied_nodes[next_loc[0], next_loc[1]] = True
        # occupied_edges.add(edge_key)
        edge_key = tuple([*current_pos, *next_loc])
        occupied_edges[edge_key] = True

        conflicting_agent = current_locs_to_agent[next_loc[0], next_loc[1]]
        if conflicting_agent != -1 and conflicting_agent != agent_id and not planned_agents[conflicting_agent]:
            # Recurse
            isvalid = pibtRecursive(grid_map, conflicting_agent, action_preferences, planned_agents,
                                move_matrix, occupied_nodes, occupied_edges, current_locs,
                                current_locs_to_agent, constrained_agents_to_action, start_time,timeLimit)
            if isvalid:
                return True
            else:
                planned_agents[agent_id] = False
                occupied_nodes[next_loc[0], next_loc[1]] = False
                # occupied_edges.remove(edge_key)
                occupied_edges[edge_key] = False
                continue
        else:
            # No conflict
            return True
        
    # No valid move found
    return False

def pibt(grid_map, action_preferences, current_locs, agent_priorities, agent_constraints, start_time, timeLimit):
    """Inputs:
        grid_map: (H,W)
        action_preferences: (N,5)
        current_locs: (N,2)
        agent_priorities: (N)
        agent_constraints: [(agent_id, action index), ...], empty list if no constraints
    Outputs:
        move_matrix: (N,2)
        pibt_worked: bool
    """
    agent_order = np.argsort(-agent_priorities) # Sort by priority, highest first
    move_matrix = np.zeros((len(agent_priorities), 2), dtype=int) # (N,2)
    occupied_nodes = np.zeros(grid_map.shape, dtype=bool) # (H,W), True denotes occupied
    # occupied_edges = set() # (row_from, col_from, row_to, col_to)
    occupied_edges = defaultdict(bool) # (row_from, col_from, row_to, col_to) -> bool, faster than set
    # pdb.set_trace()
    planned_agents = np.zeros(len(agent_priorities), dtype=bool) # (N), True denotes planned

    current_locs_to_agent = np.zeros(grid_map.shape, dtype=int) - 1  # (H,W), -1 denotes no agent
    current_locs_to_agent[current_locs[:,0], current_locs[:,1]] = np.arange(len(current_locs))  # Assigns agent_id to each location

    ### Convert agent_id constraints to actual agent indices based on agent_order
    # This means that constraints apply to highest priority agents first, which is desired as these agents are furthest from their goals
    constrained_agents_to_action = dict()
    for agent_id, action_index in agent_constraints:
        which_agent = agent_order[agent_id]
        # pdb.set_trace()
        constrained_agents_to_action[which_agent] = action_preferences[which_agent, (action_index+1)%5]

    ### Plan agents in order of priority
    for agent_id in agent_order:
        if planned_agents[agent_id]:
            continue
        pibt_worked = pibtRecursive(grid_map, agent_id, action_preferences, planned_agents, 
                            move_matrix, occupied_nodes, occupied_edges, 
                            current_locs, current_locs_to_agent, constrained_agents_to_action, start_time,timeLimit)
        if pibt_worked is False:
            break
    # if pibt_worked and len(constrained_agents_to_action):
    #     # print(constrained_agents_to_action)
    #     for agent_id, action_index in constrained_agents_to_action.items():
    #         totalStr = f"{agent_id} at {current_locs[agent_id]} -> {action_index}; "
    #     print(totalStr)
    return move_matrix, pibt_worked

def updatePriorities(prev_priorities, at_goal):
    """Inputs:
        prev_priorities: (N)
        at_goal: (N), boolean
    Outputs:
        agent_priorities: (N)
    """
    agent_priorities = prev_priorities.copy()

    # agents previously at goal (parent_priority <= 0) and still at goal (dist == 0) should have priority decreased
    agent_priorities[(prev_priorities <= 0) & at_goal] -= 1
    # agents that just got to goal (parent_priority > 0 & dist == 0) should have priority set to 0
    agent_priorities[(prev_priorities > 0) & at_goal] = 0 # Agents that are at goal should have priority decreased
    # agents that are not at goal (dist > 0) should have priority increased
    agent_priorities[~at_goal] = np.maximum(prev_priorities[~at_goal], 0) + 1
    # agent_priorities[at_goal] = 0 # Set priority to 0 if reached goal
    return agent_priorities

class LaCAMRunner:
    class HLNode:
        def __init__(self, state: np.ndarray, action_preferences: np.ndarray, 
                     parent: 'LaCAMRunner.HLNode', bd: np.ndarray, goal_locations: np.ndarray) -> None:
            self.state = state
            self.action_preferences = action_preferences
            self.parent = parent
            self.queue_of_constraints = deque() # Each element is a list of tuples (agentId, actionIndex)
            # A successor could require constraining multiple agents, therefore each constraint is a list of tuples (one per constrained agent)
            # Conceptually, we are lazily BFSing the constraints. When generating [(0,3),(1,4)] 
            # we want to add [(0,3),(1,4),(2,0)], [(0,3),(1,4),(2,1)], ... [(0,3),(1,4),(2,4)]
            self.queue_of_constraints.append([]) # Start with no constraints

            ### Compute agent priorities via PIBT rule
            if parent is None:
                self.depth = 0
                distance_to_goal = bd[range(len(state)), state[:,0], state[:,1]] # (N)
                self.agent_priorities = distance_to_goal / distance_to_goal.max() # Normalize to [0,1]
            else:
                self.depth = parent.depth + 1
                at_goal = np.all(np.equal(state, goal_locations), axis=1) # (N)
                self.agent_priorities = updatePriorities(self.parent.agent_priorities, at_goal)

        def getNextState(self, grid_map: np.ndarray, start_time, timeLimit):
            """Outputs:
                new_state: None or (N,2)
            """
            assert(len(self.queue_of_constraints) > 0)
            curConstraint = self.queue_of_constraints.popleft()
            # curConstraint is a list of tuples (agentId, actionIndex), agentId of K correponds to agent with K highest priority
            
            ### Add in next constraints
            if len(curConstraint) == 0: # Initial, start with constraining agent 0
                for i in range(0,5):
                    self.queue_of_constraints.append([(0,i)])
            else:
                # curConstraint is a list of tuples (agentId, actionIndex) dictating that agentId should take actionIndex
                curAgent = curConstraint[-1][0] # curConstraint[-1] is the last constraint added
                if curAgent + 1 < len(self.state): # Constrain the next agent
                    for i in range(0,5): # We want to constrain the next agent with the same current constraints + the new constraint
                        self.queue_of_constraints.append(curConstraint + [(curAgent+1,i)])

            # Run PIBT
            new_move, pibt_worked = pibt(grid_map, self.action_preferences, self.state, self.agent_priorities, curConstraint, start_time, timeLimit)

            if not pibt_worked: # Failed with the constraints
                return None
            new_state = self.state + new_move
            return new_state
            
    def __init__(self, real_time=False) -> None:
        self.mainStack = deque() # Stack of HLNodes
        self.stateToHLNodes = dict() # Maps str(state) to HLNode
        self.real_time = real_time
        if self.real_time:
            print("Real-Time LaCAM enabled")
        else:
            print("LaCAM enabled")
    
    def lacam(self, start_locations, goal_locations, bd, grid_map, getActionPrefsFromLocs, lacamLimit, start_time, timeLimit):
        """Inputs:
            bd: (N,H,W)
            grid_map: (H,W)
            getActionPrefsFromLocs: function that takes in (N,2) and outputs (N,5) action preferences
                This would call the NN model, or could use bds (to replicate original LaCAM)
        """

        # mainStack = deque() # Stack of HLNodes
        # stateToHLNodes = dict() # Maps str(state) to HLNode
        if not self.real_time:
            self.mainStack.clear() # Clear the stack if not real-time
            self.stateToHLNodes = dict() # Clear the cache if not real-time

        if len(self.mainStack) == 0: # First time running
            ### Initialize the HLNode
            curNode = LaCAMRunner.HLNode(start_locations, getActionPrefsFromLocs(start_locations), None, bd, goal_locations)
            self.mainStack.appendleft(curNode) # Start with the initial state
            self.stateToHLNodes[start_locations.tobytes()] = curNode

        success = False
        MAXGENERATED = lacamLimit
        numNodesExpanded = 0
        numGenerated = 1 # Start with 1 as we have already added the initial state
        while len(self.mainStack) > 0:
            curNode : LaCAMRunner.HLNode = self.mainStack.popleft()
            if len(curNode.queue_of_constraints) != 0: # Always add in the original HLNode if not exhausted
                self.mainStack.appendleft(curNode)
            new_locs = curNode.getNextState(grid_map, start_time, timeLimit)
            if time.time() - start_time > timeLimit: # Time limit hit
                break
            if new_locs is None:
                continue
            numNodesExpanded += 1

            # Check if all agents have reached their goals
            if np.all(np.equal(new_locs, goal_locations)):
                print("Stopping as found goal in LaCAM, depth: {}, nodes expanded: {}".format(curNode.depth, numNodesExpanded))
                success = True
                break

            key = new_locs.tobytes()
            if key in self.stateToHLNodes.keys(): # Already visited/created this state
                curNode = self.stateToHLNodes[key] # Get the existing HLNode
                self.mainStack.appendleft(curNode) # Add to stack
            else:
                # Create a new HLNode
                # probs = runNNOnState(new_locs, bd, grid_map, k, m, model, device)
                # action_preferences = convertProbsToPreferences(probs, conversion_type) # (N,5)
                newHLNode = LaCAMRunner.HLNode(new_locs, getActionPrefsFromLocs(new_locs), curNode, bd, goal_locations)
                numGenerated += 1

                # Add to collections
                self.stateToHLNodes[key] = newHLNode
                self.mainStack.appendleft(newHLNode)
                
            if numGenerated >= MAXGENERATED: # Limit the number of nodes generated
                break
            
        if self.real_time:
            entirePath = [start_locations, new_locs]
        else:    
            # Get path via backtracking. If not a success, this returns the path to the last state found
            entirePath = [new_locs]
            while curNode is not None:
                entirePath.append(curNode.state)
                curNode = curNode.parent
            entirePath.reverse() # Reverse to get path from start to goal
        return entirePath, success, numNodesExpanded, numGenerated

class WrapperNNWithCache:
    def __init__(self, bd, grid_map, model, device, k, m, goal_locations, timer) -> None:
        self.bd = bd
        self.grid_map = grid_map
        self.model = model
        self.device = device
        self.k = k
        self.m = m
        self.saved_calls = dict()
        self.hits = 0
        self.goal_locations = goal_locations
        self.timer = timer

    def __call__(self, locs: np.ndarray):
        key = locs.tobytes()
        if key in self.saved_calls.keys():
            self.hits += 1
            return self.saved_calls[key]
        else:
            probs = runNNOnState(locs, self.bd, self.grid_map, self.k, self.m, self.model, self.device, self.goal_locations, self.timer)
            self.saved_calls[key] = probs
            return probs

def runNNOnState(cur_locs, bd, grid_map, k, m, model, device, goal_locations, timer: CustomTimer):
    """Inputs:
        cur_locs: (N,2)
    Outputs:
        probs: (N,5)
    """

    with torch.no_grad():
        # Create the data object
        timer.start("create_nn_data")
        data = create_data_object(cur_locs, bd, grid_map, k, m, goal_locations)
        data = normalize_graph_data(data, k)
        data = data.to(device)
        timer.stop("create_nn_data")

        # Forward pass
        timer.start("forward_pass")
        _, predictions = model(data)
        # print(predictions.shape, torch.softmax(predictions, dim=1)[0])
        # predictions = torch.zeros_like(predictions)# TODO REMOVE THIS
        # predictions[:,0] = 1
        # print(predictions.shape, torch.softmax(predictions, dim=1)[0])
        probabilities = torch.softmax(predictions, dim=1) # More general version
        timer.stop("forward_pass")

        # Get the action preferences
        probs = probabilities.cpu().detach().numpy() # (N,5)
    return probs

class WrapperBDGetActionPrefs:
    def __init__(self, bd, grid_map, k, m, num_agents) -> None:
        self.bd = bd
        self.grid_map = grid_map
        self.k = k
        self.m = m
        self.range_num_agents = np.arange(num_agents)

    def __call__(self, locs):
        return get_bd_prefs(locs, self.bd, self.range_num_agents)

def simulate(device, model, k, m, grid_map, bd, start_locations, goal_locations, 
             max_steps, shield_type, lacam_lookahead, args, timer: CustomTimer):
    """Inputs:
        grid_map: (H,W), note includes padding
        bd: (N,H,W), note includes padding
        start_locations: (N,2)
        goal_locations: (N,2)
    """
    if shield_type not in ["CS-PIBT", "CS-Freeze", "LaCAM", "Real-Time-LaCAM"]:
        raise KeyError('Invalid shield type: {}'.format(shield_type))
    
    wrapper_nn = WrapperNNWithCache(bd, grid_map, model, device, k, m, goal_locations, timer)
    wrapper_bd_prefs = WrapperBDGetActionPrefs(bd, grid_map, k, m, len(start_locations)) # This returns PIBT action preferences
    def getActionPrefsFromLocs(locs):
        # probs = wrapper_nn(locs) # Using wrapper_nn is not effective with "sampled" as we almost never revisit states
        probs = runNNOnState(locs, bd, grid_map, k, m, model, device, goal_locations, timer)

        ### Mask out invalid actions
        action_mask = grid_map[cur_locs[:, 0, None] + LABEL_TO_MOVES[:, 0], cur_locs[:, 1, None] + LABEL_TO_MOVES[:, 1]] == 1  # (N,5)
        assert(not np.any(action_mask[:,0])) # First action should always be invalid
        probs[action_mask] = 1e-8  # Mask out probabilities for invalid actions, cannot set to 0 as it messes up sampling later
        probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize to sum to 1
        return convertProbsToPreferences(probs, "sampled")
    # getActionPrefsFromLocs = wrapper_bd_prefs
    
    cur_locs = start_locations # (N,2)
    # Ensure no start/goal locations are on obstacles
    assert(grid_map[start_locations[:,0], start_locations[:,1]].sum() == 0)
    assert(grid_map[goal_locations[:,0], goal_locations[:,1]].sum() == 0)

    agent_priorities = bd[range(len(start_locations)), start_locations[:,0], start_locations[:,1]] # (N)
    agent_priorities = agent_priorities / agent_priorities.max() # Normalize to [0,1]
    
    if shield_type in ["LaCAM", "Real-Time-LaCAM"]:
        lacamRunner = LaCAMRunner(real_time=(shield_type=="Real-Time-LaCAM"))

    solution_path = [cur_locs.copy()]
    success = False
    MAX_USE_LACAM = 100  # Hardcoded limit on how many times to use LaCAM since it is slow
    start_time = time.time()
    for step in tqdm(range(max_steps)):
        # Update priorities
        agents_at_goal = np.all(np.equal(cur_locs, goal_locations), axis=1) # (N)
        agent_priorities = updatePriorities(agent_priorities, agents_at_goal)
        if time.time()-start_time > args.timeLimit and args.timeLimit > 0:
            print("time limit hit")
            break
        
        
        if shield_type in ["CS-PIBT", "CS-Freeze"]:
            action_preferences = getActionPrefsFromLocs(cur_locs) # (N,5)
            if shield_type == "CS-Freeze":
                action_preferences = action_preferences[:,:2] # (N,2)
                action_preferences[:,1] = 0  # 0 action index corresponds to stop
            timer.start("cs-time")
            new_move, cspibt_worked = pibt(grid_map, action_preferences, cur_locs, agent_priorities, [], start_time, args.timeLimit)
            timer.stop("cs-time")
            if not cspibt_worked:
                if (time.time() - start_time < args.timeLimit):
                    print("ERROR: CS-PIBT failed even though it did not time out! This should never happen!")
                break
            # if not cspibt_worked:
            #     raise RuntimeError('CS-PIBT failed; should never fail when not using LaCAM constraints!')
        else:
            # Run LaCAM
            # at_goal_ratio = np.mean(agents_at_goal)
            # scaled_lookahead = 1 # Use lookahead of 1 in the beginning
            # if at_goal_ratio > 0.90 and MAX_USE_LACAM >= 0: # Only use larger lookahead when at least 90% agents are at goal and have budget
            #     scaled_lookahead = int(np.ceil(lacam_lookahead * at_goal_ratio)) # Scale lookahead based on how many agents are at goal
            #     MAX_USE_LACAM -= 1
            scaled_lookahead = lacam_lookahead
            next_locs, lacamFoundSolution, numNodesExpanded, numGenerated = lacamRunner.lacam(cur_locs, goal_locations, 
                                                    bd, grid_map, getActionPrefsFromLocs, scaled_lookahead, start_time, args.timeLimit)
            # Note: next_locs is (T1,N,2) where T1 is the lookahead depth

            if lacamFoundSolution:
                for t in range(1, len(next_locs)):
                    assert(np.all(grid_map[next_locs[t][:,0], next_locs[t][:,1]] == 0)) # Ensure no agents are on obstacles
                print("LaCAM found solution at step: {}".format(step))
                solution_path.extend(next_locs[1:]) # Add the lookahead path
                success = True
                break
            else:
                if time.time()-start_time > args.timeLimit and args.timeLimit > 0:
                    print("time limit hit")
                    break
                new_move = next_locs[1] - cur_locs # (N,2)

        cur_locs = cur_locs + new_move # (N,2)
        solution_path.append(cur_locs.copy())
        assert(np.all(grid_map[cur_locs[:,0], cur_locs[:,1]] == 0)) # Ensure no agents are on obstacles

        # Check if all agents have reached their goals
        if np.all(np.equal(cur_locs, goal_locations)):
            success = True
            break
    
    solution_path = np.array(solution_path) # (T<=max_steps+1,N,2)
    total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal = getCosts(solution_path, goal_locations)
    print("Total cache hits: {}, Total size: {}".format(wrapper_nn.hits, len(wrapper_nn.saved_calls)))

    return solution_path, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, success


def main(args: argparse.ArgumentParser):
    # Setting constants
    torch.set_num_threads(1) # Make pytorch use only 1 thread, otherwise by default will try using all threads    
    k = 4 # Model has a local window radius of 4
    m = 5 # 5 closest neighbors
    # Load the map
    if not os.path.exists(args.mapNpzFile):
        raise FileNotFoundError('Map file: {} not found.'.format(args.mapNpzFile))
    map_npz = np.load(args.mapNpzFile) # Keys are {MAPNAME}.map -> (H,W)
    if args.mapName+".map" not in map_npz:
        raise ValueError('Map name not found in the map file.')
    map_grid = map_npz[args.mapName+".map"] # (H,W)
    map_grid = np.pad(map_grid, k, 'constant', constant_values=1) # Add padding

    # Load the scen
    if not os.path.exists(args.scenFile):
        raise FileNotFoundError('Scen file: {} not found.'.format(args.scenFile))
    start_locations, goal_locations = parse_scene(args.scenFile) # Each (max agents,2)
    num_agents = args.agentNum # This is N
    if start_locations.shape[0] < num_agents:
        raise ValueError('Not enough agents in the scen file.')
    start_locations = start_locations[:num_agents] + k # (N,2)
    goal_locations = goal_locations[:num_agents] + k # (N,2)

    # Load the bd
    # pdb.set_trace()
    if not os.path.exists(args.bdNpzFile):
        raise FileNotFoundError('BD file: {} not found.'.format(args.bdNpzFile))
    scen_num = args.scenFile.split('-')[-1].split('.')[0]
    bd_key = f"{args.mapName}-random-{scen_num}"
    bd_npz = np.load(args.bdNpzFile)
    if bd_key not in bd_npz:
        raise ValueError('BD key {} not found in the bd file'.format(bd_key))
    bd = bd_npz[bd_key][:num_agents] # (max agents,H,W)->(N,H,W)
    bd = np.pad(bd, ((0,0),(k,k),(k,k)), 'constant', constant_values=12345678) # Add padding

    # Load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.useGPU else "cpu") # Use GPU if available
    if not os.path.exists(args.modelPath):
        raise FileNotFoundError('Model file: {} not found.'.format(args.modelPath))
    model = torch.load(args.modelPath, map_location=device)
    model.eval()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get max steps
    if args.maxSteps.endswith('x'):
        longest_single_path = bd[range(num_agents), start_locations[:,0], start_locations[:,1]].max()
        max_steps = int(args.maxSteps[:-1]) * longest_single_path
    else:
        max_steps = int(args.maxSteps)
        
    # Simulate
    if args.debug:
        profiler = cProfile.Profile()
        profiler.enable()
        
    timer = CustomTimer()
    timer.start("total_simulate")
    solution_path, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, success = simulate(device,
            model, k, m, map_grid, bd, start_locations, goal_locations, 
            max_steps, args.shieldType, args.lacamLookahead, args, timer)
    timer.stop("total_simulate")
    total_simulate_time = timer.getTimes("total_simulate")
    print("Success: {}, Total cost true: {}, Total cost not at goal: {}, Num agents at goal: {}/{}, Seconds spent: {}".format(success, 
                                    total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, num_agents, total_simulate_time))
    solution_path = solution_path - k # (T,N,2) Removes padding
    goal_locations = goal_locations - k # (N,2) Removes padding
    if args.debug:
        profiler.disable()
        profiler.dump_stats('profile.prof')
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(30) # Print the top 30
    
    # Save the statistics into the csv file
    if not os.path.exists(args.outputCSVFile):
        # Create the file and write the header
        with open(args.outputCSVFile, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['mapName', 'scenFile', 'agentNum', 'seed', 'shieldType', 'lacamLookahead',
                             'modelPath', 'useGPU', 'maxSteps', 
                             'success', 'total_cost_true', 'total_cost_not_resting_at_goal',
                             'num_agents_at_goal', 'runtime', 'create_nn_data', 'forward_pass', 'cs-time'])
            
    with open(args.outputCSVFile, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([args.mapName, args.scenFile, args.agentNum, args.seed, args.shieldType, args.lacamLookahead,
                         args.modelPath, args.useGPU, args.maxSteps,
                         success, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, total_simulate_time,
                         timer.getTimes("create_nn_data"), timer.getTimes("forward_pass"), timer.getTimes("cs-time")])

    # Save the paths
    if args.outputPathsFile is not None:
        assert(args.outputPathsFile.endswith('.npy'))
        np.save(args.outputPathsFile, solution_path)


### Example command
"""   
python -m main_pys.simulator --mapNpzFile=data/constant_npzs/all_maps.npz \
      --mapName=den312d --scenFile=data/mapf-scen-random/den312d-random-1.scen \
      --bdNpzFile=data/constant_npzs/bd_npzs/den312d_bds.npz \
      --modelPath=data/model/max_test_acc.pt \
      --outputCSVFile=logs/results.csv \
      --outputPathsFile=logs/paths.npy \
      --maxSteps=3x --seed=0 --useGPU=True \
      --agentNum=50 --shieldType=Real-Time-LaCAM --lacamLookahead=1
"""

if __name__ == '__main__':
    # testGetCosts()
    parser = argparse.ArgumentParser()
    # Map / scen parameters
    parser.add_argument('--mapNpzFile', type=str, required=True)
    parser.add_argument('--mapName', type=str, help="Without .map", required=True)
    parser.add_argument('--scenFile', type=str, required=True)
    parser.add_argument('--agentNum', type=int, required=True)
    parser.add_argument('--bdNpzFile', type=str, required=True)
    parser.add_argument('--debug', type=lambda x: bool(str2bool(x)), help="Whether to enable debugging stats", default=False)
    # Simulator parameters
    parser.add_argument('--modelPath', type=str, required=True)
    parser.add_argument('--useGPU', type=lambda x: bool(str2bool(x)), required=True)
    parser.add_argument('--maxSteps', type=str, help="int or [int]x, e.g. 100 or 2x to denote multiplicative factor", required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shieldType', type=str, default='CS-PIBT', choices=['CS-PIBT', 'CS-Freeze', 'LaCAM', 'Real-Time-LaCAM'])
    parser.add_argument('--lacamLookahead', type=int, help="LaCAM node expansion limit", default=0)
    parser.add_argument('--timeLimit', type=int, help="Time limit (s)", default=60)
    # Output parameters
    parser.add_argument('--outputCSVFile', type=str, help="where to output statistics", required=True)
    parser.add_argument('--outputPathsFile', type=str, help="where to output path, ends with .npy", default=None)

    args = parser.parse_args()

    if args.mapName.endswith('.map'): # Remove ending .map
        args.mapName = args.mapName.removesuffix('.map')
    if args.shieldType == "LaCAM" and args.lacamLookahead == 0:
        raise ValueError('LaCAM lookahead must be set when using LaCAM shield type.')
    if args.shieldType == "Real-Time-LaCAM":
        if args.lacamLookahead != 1 or args.lacamLookahead != 0:
            print("Warning: Real-Time-LaCAM only works with lookahad set to 1, ignoring input {} and setting to 1".format(args.lacamLookahead))
        args.lacamLookahead = 1
        
    
    # pdb.set_trace()

    main(args)