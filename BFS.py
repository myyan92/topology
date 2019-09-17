# Run this code only using python3
from topology.representation import AbstractState
import random, copy
import pdb

def generate_next(state):
    # cross action
    for over_idx in range(0, state.pts+1):
        for under_idx in range(0, state.pts+1):
            for sign in [-1, 1]:
                new_state = copy.deepcopy(state)
                success = new_state.cross(over_idx, under_idx, sign)
                if success:
                    action = {'move':'cross', 'over_idx':over_idx, 'under_idx':under_idx, 'sign':sign}
                    yield new_state, action

    # R1 action
    for idx in range(0, state.pts+1):
        for sign in [1, -1]:
            for left in [1, -1]:
                new_state = copy.deepcopy(state)
                success = new_state.Reide1(idx, sign, left)
                if success:
                    action = {'move':'R1', 'idx':idx, 'sign':sign, 'left':left}
                    yield new_state, action

    # R2 action
    for over_idx in range(0, state.pts+1):
        for under_idx in range(0, state.pts+1):
            for left in [-1, 1]:
                for obu in [-1, 1]:
                    new_state = copy.deepcopy(state)
                    success = new_state.Reide2(over_idx, under_idx, left=left, over_before_under=obu)
                    if success:
                        action = {'move':'R2', 'over_idx':over_idx, 'under_idx':under_idx, 'left':left, 'over_before_under':obu}
                        yield new_state, action


def bfs(start, goal, max_depth=None):
    # using max_depth to limit search. mainly used for data generation.
    visited, parents, actions = [start], [0], [{}]
    depth_index = [0]
    if start == goal:
        return [start], []
    head = 0
    while head < len(visited):
        state = visited[head]
        if head>depth_index[-1]:
            depth_index.append(len(visited)-1)
        if max_depth is not None and len(depth_index) > max_depth:
            return [],[]
        for new_state, action in generate_next(state):
            append = True
            for visited_state in visited:
                if new_state == visited_state:
                    append = False
            if append:
                visited.append(new_state)
                parents.append(head)
                actions.append(action)
            if new_state == goal:
                break
        head += 1
        if visited[-1] == goal:
            head = len(visited)-1
            break
    # backtrack to find solution
    if head < len(visited):
        path = [visited[head]]
        path_action = [actions[head]]
        while parents[head]>0:
            head = parents[head]
            path.insert(0, visited[head])
            path_action.insert(0, actions[head])
        path.insert(0, visited[0])
    return path, path_action


def bfs_all_path(start, goal, max_depth):
    # find all paths connecting start to goal.
    # limit search using max_depth.
    visited, parents, actions = [start], [0], [{}]
    depth_index = [0]
    if start == goal:
        return [start], []
    head = 0
    goal_index = []
    while head < len(visited):
        state = visited[head]
        if head>depth_index[-1]:
            depth_index.append(len(visited)-1)
        if max_depth is not None and len(depth_index) > max_depth:
            break
        for new_state, action in generate_next(state):
            visited.append(new_state)
            parents.append(head)
            actions.append(action)
            if new_state == goal:
                goal_index.append(len(visited)-1)
        head += 1

    # backtrack to find solutions
    paths = []
    for idx in goal_index:
        current = idx
        path = [visited[current]]
        path_action = [actions[current]]
        while parents[current]>0:
            current = parents[current]
            path.insert(0, visited[current])
            path_action.insert(0, actions[current])
        path.insert(0, visited[0])
        paths.append((path, path_action))
    return paths


if __name__=="__main__":
    start = AbstractState()
    goal = AbstractState()
    goal.Reide1(0, 1, 1)
    goal.cross(1,2)
    goal.cross(4,2)
    path, path_action = bfs(start, goal)
    print(path)
    paths = bfs_all_path(start, goal, max_depth=3)
    print(paths)
