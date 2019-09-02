# Run this code only using python3
from topology.representation import AbstractState
import random, copy
import pdb

def generate_next(state):
    # undo cross action
    print("undo cross head")
    new_state = copy.deepcopy(state)
    success = new_state.undo_cross(True)
    if success:
        action = {'move':'undo_cross', 'head':True}
        yield new_state, action
    print("undo cross tail")
    new_state =copy.deepcopy(state)
    success = new_state.undo_cross(False)
    if success:
        action = {'move':'undo_cross', 'head':False}
        yield new_state, action

    # undo R1 action
    for idx in range(1, state.pts+1):
        print("undo R1", idx)
        new_state = copy.deepcopy(state)
        success = new_state.undo_Reide1(idx)
        if success:
            action = {'move':'undo_R1', 'idx':idx}
            yield new_state, action

    # R2 action
    for over_idx in range(1, state.pts):
        for under_idx in range(1, state.pts):
            print("undo R2", over_idx, under_idx)
            new_state = copy.deepcopy(state)
            success = new_state.undo_Reide2(over_idx, under_idx)
            if success:
                action = {'move':'undo_R2', 'over_idx':over_idx, 'under_idx':under_idx}
                yield new_state, action


def bfs(start, goal):
    visited, parents, actions = [start], [0], [{}]
    head = 0
    while head < len(visited):
        state = visited[head]
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
    return path

if __name__=="__main__":
    start = AbstractState()
    goal = AbstractState()
    goal.Reide1(0, 1, 1)
    goal.cross(1,2)
    goal.cross(4,2)
    path = bfs(goal, start)
    print(path[::-1])
