import random
import numpy as np
import heapq
from heaptree import Tree
import pdb
from povray_render.sample_spline import sample_b_spline
from dynamics_inference.dynamic_models import physbam_2d
from state_2_topology import state2topology

class RRT(object):
    def __init__(self, x_init, topology_path, max_samples):
        """
        Template RRT planner
        :param x_init: tuple, initial location
        :param x_goal: tuple, goal location
        :param max_samples: int, maximum number of samples
        """
        self.samples_taken = 0
        self.max_samples = max_samples
        self.x_init = x_init
        self.topology_path = topology_path
        self.trees = [Tree()]  # list of all trees
        self.mental_dynamics = physbam_2d(physbam_args=' -disable_collisions -stiffen_linear 25.884  -stiffen_bending 64.297')

    def select_node(self, tree):
        priority, id, node = tree.V[0]
        try:
            heapq.heapreplace(tree.V, (priority*1.005, id, node))
        except:
            pass
        return (priority, id, node)

    def sample_branch(self, parent):
        action_node = np.random.choice(list(range(4,60,5)))
        action_traj = np.random.uniform(-0.5,0.5,4)
        start_state = np.array(parent).reshape((64,3))
        knots = [start_state[action_node][:2], start_state[action_node][:2],
                 action_traj[0:2],
                 action_traj[2:4], action_traj[2:4]]
        traj = sample_b_spline(knots).transpose()
        moves = traj[1:]-traj[:-1]
        actions = [(action_node, m) for m in moves]
        child = self.mental_dynamics.execute(start_state[:,:2], actions)
        child_z = 0.1 - np.abs(np.linspace(0,63,64)-action_node) / 630.0  # artificial z for intersection ordering.
        child = np.concatenate([child,child_z[:,np.newaxis]],axis=-1)
        return child, actions

    def score_priority(self, parent_priority, parent, child):
        parent_topology, intersections = state2topology(parent, update_edges=True, update_faces=False)
        child_topology, intersections = state2topology(child, update_edges=True, update_faces=False)
        parent_index = self.topology_path.index(parent_topology)
        try:
            child_index = self.topology_path.index(child_topology)
        except ValueError:
            return None
        if child_index < parent_index:
            return None
        child_priority = parent_priority * 2 / (2**(child_index-parent_index))
        return child_priority

    def is_solution(self, node):
        topology, _ = state2topology(node, update_edges=True, update_faces=False)
        if topology == self.topology_path[-1]:
            return True
        else:
            return False

    def rrt_search(self):
        """
        Create and return a Rapidly-exploring Random Tree, keeps expanding until can connect to goal
        https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
        :return: list representation of path, dict representing edges of tree in form E[child] = parent
        """
        self.trees[0].add_root((1.0, 0, self.x_init))

        while self.samples_taken < self.max_samples:
            parent = self.select_node(self.trees[0])
            child, traj = self.sample_branch(parent[2])
            priority = self.score_priority(parent[0], parent[2], child)
            if priority is not None:
                self.samples_taken += 1
                self.trees[0].add_leaf(parent, (priority, self.samples_taken, child), traj)
                if self.is_solution(child):
                    waypoints, actions = self.trees[0].reconstruct_path(child)
                    waypoints = [np.array(w).reshape((64,3)) for w in waypoints]
                    return waypoints, actions
        print("Cannot find solution!")
        return None
