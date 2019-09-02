import numpy as np
import matplotlib.pyplot as plt
from dynamics_inference.dynamic_models import physbam_3d
from povray_render.sample_spline import sample_b_spline, sample_equdistance
from representation import AbstractState
from state_2_topology import state2topology
from BFS import bfs
import pickle
import pdb


start_state = np.zeros((64,3))
start_state_knots = [np.array([0,0]), np.array([0,0]), np.array([0.5,0]),
                     np.array([0.1,-0.4]), np.array([0.2,0.3]), np.array([0.2,0.3])]
samples = sample_b_spline(start_state_knots)
start_state[:,:2] = sample_equdistance(samples, 64).transpose()
#start_state[:,0] = np.linspace(-0.5,0.5,64)
start_state[:,2] = np.linspace(0.0, 0.1, 64)

action_node = 3
action_knots = [start_state[action_node][:2], start_state[action_node][:2],
                np.array([-0.05,-0.1]),
                np.array([-0.1,-0.1]), np.array([-0.1,-0.1])]
#action_node = 62
#action_knots = [start_state[action_node][:2], start_state[action_node][:2],
#                np.array([0.25,0.4]),
#                np.array([0.3,0.5]), np.array([0.3,0.5])]

#with open('fitted_gaussian.pkl', 'rb') as f:
#    fitted = pickle.load(f)
#gaussian=fitted['left:1 move:R2 over_before_under:1 over_idx:0 under_idx:0']
#gaussian_mean, gaussian_std = gaussian[0], gaussian[1]

#action = np.random.normal(loc=gaussian_mean, scale=gaussian_std)
#action_node = int(action[0]*63)
#action_node = np.clip(action_node, 0, 63)
#action_traj = action[1:5]
#action_knots = [start_state[action_node][:2], start_state[action_node][:2],
#               action_traj[0:2],
#               action_traj[2:4], action_traj[2:4]]

traj = sample_b_spline(action_knots)
traj = sample_equdistance(traj, None, seg_length=0.01).transpose()
moves = traj[1:]-traj[:-1]
moves = np.concatenate([moves, np.zeros((moves.shape[0],1))], axis=-1)
actions = [(0, np.array([0.0,0.0,0.0]))] + [(action_node, np.array([0.0,0.0,0.02]))] + [(action_node, m) for m in moves] + [(action_node, np.array([0.0,0.0,-0.02]))] 
#actions = [(action_node, m) for m in moves]

dynamic_inference = physbam_3d(physbam_args=" -friction 0.176 -stiffen_linear 2.223 -stiffen_bending 0.218")
states = dynamic_inference.execute(start_state, actions, return_3d=True)
#for state in states:
#    plt.plot(state[:,0], state[:,1])
#    plt.show()
#start_abstract_state = AbstractState()
#end_abstract_state = [state2topology(state, update_edges=True, update_faces=False) for state in states]
#batch_abstract_actions = []
#pdb.set_trace()
#for astate, intersection in end_abstract_state:
#    _, path_action = bfs(start_abstract_state, astate)
#    if len(path_action)==1:
#        batch_abstract_actions.append(path_action[0])
#    else:
#        pdb.set_trace()

