import numpy as np
import matplotlib.pyplot as plt
from physbam_python.rollout_physbam_2d import rollout_single
from dynamics_inference.dynamic_models import physbam_2d
from povray_render.sample_spline import sample_b_spline, sample_equdistance
from representation import AbstractState
from state_2_topology import state2topology
from BFS import bfs
import pickle
import pdb


with open('fitted_gaussian.pkl', 'rb') as f:
    fitted = pickle.load(f)
gaussian=fitted['idx:0 left:-1 move:R1 sign:-1']
gaussian_mean, gaussian_std = gaussian[0], gaussian[1]

start_state = np.zeros((64,3))
start_state[:,0] = np.linspace(-0.5, 0.5, 64)

batch_actions = []
traj_params = []
for _ in range(200):
    action = np.random.normal(loc=gaussian_mean, scale=gaussian_std)
    action_node = int(action[0]*63)
    action_node = np.clip(action_node, 0, 63)
    action_traj = action[1:5]
    knots = [start_state[action_node][:2], start_state[action_node][:2],
             action_traj[0:2],
             action_traj[2:4], action_traj[2:4]]
    traj = sample_b_spline(knots)
    traj = sample_equdistance(traj, None, seg_length=0.01).transpose()
    print(traj.shape)
    moves = traj[1:]-traj[:-1]
    actions = [(action_node, m) for m in moves]
    batch_actions.append(actions)
    traj_params.append((action_node/63, action_traj))

dynamic_inference = physbam_2d(' -disable_collisions -stiffen_linear 25.884  -stiffen_bending 64.297')
states = dynamic_inference.execute_batch(start_state[:,:2], batch_actions)

lifted_states = []
for state,action in zip(states, batch_actions):
    state_z = np.arange(64)
    state_z = (state_z - action[0][0])**2/32
    state_z = np.exp(-state_z)
    state = np.concatenate([state, state_z[:,np.newaxis]],axis=-1)
    lifted_states.append(state)
start_abstract_state = AbstractState()
end_abstract_state = [state2topology(state, update_edges=True, update_faces=False) for state in lifted_states]

count = 856

for i, (astate, intersection) in enumerate(end_abstract_state):
    intersect_points = [i[0] for i in intersection] + [i[1] for i in intersection]
    if len(set(intersect_points)) < len(intersect_points):
        continue
    _, path_action = bfs(start_abstract_state, astate)
    if len(path_action)==1:
        np.savetxt('%04d.txt'%(count), lifted_states[i])
        with open('%04d_act_param.txt'%(count), 'w') as f:
            f.write('%d %f %f %f %f\n'%(traj_params[i][0], traj_params[i][1][0],
                                                           traj_params[i][1][1],
                                                           traj_params[i][1][2],
                                                           traj_params[i][1][3]))
        count += 1

print(count)
