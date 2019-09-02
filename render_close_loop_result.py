from physbam_python.rollout_physbam_3d import rollout_single_internal as rollout_single_3d
from physbam_python.rollout_physbam_3d import rollout_single
from physbam_python.state_to_mesh import state_to_mesh
import numpy as np
import matplotlib.pyplot as plt
import pdb

# closed loop actions
file = np.load('/scr-ssd/mengyuan/rope_envs_3d/history.npz')
#actions = file['actions']
#states = file['states']

# open loop actions
actions = []
states = []
for i in range(3):
    with open('nominal_traj_%d.action'%(i+1)) as f:
        lines = f.readlines()
    lines = [l.strip().split() for l in lines]
    action = [np.array([float(l[1]), float(l[2]), 0.0, int(l[0])/63.0]) for l in lines]
    actions.append(np.array([0.0,0.0,0.05,action[0][-1]]))
    actions.extend(action)
    actions.append(np.array([0.0,0.0,-0.05,action[0][-1]]))
    state = np.load('nominal_traj_%d.state.npy'%(i+1))
    states.append(state[0:1]) # match space for additional actions
    states.append(state)
states.append(state[-1:])

actions = np.array(actions)
states = np.concatenate(states, axis=0)

state = np.zeros((64,2))
state[:,0] = np.linspace(0,1,64)
physbam_args = ' -friction 0.176 -stiffen_linear 2.223 -stiffen_bending 1.218' # bending 0.218 friction 0.176
state = state_to_mesh(state)
state = state.dot(np.array([[1,0,0],[0,0,1],[0,-1,0]]))
return_states = []
for i in range(5): #range(actions.shape[0]):
    state = rollout_single_3d(state, actions[i:i+1], physbam_args=' -dt 1e-3 ' + physbam_args,
                              keep_files=True, curve_proto_file='%d.state'%(i+1), action_proto_file='%d.action'%(i+1), output_dir='%d-output'%(i+1))
    n = state.shape[0]
    return_state = (state[:n//2,:] + state[n//2:,:]) * 0.5
    return_states.append(return_state)
#    diff = np.linalg.norm(states[i+1]-return_state[:,:2])
#    print(diff)
#    if i in [29,69,108,119]:
#        plt.plot(states[i+1][:,0], states[i+1][:,1])
#        plt.show()

state = np.zeros((64,2))
state[:,0] = np.linspace(0,1,64)
#state = state_to_mesh(state)
#state = state.dot(np.array([[1,0,0],[0,0,1],[0,-1,0]]))
return_states_2 = rollout_single(state, actions[:5], physbam_args=' -dt 1e-3 ' + physbam_args,
                          keep_files=True, curve_proto_file='200.state', action_proto_file='200.action', output_dir='200-output',
                          return_traj=True)

pdb.set_trace()

for i in range(120):
    diff = np.linalg.norm(return_states[i][:,:2]-return_states_2[i])
    print(diff)
#n = state.shape[0]
#return_state = (state[:n//2,:] + state[n//2:,:]) * 0.5
#diff = np.linalg.norm(states[-1]-return_state[:,:2])
#print(diff)
#plt.plot(return_state[:,0], return_state[:,1])
#plt.show()

