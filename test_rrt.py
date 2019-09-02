from rrt import RRT
from representation import AbstractState
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

init_state= np.zeros((64,3))
init_state[:,0] = np.linspace(-0.5,0.5,64)

init_topology = AbstractState()
topology_path = [deepcopy(init_topology)]
init_topology.Reide1(0, left=1, sign=1)
topology_path.append(deepcopy(init_topology))
init_topology.cross(0,1, sign=1)
topology_path.append(deepcopy(init_topology))
rrt_search = RRT(init_state, topology_path, max_samples = 2000)
trajectory = rrt_search.rrt_search()
if trajectory is not None:
    waypoints, actions = trajectory
    np.save('rrt_waypoints.npy', np.array(waypoints))
    for i,ac in enumerate(actions):
        np.save('rrt_actions_%d.npy'%(i), np.array(ac))
    plt.plot(waypoints[-1][:,0], waypoints[-1][:,1])
    plt.show()
