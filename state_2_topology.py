from topology.representation import AbstractState, Face
from povray_render.lift_3d import intersection
import numpy as np
import pdb

def find_intersections(state):
    intersections = []
    for i in range(state.shape[0]-1):
        for j in range(i+2, state.shape[0]-1):
            intersect = intersection(state[i][:2], state[i+1][:2], state[j][:2], state[j+1][:2])
            if intersect is not None:
                alpha = np.linalg.norm(intersect-state[i][:2]) / np.linalg.norm(state[i+1][:2]-state[i][:2])
                beta = np.linalg.norm(intersect-state[j][:2]) / np.linalg.norm(state[j+1][:2]-state[j][:2])
                h_i = alpha*state[i+1][2]+(1-alpha)*state[i][2]
                h_j = beta*state[j+1][2]+(1-beta)*state[j][2]
                over = h_i > h_j
                sign = np.cross(state[i+1][:2]-state[i][:2], state[j+1][:2]-state[j][:2])
                sign = sign > 0
                if not over:
                    sign = not sign
                over = 1 if over else -1
                sign = 1 if sign else -1
                intersections.append((i,j, over, sign))
    return intersections

def intersect2topology(intersections, update_edges=True, update_faces=True):
    points = [it[0] for it in intersections] + [it[1] for it in intersections]
    points.sort()
    pointDict = {p:i for i,p in enumerate(points)}
    topology = AbstractState()

    for p in points:
        topology.addPoint(1)
    for it in intersections:
        if it[2]==1:
            topology.point_intersect(pointDict[it[0]]+1, pointDict[it[1]]+1, it[3])
        else:
            topology.point_intersect(pointDict[it[1]]+1, pointDict[it[0]]+1, it[3])
        if update_edges:
            if it[3] != it[2]:  # XOR
                i, j = pointDict[it[0]]+1, pointDict[it[1]]+1
            else:
                i, j = pointDict[it[1]]+1, pointDict[it[0]]+1
            topology.link_edges(2*i-2, 2*j-1)
            topology.link_edges(2*j-2, 2*i)
            topology.link_edges(2*i+1, 2*j)
            topology.link_edges(2*j+1, 2*i-1)
    if update_faces:
        visited_edge = [False] * len(topology.edges)
        start_edge_idx = 0
        next_edge = topology.edges[start_edge_idx]
        visited_edge[start_edge_idx] = True
        while next_edge.next != start_edge_idx:
            visited_edge[next_edge.next] = True
            next_edge = topology.edges[next_edge.next]
        while not all(visited_edge):
            start_edge_idx = visited_edge.index(False)
            new_face_idx = len(topology.faces)
            topology.faces.append(Face(start_edge_idx))
            next_edge = topology.edges[start_edge_idx]
            visited_edge[start_edge_idx]=True
            next_edge.face = new_face_idx
            while next_edge.next != start_edge_idx:
                visited_edge[next_edge.next] = True
                next_edge = topology.edges[next_edge.next]
                next_edge.face = new_face_idx
    return topology

def state2topology(state, update_edges, update_faces):
    intersections = find_intersections(state)
    it_points = [it[0] for it in intersections] + [it[1] for it in intersections]
    new_intersections = intersections
    while len(set(it_points)) < len(it_points):
        # deduplicate by breaking segment into smaller ones.
        it_points = sorted(it_points) + [64]
        segs = []
        current_pt = 0
        current_rep_count = 0
        for it in it_points:
            if it > current_pt:
                if current_rep_count < 2:
                    segs.append(state[current_pt:it])
                else:
                    alpha = np.linspace(0.0,1.0,current_rep_count*2+1)
                    points = state[current_pt]*(1-alpha[0:-1,np.newaxis]) + state[current_pt+1]*alpha[0:-1,np.newaxis]
                    segs.append(points)
                    segs.append(state[current_pt+1:it])
                current_pt = it
                current_rep_count = 1
            else:
                current_rep_count +=1

        new_state = np.concatenate(segs, axis=0)
        new_intersections = find_intersections(new_state)
        it_points = [it[0] for it in new_intersections] + [it[1] for it in new_intersections]

    topology = intersect2topology(new_intersections, update_edges, update_faces)
    return topology, intersections
