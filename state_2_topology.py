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
                over = state[i][2]+state[i+1][2] > state[j][2]+state[j+1][2]
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
                next_edge.face = new_face_idx
                next_edge = topology.edges[next_edge.next]
    return topology

def state2topology(state, update_edges, update_faces):
    intersections = find_intersections(state)
    topology = intersect2topology(intersections, update_edges, update_faces)
    return topology, intersections
