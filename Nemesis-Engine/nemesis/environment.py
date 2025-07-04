import networkx as nx
import pandas as pd
import math
import random
from collections import defaultdict
import numpy as np

class Environment():

    def __init__(self, nodes, edges, t=0):
        self.timestep = 0
        self.nodes = nodes
        self.edges = edges
        self.G = self.CreateGraph()
        self.mid_lat, self.mid_lon = self.FindCentroid()
        self.seeker_nodes, self.hider_nodes = self.SplitEnvironmentSpawns()
        self.edgeIDHash = self.BuildEdgeIDHash()
        self.nodeToEdgeHash = self.BuildNodeToEdgesHash()
        self.static_features = self.BuildStaticFeatures()
        self.lastSeen_nodes = []
        self.seekers = None
        self.hider = None
        self.lastSeen_nodes = []

    def CreateGraph(self):
        G = nx.Graph()
        for _, row in self.nodes.iterrows():
            G.add_node(row['Node'], **row.drop('Node').to_dict())
        for _, row in self.edges.iterrows():
            G.add_edge(row['u'], row['v'], **row.drop(['u', 'v']).to_dict())
        return G

    def FindCentroid(self):
        lats = [data['latitude'] for _, data in self.G.nodes(data=True)]
        lons = [data['longitude'] for _, data in self.G.nodes(data=True)]
        avg_lat = sum(lats) / len(lats)
        avg_lon = sum(lons) / len(lons)
        return avg_lat, avg_lon

    def defineSubBoundingBox(self, box_side_length=5):
        lat_step = box_side_length/69
        lon_step = box_side_length/(69*math.cos(math.radians(self.mid_lat)))
        north = self.mid_lat + lat_step
        south = self.mid_lat - lat_step
        east = self.mid_lon - lon_step
        west = self.mid_lon + lon_step
        return north, south , east, west 

    def SplitEnvironmentSpawns(self):
        sub_north, sub_south, sub_east, sub_west = self.defineSubBoundingBox()
        seeker_nodes = self.nodes[(self.nodes['longitude'].between(sub_west,sub_east)) & (self.nodes['latitude'].between(sub_south,sub_north))]
        seeker_nodes = seeker_nodes['Node'].values
    
        hider_nodes = self.nodes[(~self.nodes['Node'].isin(seeker_nodes))]
        hider_nodes = hider_nodes['Node'].values
        return seeker_nodes, hider_nodes

    def GetNearbyNodes(self, source_node, top_k = 9):
        lengths = nx.single_source_dijkstra_path_length(self.G, source_node, weight='Distance_miles')
        lengths.pop(source_node, None)
        sorted_nodes = sorted(lengths.items(), key=lambda x: x[1])
        top_nodes = [node for node, dist in sorted_nodes[:top_k]]

        if not top_nodes:
            raise ValueError("No nearby nodes found.")

        top_nodes = top_nodes + [source_node]
        top_nodes = random.shuffle(top_nodes)
        return top_nodes

    def BuildEdgeIDHash(self):
        return dict(zip(self.edges['edge_id'], zip(self.edges['u'], self.edges['v'])))

    def BuildNodeToEdgesHash(self):
        node_to_edges = defaultdict(list)
    
        for _, row in self.edges.iterrows():
            u = row['u']
            edge_id = row['edge_id']
            node_to_edges[u].append(edge_id)
        return node_to_edges

    def BuildStaticFeatures(self):
        return self.edges[['Distance_miles', 'TimeToCross_minutes', 'lon_u',
                            'lat_u', 'isHighPriority_u', 'lon_v', 'lat_v', 
                            'isHighPriority_v','heading_u', 'heading_v']].values

    def BuildStateVector(self):
        t = t # timestep
        H_o = self.hider.heading ## hider's true heading
        omega_h = int(self.hider.isOnHidingSpot) ## if hider in in a hiding spot or not
        d_hav = [i.HaversineDistance(self.hider.Node) for i in self.seekers] ## air distance from each agent to hider's true position
        d_road = [i.RoadDistance(self.seekers.Node) for i in self.seekers] ## road distance from each agent to hider's true position
        alpha = [i.heading for i in self.seekers] 
        beta = self.seekers[0].headingToLastSeen ##heading towards last known location; may be equal to true heading w/ static hider agent
        omega_a = [int(i.isOnHidingSpot) for i in self.seekers] ## are any of the seeker agents on hiding spots
        S_spread = np.array([[j for j in i.hav_DistanceToTeam if j != 0] for i in self.seekers ]).mean()
        S_min_distance = np.array([[j for j in i.hav_DistanceToTeam if j != 0] for i in self.seekers ]).min()
        S_nearTeam =  max(len([i for i in set(np.array([[j for j in i.hav_DistanceToTeam if j != 0] for i in self.seekers ]).flatten()) if i!=100])-1, 0)
        s_t =  [t, H_o] + d_hav + d_road + alpha + [beta] + omega_a + [omega_h] + [S_spread, S_min_distance, S_nearTeam]
        s_t = np.expand_dims(np.array(s_t), axis=0)
        return s_t