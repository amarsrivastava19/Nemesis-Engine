import networkx as nx
import pandas as pd
import math
import random
from collections import defaultdict
import numpy as np
from nemesis.Agents import Seeker, Hider, Agent
import copy
import pickle
import os

class Environment():

    def __init__(self, nodes, edges, t=0, G=None, nodeToEdgeHash=None, precomputed_paths=None):
        self.timestep = t
        self.nodes = nodes
        self.edges = edges
        if G is not None:
            self.G = G
        else:
            self.G = self.CreateGraph()
        if precomputed_paths is not None:
            self.precomputed_paths = precomputed_paths
        else:
            self.precomputed_paths = self.load_or_precompute_paths(cache_path="data/precomputed_paths.pkl")
        self.mid_lat, self.mid_lon = self.FindCentroid()
        self.seeker_nodes, self.hider_nodes = self.SplitEnvironmentSpawns()
        self.edgeIDHash = self.BuildEdgeIDHash()
        if nodeToEdgeHash is not None:
            self.nodeToEdgeHash = nodeToEdgeHash
        else:
            self.nodeToEdgeHash = self.BuildNodeToEdgesHash()
        self.static_features = self.BuildStaticFeatures()
        self.lastSeen_nodes, self.seekers, self.hider = self.init_Agents()

    def load_or_precompute_paths(self, cache_path="data/precomputed_paths.pkl"):
        if os.path.exists(cache_path):
            print(f"[Nemesis] Loading precomputed paths from {cache_path}")
            with open(cache_path, "rb") as f:
                precomputed_paths = pickle.load(f)
        else:
            print(f"[Nemesis] Building precomputed paths (may take a while)...")
            precomputed_paths = dict(nx.all_pairs_dijkstra_path_length(self.G))
            with open(cache_path, "wb") as f:
                pickle.dump(precomputed_paths, f)
            print(f"[Nemesis] Saved precomputed paths to {cache_path}")
        return precomputed_paths

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
        east = self.mid_lon + lon_step
        west = self.mid_lon - lon_step
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
        random.shuffle(top_nodes)
        return top_nodes


    def init_Agents(self, n_seekers=5):
        avg_lat, avg_lon = self.FindCentroid()
        north, south, east, west =  self.defineSubBoundingBox()
        seeker_nodes, hider_nodes = self.SplitEnvironmentSpawns()

        hider = Hider(self.G)
        hider.ID = 1
        hider.Node = random.choice(list(hider_nodes))
        hider.timestep = 0
        hider.isOnHidingSpot = hider.G.nodes[hider.Node]['isHighPriority'] 
        hider.heading = hider.CalculateNormalizedHeading()
    
        last_seen_nodes = self.GetNearbyNodes(hider.Node, top_k=9)
        dummy = Agent(self.G)
        dummy.Node = last_seen_nodes[0]
        last_seen_heading = dummy.CalculateNormalizedHeading()
    
        seekers = [Seeker(self.G) for i in range(n_seekers)]

        for i,seeker in enumerate(seekers):
            seeker.ID = i
            seeker.Node = random.choice(list(seeker_nodes))
            seeker.heading = seeker.CalculateNormalizedHeading()
            seeker.timestep = 0
            seeker.isOnHidingSpot = seeker.G.nodes[seeker.Node]['isHighPriority']        
            seeker.headingToLastSeen = copy.deepcopy(last_seen_heading)
            seeker.hav_DistanceToLastSeen = seeker.HaversineDistance(dummy.Node)
            seeker.road_DistanceToLastSeen = seeker.RoadDistance(self.precomputed_paths,dummy.Node)

        for i,seeker in enumerate(seekers):
            for seeker2 in seekers:
                seeker.hav_DistanceToTeam.append(seeker.HaversineDistance(seeker2.Node))
                seeker.road_DistanceToTeam.append(seeker.RoadDistance(self.precomputed_paths,seeker2.Node))
            seeker.nearbyTeammates = len([i for i in seeker.hav_DistanceToTeam if i < 5 and i != 0])
            seeker.hav_DistanceToTeam = [100 if i > 5 else i for i in seeker.hav_DistanceToTeam]
            seeker.road_DistanceToTeam = [100 if i > 10 else i for i in seeker.hav_DistanceToTeam]

        return last_seen_nodes, seekers, hider

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


    def simulate_joint_action(self, joint_action):
           
        new_state = Environment(self.nodes, self.edges, t=self.timestep+1, G=self.G, nodeToEdgeHash=self.nodeToEdgeHash, precomputed_paths=self.precomputed_paths)
        new_state.lastSeen_nodes = self.lastSeen_nodes
        new_state.seekers = new_state.seekers = [s.copy() for s in self.seekers]
        new_state.hider = self.hider.copy()
        
        for i, action in enumerate(joint_action):
            new_state.seekers[i].Node = action

        return new_state

    def BuildStateVector(self):
        t = self.timestep # timestep
        H_o = self.hider.heading ## hider's true heading
        omega_h = int(self.hider.isOnHidingSpot) ## if hider in in a hiding spot or not
        d_hav = [i.HaversineDistance(self.hider.Node) for i in self.seekers] ## air distance from each agent to hider's true position
        d_road = [i.RoadDistance(self.precomputed_paths,self.hider.Node) for i in self.seekers] ## road distance from each agent to hider's true position
        alpha = [i.heading for i in self.seekers] 
        beta = self.seekers[0].headingToLastSeen ##heading towards last known location; may be equal to true heading w/ static hider agent
        omega_a = [int(i.isOnHidingSpot) for i in self.seekers] ## are any of the seeker agents on hiding spots
        S_spread = np.array([[j for j in i.hav_DistanceToTeam if j != 0] for i in self.seekers ]).mean()
        S_min_distance = np.array([[j for j in i.hav_DistanceToTeam if j != 0] for i in self.seekers ]).min()
        S_nearTeam =  max(len([i for i in set(np.array([[j for j in i.hav_DistanceToTeam if j != 0] for i in self.seekers ]).flatten()) if i!=100])-1, 0)
        s_t =  [t, H_o] + d_hav + d_road + alpha + [beta] + omega_a + [omega_h] + [S_spread, S_min_distance, S_nearTeam]
        s_t = np.expand_dims(np.array(s_t), axis=0)
        return s_t