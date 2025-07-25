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

AGENT_SPEED_MILES_PER_TIMESTEP = 1.0 # Example: 60 mph at 1 min timesteps


def CreateGraph(nodes,edges):
    G = nx.Graph()
    for _, row in nodes.iterrows():
        G.add_node(row['Node'], **row.drop('Node').to_dict())
    for _, row in edges.iterrows():
        G.add_edge(row['u'], row['v'], **row.drop(['u', 'v']).to_dict())
    return G

def load_or_precompute_paths(cache_path="data/precomputed_paths.pkl"):
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

def FindCentroid(G):
    lats = [data['latitude'] for _, data in G.nodes(data=True)]
    lons = [data['longitude'] for _, data in G.nodes(data=True)]
    avg_lat = sum(lats) / len(lats)
    avg_lon = sum(lons) / len(lons)
    return avg_lat, avg_lon

def defineSubBoundingBox(mid_lat, mid_lon, box_side_length=5):
    lat_step = box_side_length/69
    lon_step = box_side_length/(69*math.cos(math.radians(mid_lat)))
    north = mid_lat + lat_step
    south = mid_lat - lat_step
    east = mid_lon + lon_step
    west = mid_lon - lon_step
    return north, south , east, west 

def SplitEnvironmentSpawns(nodes, sub_north, sub_south, sub_east, sub_west):
    seeker_nodes = nodes[(nodes['longitude'].between(sub_west,sub_east)) & (nodes['latitude'].between(sub_south,sub_north))]
    seeker_nodes = seeker_nodes['Node'].values
    
    hider_nodes = nodes[(~nodes['Node'].isin(seeker_nodes))]
    hider_nodes = hider_nodes['Node'].values
    return seeker_nodes, hider_nodes

def GetNearbyNodes(G, source_node, top_k = 9):
    lengths = nx.single_source_dijkstra_path_length(G, source_node, weight='Distance_miles')
    lengths.pop(source_node, None)
    sorted_nodes = sorted(lengths.items(), key=lambda x: x[1])
    top_nodes = [node for node, dist in sorted_nodes[:top_k]]

    if not top_nodes:
        raise ValueError("No nearby nodes found.")

    top_nodes = top_nodes + [source_node]
    random.shuffle(top_nodes)
    return top_nodes


def init_Agents(nodes, G, n_seekers=5):
    avg_lat, avg_lon = FindCentroid(G)
    north, south, east, west =  defineSubBoundingBox(avg_lat, avg_lon)
    seeker_nodes, hider_nodes = SplitEnvironmentSpawns(nodes, north, south, east, west)

    hider = Hider(G)
    hider.ID = 1
    hider.Node = random.choice(list(hider_nodes))
    hider.timestep = 0
    hider.isOnHidingSpot = hider.G.nodes[hider.Node]['isHighPriority'] 
    hider.heading = hider.CalculateNormalizedHeading()
    
    last_seen_nodes = GetNearbyNodes(G, hider.Node, top_k=9)
    dummy = Agent(G)
    dummy.Node = last_seen_nodes[0]
    last_seen_heading = dummy.CalculateNormalizedHeading()
    
    seekers = [Seeker(G) for i in range(n_seekers)]

    for i,seeker in enumerate(seekers):
        seeker.ID = i
        seeker.Node = random.choice(list(seeker_nodes))
        seeker.heading = seeker.CalculateNormalizedHeading()
        seeker.timestep = 0
        seeker.isOnHidingSpot = seeker.G.nodes[seeker.Node]['isHighPriority']        
        seeker.headingToLastSeen = copy.deepcopy(last_seen_heading)
        seeker.hav_DistanceToLastSeen = seeker.HaversineDistance(dummy.Node)
        seeker.road_DistanceToLastSeen = seeker.RoadDistance(precomputed_paths,dummy.Node)

    for i,seeker in enumerate(seekers):
        for seeker2 in seekers:
            seeker.hav_DistanceToTeam.append(seeker.HaversineDistance(seeker2.Node))
            seeker.road_DistanceToTeam.append(seeker.RoadDistance(precomputed_paths,seeker2.Node))
        seeker.nearbyTeammates = len([i for i in seeker.hav_DistanceToTeam if i < 5 and i != 0])
        seeker.hav_DistanceToTeam = [100 if i > 5 else i for i in seeker.hav_DistanceToTeam]
        seeker.road_DistanceToTeam = [100 if i > 10 else i for i in seeker.hav_DistanceToTeam]

    return last_seen_nodes, seekers, hider

def BuildEdgeIDHash(edges):
    return dict(zip(edges['edge_id'], zip(edges['u'], edges['v'])))

def BuildNodeToEdgesHash(edges):
    node_to_edges = defaultdict(list)
    
    for _, row in edges.iterrows():
        u = row['u']
        edge_id = row['edge_id']
        node_to_edges[u].append(edge_id)
    return node_to_edges

def BuildStaticFeatures(edges):
    return edges[['Distance_miles', 'TimeToCross_minutes', 'lon_u',
                        'lat_u', 'isHighPriority_u', 'lon_v', 'lat_v', 
                        'isHighPriority_v','heading_u', 'heading_v']].values


nodes = pd.read_csv('data/DSMMetro_nodes.csv')
edges = pd.read_csv('data/DSMMetro_edges.csv')
precomputed_paths = load_or_precompute_paths(cache_path="data/precomputed_paths.pkl")
G = CreateGraph(nodes,edges)
avg_lat, avg_lon = FindCentroid(G)
north,south ,east,west =  defineSubBoundingBox(avg_lat, avg_lon)
seeker_nodes, hider_nodes = SplitEnvironmentSpawns(nodes, north,south ,east,west)
edgeIDHash = BuildEdgeIDHash(edges)
nodeToEdgeHash = BuildNodeToEdgesHash(edges)
static_features = BuildStaticFeatures(edges)

class Environment():

    def __init__(self, t=0, lastSeen_nodes = None,seekers = None, hider = None):
        self.timestep = t

        if ((lastSeen_nodes is None) and (seekers is None) and (hider is None)):
            self.lastSeen_nodes, self.seekers, self.hider = init_Agents(nodes, G)
        else:
            self.lastSeen_nodes = lastSeen_nodes
            self.seekers = seekers
            self.hider = hider

    def copy(self):
        """
        Creates a deep, independent copy of the current environment state.
        """
        # Create copies of the agent objects using their own .copy() methods
        copied_seekers = [seeker.copy() for seeker in self.seekers]
        copied_hider = self.hider.copy()
        
        # Create a copy of the last seen nodes list
        copied_lastSeen = self.lastSeen_nodes.copy()

        # Create a new Environment instance with the copied data
        new_env = Environment(
            t=self.timestep, 
            lastSeen_nodes=copied_lastSeen,
            seekers=copied_seekers, 
            hider=copied_hider
        )
        return new_env
       

    # def simulate_joint_action(self, joint_action):
           
    #     new_state = Environment(t=self.timestep+1, lastSeen_nodes = self.lastSeen_nodes, seekers = self.seekers, hider = self.hider)
        
    #     for i, action in enumerate(joint_action):
    #         action = edgeIDHash[action][1]
    #         new_state.seekers[i].Node = action

    #     return new_state

    def simulate_joint_action(self, joint_action):
        # CRITICAL: Ensure this creates a deep copy of the environment,
        # especially the list of agents, to avoid state corruption.
        new_state = self.copy() # Assuming you have or create a .copy() method
        new_state.timestep += 1

        for i, agent in enumerate(new_state.seekers):
            action_id = joint_action[i]

            # Case 1: Agent is at a node and chose a new edge to travel.
            if agent.status == 'at_node':
                # The action is an edge ID. -1 means the agent stays put.
                if action_id != -1: 
                    destination_node = edgeIDHash[action_id][1]
                    agent.status = 'traveling'
                
                    # Get edge data
                    edge_len_miles = agent.RoadDistance(precomputed_paths, destination_node)
                    destination_node = edgeIDHash[action_id][1]
                
                    # Calculate travel time in discrete timesteps
                    agent.time_to_arrival = math.ceil(edge_len_miles / AGENT_SPEED_MILES_PER_TIMESTEP)
                    agent.destination_node = destination_node

            # Case 2: Agent is already traveling.
            elif agent.status == 'traveling':
                # The only possible action was -1 ("continue_travel")
                agent.time_to_arrival -= 1

                # Check for arrival
                if agent.time_to_arrival <= 0:
                    agent.status = 'at_node'
                    agent.Node = agent.destination_node
                    agent.destination_node = None # Clean up

        return new_state


    def BuildStateVector(self):
        t = self.timestep # timestep
        H_o = self.hider.heading ## hider's true heading
        omega_h = int(self.hider.isOnHidingSpot) ## if hider in in a hiding spot or not
        d_hav = [i.HaversineDistance(self.hider.Node) for i in self.seekers] ## air distance from each agent to hider's true position
        d_road = [i.RoadDistance(precomputed_paths,self.hider.Node) for i in self.seekers] ## road distance from each agent to hider's true position
        alpha = [i.heading for i in self.seekers] 
        beta = self.seekers[0].headingToLastSeen ##heading towards last known location; may be equal to true heading w/ static hider agent
        omega_a = [int(i.isOnHidingSpot) for i in self.seekers] ## are any of the seeker agents on hiding spots
        S_spread = np.array([[j for j in i.hav_DistanceToTeam if j != 0] for i in self.seekers ]).mean()
        S_min_distance = np.array([[j for j in i.hav_DistanceToTeam if j != 0] for i in self.seekers ]).min()
        S_nearTeam =  max(len([i for i in set(np.array([[j for j in i.hav_DistanceToTeam if j != 0] for i in self.seekers ]).flatten()) if i!=100])-1, 0)

        seeker_statuses = [1 if s.status == 'traveling' else 0 for s in self.seekers]
    
        # Create a list of remaining travel times
        seeker_travel_times = [s.time_to_arrival for s in self.seekers]

        s_t =  [t, H_o] + d_hav + d_road + alpha + [beta] + omega_a + [omega_h] + seeker_statuses + seeker_travel_times + [S_spread, S_min_distance, S_nearTeam]
        s_t = np.expand_dims(np.array(s_t), axis=0)
        return s_t