import networkx as nx
import math
import numpy as np

def FindCentroid(graph):
    lats = [data['latitude'] for _, data in graph.nodes(data=True)]
    lons = [data['longitude'] for _, data in graph.nodes(data=True)]
    avg_lat = sum(lats) / len(lats)
    avg_lon = sum(lons) / len(lons)
    return avg_lat, avg_lon

class Agent():
    def __init__(self,G):
        self.ID = None
        self.Node = ""
        self.heading = 0
        self.timestep = 0
        self.isOnHidingSpot = False 
        self.G = G
        self.status = 'at_node'  # Can be 'at_node' or 'traveling'
        self.destination_node = None
        self.time_to_arrival = 0

    def CalculateNormalizedHeading(self):

        lat_mid,lon_mid = FindCentroid(self.G)
        lat_point = self.G.nodes[self.Node]['latitude']
        lon_point = self.G.nodes[self.Node]['longitude']
        
        lat1 = math.radians(lat_mid)
        lat2 = math.radians(lat_point)
        delta_lon = math.radians(lon_point - lon_mid)
    
        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    
        angle = math.atan2(x, y)
        angle = (angle + 2 * math.pi) % (2 * math.pi)
    
        return angle / (2 * math.pi)  # Normalized to [0, 1]
    
    def HaversineDistance(self, target_id):
        lat1 = self.G.nodes[self.Node]['latitude']
        lon1 = self.G.nodes[self.Node]['longitude']
        lat2 = self.G.nodes[target_id]['latitude']
        lon2 = self.G.nodes[target_id]['longitude']
        R = 3958.8  # Earth radius in miles
    
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
    
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad
    
        a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
        distance = R * c
        return distance
    
    def RoadDistance(self, precomputed_paths, v):
        return precomputed_paths[self.Node][v]

class Seeker(Agent):
    def __init__(self,G):
        super().__init__(G)
        self.headingToLastSeen = 0
        self.hav_DistanceToLastSeen = 0
        self.road_DistanceToLastSeen = 0
        self.nearbyTeammates = 0
        self.hav_DistanceToTeam = []
        self.road_DistanceToTeam = []

    def copy(self):
        new = Seeker(self.G)
        new.ID = self.ID
        new.Node = self.Node
        new.heading = self.heading
        new.timestep = self.timestep
        new.isOnHidingSpot = self.isOnHidingSpot
        new.headingToLastSeen = self.headingToLastSeen
        new.hav_DistanceToLastSeen = self.hav_DistanceToLastSeen
        new.road_DistanceToLastSeen = self.road_DistanceToLastSeen
        new.nearbyTeammates = self.nearbyTeammates
        new.hav_DistanceToTeam = list(self.hav_DistanceToTeam)
        new.road_DistanceToTeam = list(self.road_DistanceToTeam)
        new.status = self.status
        new.destination_node = self.destination_node
        new.time_to_arrival = self.time_to_arrival
        return new


    def BuildObservationVector(self):
        timestep = self.timestep
        n = self.nearbyTeammates
        d_hav = self.hav_DistanceToTeam
        d_road = self.road_DistanceToTeam
        alpha = self.heading
        beta = self.headingToLastSeen
        D_hav = self.hav_DistanceToLastSeen
        D_road = self.road_DistanceToLastSeen
        omega = int(self.isOnHidingSpot)
        is_traveling = 1 if self.status == 'traveling' else 0
        time_left = self.time_to_arrival

        obs_i = [timestep, n, alpha, beta, D_hav, D_road, omega, is_traveling, time_left] + d_hav + d_road
        obs_i = np.array(obs_i)  # shape (19,)
        #obs_i = np.expand_dims(obs_i, axis=0) 
        return obs_i
        
class Hider(Agent):
    def __init__(self, G):
        super().__init__(G)

    def copy(self):
        new = Hider(self.G)
        new.ID = self.ID
        new.Node = self.Node
        new.heading = self.heading
        new.timestep = self.timestep
        new.isOnHidingSpot = self.isOnHidingSpot
        new.status = self.status
        new.destination_node = self.destination_node
        new.time_to_arrival = self.time_to_arrival
        return new




