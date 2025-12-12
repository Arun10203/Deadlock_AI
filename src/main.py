import joblib
import pandas as pd
import warnings
import networkx as nx
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

class OperatingSystem:
    def __init__(self):
        self.resources = {}
        self.processes = []
        self.allocations = [] # Track who has what: (Process, Resource)
        
        # Load AI
        try:
            self.model = joblib.load('models/deadlock_model.pkl')
            self.encoder = joblib.load('models/resource_encoder.pkl')
        except:
            # Fallback for Colab if files are in root
            try:
                self.model = joblib.load('deadlock_model.pkl')
                self.encoder = joblib.load('resource_encoder.pkl')
            except:
                print("[Warning] AI Models not found. Visualization will run without AI predictions.")

    def add_resource(self, name, total_instances):
        self.resources[name] = {'total': total_instances, 'available': total_instances}

    def predict_safety(self, resource_name, request_amount):
        if not hasattr(self, 'model'): return True # Allow if no AI
        
        available = self.resources[resource_name]['available']
        try:
            res_encoded = self.encoder.transform([resource_name])[0]
            prediction = self.model.predict([[res_encoded, available, request_amount]])
            return prediction[0] == 1
        except:
            return True

    def handle_request(self, pid, resource_name, count):
        print(f"\n[Request] Process {pid} asks for {count} {resource_name}...")
        
        is_safe = self.predict_safety(resource_name, count)
        
        if is_safe:
            self.resources[resource_name]['available'] -= count
            self.allocations.append((pid, resource_name)) # Record for graph
            print(f"  -> [AI DECISION] GRANTED. Remaining {resource_name}: {self.resources[resource_name]['available']}")
            return True
        else:
            print(f"  -> [AI DECISION] BLOCKED (Deadlock Risk).")
            return False

    def visualize_system(self):
        """
        Generates a Resource Allocation Graph (RAG) and saves it as an image.
        """
        print("\n[System] Generating Resource Allocation Graph...")
        G = nx.DiGraph()
        
        # Add Nodes
        color_map = []
        for r in self.resources:
            G.add_node(r)
            color_map.append('lightblue') # Resources are Blue
        
        # Add Edges (Allocations)
        # We need to track unique processes for coloring
        processes = set()
        for pid, r_name in self.allocations:
            p_node = f"Process {pid}"
            processes.add(p_node)
            G.add_edge(r_name, p_node) # Edge from Resource -> Process means "Held by"
        
        for p in processes:
            G.add_node(p)
            color_map.append('lightgreen') # Processes are Green
            
        # Draw
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=2000, font_weight='bold', arrows=True)
        plt.title("AI Deadlock System - Resource Allocation Graph")
        plt.savefig("deadlock_graph.png")
        print("[System] Graph saved as 'deadlock_graph.png'")

if __name__ == "__main__":
    os_sim = OperatingSystem()
    os_sim.add_resource("CPU", 10)
    os_sim.add_resource("Memory", 20)
    
    # Simulate some traffic
    os_sim.handle_request(1, "CPU", 2)
    os_sim.handle_request(1, "Memory", 5)
    os_sim.handle_request(2, "CPU", 1)
    
    # Generate the visualization
    os_sim.visualize_system()
