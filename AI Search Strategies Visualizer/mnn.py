import tkinter as tk
from tkinter import messagebox
from abc import ABC, abstractmethod
import heapq
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx

# 1. Define the Node Class
class Node:
    def __init__(self, value, heuristic=0):
        self.value = value
        self._neighbors = []
        self._neighbors_with_cost = []
        self.heuristic_value = heuristic

    def add_neighbor(self, neighbor, cost=1):
        self._neighbors.append(neighbor)
        self._neighbors_with_cost.append((neighbor, cost))

    def get_neighbors(self):
        return self._neighbors

    def get_neighbors_with_cost(self):
        return self._neighbors_with_cost

    def heuristic(self, goal):
        return self.heuristic_value

# 2. Define the Search Strategy Interface
class SearchStrategy(ABC):
    @abstractmethod
    def search(self, start, goal):
        pass

# 3. Implement Search Strategies
class BFS(SearchStrategy):
    def search(self, start, goal):
        frontier = deque([start])
        explored = set()
        parent = {start: None}

        while frontier:
            current = frontier.popleft()
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1]

            explored.add(current)
            for neighbor in current.get_neighbors():
                if neighbor not in explored and neighbor not in frontier:
                    frontier.append(neighbor)
                    parent[neighbor] = current
        return None

class DFS(SearchStrategy):
    def search(self, start, goal):
        frontier = [start]
        explored = set()
        parent = {start: None}

        while frontier:
            current = frontier.pop()
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1]

            explored.add(current)
            for neighbor in current.get_neighbors():
                if neighbor not in explored and neighbor not in frontier:
                    frontier.append(neighbor)
                    parent[neighbor] = current
        return None

class UCS(SearchStrategy):
    def search(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start))
        explored = set()
        parent = {start: None}
        cost = {start: 0}

        while frontier:
            current_cost, current = heapq.heappop(frontier)
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1]

            explored.add(current)
            for neighbor, weight in current.get_neighbors_with_cost():
                new_cost = current_cost + weight
                if neighbor not in explored or new_cost < cost.get(neighbor, float('inf')):
                    cost[neighbor] = new_cost
                    heapq.heappush(frontier, (new_cost, neighbor))
                    parent[neighbor] = current
        return None

class Greedy(SearchStrategy):
    def search(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (start.heuristic(goal), start))
        explored = set()
        parent = {start: None}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1]

            explored.add(current)
            for neighbor in current.get_neighbors():
                if neighbor not in explored:
                    heapq.heappush(frontier, (neighbor.heuristic(goal), neighbor))
                    parent[neighbor] = current
        return None

class AStar(SearchStrategy):
    def search(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0 + start.heuristic(goal), start))
        explored = set()
        parent = {start: None}
        cost = {start: 0}

        while frontier:
            current_cost, current = heapq.heappop(frontier)
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = parent[current]
                return path[::-1]

            explored.add(current)
            for neighbor, weight in current.get_neighbors_with_cost():
                new_cost = cost[current] + weight
                if neighbor not in explored or new_cost < cost.get(neighbor, float('inf')):
                    cost[neighbor] = new_cost
                    heapq.heappush(frontier, (new_cost + neighbor.heuristic(goal), neighbor))
                    parent[neighbor] = current
        return None

# 4. Define the GUI Application
class SearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Search Strategies")

        self.nodes = {}

        self.strategies = {
            'BFS': BFS(),
            'DFS': DFS(),
            'UCS': UCS(),
            'Greedy': Greedy(),
            'A*': AStar()
        }

        self.create_widgets()

    def create_widgets(self):
        # Strategy Frame
        self.strategy_frame = tk.Frame(self.root, padx=10, pady=10)
        self.strategy_frame.pack(fill=tk.X)
        tk.Label(self.strategy_frame, text="Strategy:", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5)
        self.algorithm_var = tk.StringVar()
        self.algorithm_var.trace("w", self.toggle_heuristic_input)
        tk.OptionMenu(self.strategy_frame, self.algorithm_var, *self.strategies.keys()).grid(row=0, column=1, padx=5, pady=5)

        # Node Frame
        self.node_frame = tk.Frame(self.root, padx=10, pady=10)
        self.node_frame.pack(fill=tk.X)
        tk.Label(self.node_frame, text="Node:", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5)
        self.node_name_entry = tk.Entry(self.node_frame)
        self.node_name_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self.node_frame, text="Heuristic:", font=("Arial", 12)).grid(row=0, column=2, padx=5, pady=5)
        self.heuristic_entry = tk.Entry(self.node_frame)
        self.heuristic_entry.grid(row=0, column=3, padx=5, pady=5)
        tk.Button(self.node_frame, text="Add Node", command=self.add_node).grid(row=0, column=4, padx=5, pady=5)

        # Neighbor Frame
        self.neighbor_frame = tk.Frame(self.root, padx=10, pady=10)
        self.neighbor_frame.pack(fill=tk.X)
        tk.Label(self.neighbor_frame, text="From:", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5)
        self.from_node_entry = tk.Entry(self.neighbor_frame)
        self.from_node_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self.neighbor_frame, text="To:", font=("Arial", 12)).grid(row=0, column=2, padx=5, pady=5)
        self.to_node_entry = tk.Entry(self.neighbor_frame)
        self.to_node_entry.grid(row=0, column=3, padx=5, pady=5)
        tk.Label(self.neighbor_frame, text="Cost:", font=("Arial", 12)).grid(row=0, column=4, padx=5, pady=5)
        self.cost_entry = tk.Entry(self.neighbor_frame)
        self.cost_entry.grid(row=0, column=5, padx=5, pady=5)
        tk.Button(self.neighbor_frame, text="Add Cost", command=self.add_neighbor).grid(row=0, column=6, padx=5, pady=5)

        # Search Frame
        self.search_frame = tk.Frame(self.root, padx=10, pady=10)
        self.search_frame.pack(fill=tk.X)
        tk.Label(self.search_frame, text="Start:", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5)
        self.start_entry = tk.Entry(self.search_frame)
        self.start_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(self.search_frame, text="Goal:", font=("Arial", 12)).grid(row=0, column=2, padx=5, pady=5)
        self.goal_entry = tk.Entry(self.search_frame)
        self.goal_entry.grid(row=0, column=3, padx=5, pady=5)
        tk.Button(self.search_frame, text="Search", command=self.perform_search).grid(row=0, column=4, padx=5, pady=5)

        # Visualize Button
        tk.Button(self.root, text="Visualize Graph", command=self.visualize_graph, padx=10, pady=10).pack()

        # Result Frame
        self.result_frame = tk.Frame(self.root, padx=10, pady=10)
        self.result_frame.pack(fill=tk.X)
        self.result_label = tk.Label(self.result_frame, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def toggle_heuristic_input(self, *args):
        algorithm = self.algorithm_var.get()
        if algorithm in ('Greedy', 'A*'):
            self.heuristic_entry.config(state=tk.NORMAL)
        else:
            self.heuristic_entry.config(state=tk.DISABLED)

    def add_node(self):
        node_name = self.node_name_entry.get().upper()
        heuristic_value = self.heuristic_entry.get()
        if node_name in self.nodes:
            messagebox.showerror("Error", "Node already exists!")
            return

        try:
            heuristic_value = int(heuristic_value) if heuristic_value else 0
        except ValueError:
            messagebox.showerror("Error", "Heuristic must be an integer!")
            return

        self.nodes[node_name] = Node(node_name, heuristic_value)
        self.node_name_entry.delete(0, tk.END)
        self.heuristic_entry.delete(0, tk.END)
        messagebox.showinfo("Success", f"Node {node_name} added!")

    def add_neighbor(self):
        from_node = self.from_node_entry.get().upper()
        to_node = self.to_node_entry.get().upper()
        try:
            cost = int(self.cost_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Cost must be an integer!")
            return

        if from_node not in self.nodes or to_node not in self.nodes:
            messagebox.showerror("Error", "Both nodes must exist!")
            return

        self.nodes[from_node].add_neighbor(self.nodes[to_node], cost)
        self.from_node_entry.delete(0, tk.END)
        self.to_node_entry.delete(0, tk.END)
        self.cost_entry.delete(0, tk.END)
        messagebox.showinfo("Success", f"Neighbor added from {from_node} to {to_node} with cost {cost}!")

    def perform_search(self):
        start = self.start_entry.get().upper()
        goal = self.goal_entry.get().upper()
        algorithm = self.algorithm_var.get()

        if start not in self.nodes or goal not in self.nodes:
            messagebox.showerror("Error", "Both start and goal nodes must exist!")
            return

        if algorithm not in self.strategies:
            messagebox.showerror("Error", "Please select a valid search strategy!")
            return

        path = self.strategies[algorithm].search(self.nodes[start], self.nodes[goal])

        if path:
            path_str = " → ".join(node.value for node in path)
            self.result_label.config(text=f"Path: {path_str}")
        else:
            self.result_label.config(text="No path found!")

    def visualize_graph(self):
        G = nx.DiGraph()

        for node in self.nodes.values():
            G.add_node(node.value)
            for neighbor, cost in node.get_neighbors_with_cost():
                G.add_edge(node.value, neighbor.value, weight=cost)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()
