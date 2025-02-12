import networkx as nx
from typing import List, Tuple, Dict, Optional
from .GraphBackend import GraphBackend

class NetworkXGraphBackend(GraphBackend):
    """
    NetworkX-based implementation of GraphBackend that supports multiple edges (relationships) between nodes.
    Uses a directed multigraph (MultiDiGraph), allowing multiple edges between the same node pairs.
    """

    def __init__(self, depth_exceptions: Optional[Dict[str, int]] = None):
        """
        Initializes the NetworkX-based graph backend.

        Args:
            depth_exceptions (Optional[Dict[str, int]]): A dictionary specifying depth exceptions 
                for certain edges, where the traversal cost is different from the default.
        """
        self.graph = nx.MultiDiGraph()
        self.depth_exceptions = depth_exceptions or {}

    def _has_node(self, node: str) -> bool:
        """
        Checks if a node exists in the graph.

        Args:
            node (str): The node to check.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        return self.graph.has_node(node)

    def _add_node(self, node: str):
        """
        Adds a node to the graph if it does not already exist.

        Args:
            node (str): The node to add.
        """
        self.graph.add_node(node)

    def _add_edge(self, source: str, target: str, edge: str, **metadata):
        """
        Adds an edge between two nodes with optional metadata.

        Args:
            source (str): The starting node of the edge.
            target (str): The ending node of the edge.
            edge (str): The label identifying the edge type.
            **metadata: Additional metadata to store with the edge.

        Example:
            _add_edge("Alice", "Bob", "knows", confidence=0.9)
        """
        if self.graph.has_edge(source, target, key=edge):
            # Ensure complete replacement of metadata
            self.graph[source][target][edge].clear()  # Remove old metadata
            self.graph[source][target][edge].update({"edge": edge, **metadata})  # Add new metadata
        else:
            self.graph.add_edge(source, target, key=edge, edge=edge, **metadata)

    def _get_all_edges_between_two_nodes(self, source: str, target: str) -> Dict[str, Dict]:
        """
        Retrieves metadata for all edges between two nodes.

        Args:
            source (str): The starting node of the edges.
            target (str): The ending node of the edges.

        Returns:
            Dict[str, Dict]: A dictionary where keys are edge labels, 
            and values are metadata dictionaries.

        Example Output:
            {
                "knows": {"confidence": 0.9},
                "colleague_of": {"since": "2021"}
            }
        """
        return self.graph.get_edge_data(source, target, default={}) or {}

    def _out_edges(self, node: str) -> List[Tuple[str, str, Dict]]:
        """
        Retrieves all outgoing edges from a node.

        Args:
            node (str): The node to retrieve outgoing edges from.

        Returns:
            List[Tuple[str, str, Dict]]: A list of tuples containing (source, target, metadata).

        Example Output:
            [
                ("Alice", "Bob", {"edge": "knows", "confidence": 0.9}),
                ("Alice", "Charlie", {"edge": "friend_of", "since": "2022"})
            ]
        """
        return list(self.graph.out_edges(node, data=True))

    def _in_edges(self, node: str) -> List[Tuple[str, str, Dict]]:
        """
        Retrieves all incoming edges to a node.

        Args:
            node (str): The node to retrieve incoming edges for.

        Returns:
            List[Tuple[str, str, Dict]]: A list of tuples containing (source, target, metadata).

        Example Output:
            [
                ("Bob", "Alice", {"edge": "knows", "confidence": 0.9}),
                ("Charlie", "Alice", {"edge": "friend_of", "since": "2022"})
            ]
        """
        return list(self.graph.in_edges(node, data=True))

    def _has_edge(self, source: str, target: str) -> bool:
        """
        Checks if any edge exists between two nodes.

        Args:
            source (str): The starting node.
            target (str): The ending node.

        Returns:
            bool: True if at least one edge exists between the nodes, False otherwise.
        """
        return self.graph.has_edge(source, target)

    def _remove_edge(self, source: str, target: str, edge: Optional[str] = None):
        """
        Removes an edge between two nodes. If no specific edge label is provided, all edges between
        the nodes are removed.

        Args:
            source (str): The starting node.
            target (str): The ending node.
            edge (Optional[str]): The specific edge label to remove. If None, all edges are removed.

        Example:
            _remove_edge("Alice", "Bob", "knows")  # Removes only the "knows" edge.
            _remove_edge("Alice", "Bob")  # Removes all edges between Alice and Bob.
        """
        if self.graph.has_edge(source, target):
            if edge:
                if self.graph.has_edge(source, target, key=edge):  # Prevent NetworkXError
                    self.graph.remove_edge(source, target, key=edge)
            else:
                # Get a list of edge keys before removing to avoid KeyError
                edge_keys = list(self.graph[source][target].keys())
                for edge_key in edge_keys:
                    if self.graph.has_edge(source, target, key=edge_key):  # Prevent KeyError
                        self.graph.remove_edge(source, target, key=edge_key)

    def _remove_node(self, node: str):
        """
        Removes a node from the graph along with all its connected edges.

        Args:
            node (str): The node to remove.

        Example:
            _remove_node("Alice")  # Removes Alice and all edges connected to Alice.
        """
        if self.graph.has_node(node):
            self.graph.remove_node(node)

    def _clear_graph(self):
        """
        Removes all nodes and edges from the graph.

        Example:
            _clear_graph()  # Completely resets the graph.
        """
        self.graph.clear()

    def _n_edges(self) -> int:
        """
        Returns the number of edges in the graph.
        """
        return self.graph.number_of_edges()

    def _list_of_edges(self) -> List[Dict[str, any]]:
        """
        Returns a list of all edges in the graph in the form of {'source': source, 'edge': edge, 'target': target, **metadata}
        """
        return [
            {
                "source": source,
                "edge": metadata.get("edge", edge_key),  # Ensure edge label is included
                "target": target,
                **{key: value for key, value in metadata.items() if key not in {'source', 'target', 'edge'}}
            }
            for source, target, edge_key, metadata in self.graph.edges(keys=True, data=True)
        ]

    def _copy(self, depth_exceptions: Optional[Dict[str, int]] = None) -> 'NetworkXGraphBackend':
        """
        Returns a copy of the graph.
        """
        new_graph_backend = NetworkXGraphBackend(depth_exceptions=depth_exceptions or self.depth_exceptions.copy())
        new_graph_backend.graph = self.graph.copy()
        return new_graph_backend
