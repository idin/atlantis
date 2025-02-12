from typing import List, Tuple, Dict, Optional, Union
from .GraphBackend import GraphBackend

class MultiMetadata:
    def __init__(self, metadata_dict: Optional[Dict[str, Dict[str, any]]] = None):
        self.metadata_dict = metadata_dict or {}
        self._last_metadata = None

    def add_metadata(self, key: str, metadata: Dict[str, any]):
        self.metadata_dict[key] = metadata
        self._last_metadata = metadata

    def __delitem__(self, key: str):
        if len(self.metadata_dict) == 1 and key in self._last_metadata:
            del self._last_metadata[key]
        else:
            del self.metadata_dict[key]

    def __getitem__(self, key: Union[str, tuple]):
        if isinstance(key, tuple):
            return self.metadata_dict[key[0]][key[1]]

        if key in self.metadata_dict:
            return self.metadata_dict[key]
        if len(self.metadata_dict) == 1:
            return self._last_metadata[key]
        if len(self.metadata_dict) > 1:
            raise KeyError(f'More than one metadata found. Please specify the key.')
        else:
            raise KeyError(f"No metadata found for key {key}. Available keys: {self.metadata_dict.keys()}")
        
    def __setitem__(self, key: Union[str, tuple], value: any):
        if isinstance(key, tuple):
            self.metadata_dict[key[0]][key[1]] = value
        else:
            self.metadata_dict[key] = value 

    def __contains__(self, key: str):
        if len(self.metadata_dict) == 1:
            if key in self._last_metadata:
                return True
        return key in self.metadata_dict

class SimpleGraphBackend(GraphBackend):
    """
    A simple in-memory graph backend using dictionaries for node and edge storage.
    Supports multiple edges between the same pair of nodes with O(1) lookups.
    """

    def __init__(self, depth_exceptions: Optional[Dict[str, int]] = None):
        """
        Initializes the SimpleGraphBackend with empty node and edge dictionaries.
        Args:
            depth_exceptions (Dict[str, int]): A dictionary of edge types and their corresponding depth exceptions.
            if an edge type is not in the dictionary, the default depth exception is 1.
            if an edge type is in the dictionary, the depth exception is the value of the dictionary for that edge type.
            For example if {'is_subject_of': 0} is in the dictionary, the depth exception for 'is_subject_of' is 0 which means traversing through this edge does not count as depth
        """
        self._nodes: Dict[str, Dict] = {}  # Stores node properties (if any)
        self._edges: Dict[Tuple[str, str], Dict[str, Dict]] = {}  # Stores edges as {(source, target): {edge: metadata}}
        super().__init__(depth_exceptions=depth_exceptions)

    def _has_node(self, node: str) -> bool:
        """
        Checks if a node exists in the graph.

        Args:
            node (str): The node to check.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        return node in self._nodes

    def _add_node(self, node: str):
        """
        Adds a node to the graph if it does not already exist.

        Args:
            node (str): The node to add.
        """
        if node not in self._nodes:
            self._nodes[node] = {}

    def _add_edge(self, source: str, target: str, edge: str, metadata_key: Optional[str] = None, **metadata):
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
        if (source, target) not in self._edges:
            self._edges[(source, target)] = {}
        metadata["edge"] = edge  # Ensure "edge" is stored explicitly
        if metadata_key:
            # if this edge already has a metadata key, we need to merge the metadata
            if edge in self._edges[(source, target)]:
                prev_metadata = self._edges[(source, target)][edge]
                if isinstance(prev_metadata, dict)
        self._edges[(source, target)][edge] = metadata


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
        return self._edges.get((source, target), {})
    
    def _list_of_edges(self) -> List[Dict[str, any]]:
        """
        Returns a list of all edges in the graph in the form of {'source': source, 'edge': edge, 'target': target, **metadata}
        """
        return [
            {
                "source": source,
                "edge": edge,
                "target": target,
                **{key: value for key, value in metadata.items() if key not in {'source', 'target', 'edge'}}
            }
            for ((source, target), edge_dict) in self._edges.items()    
            for edge, metadata in edge_dict.items()
        ]

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
        return [(source, target, {"edge": edge, **metadata}) 
                for (source, target), edges in self._edges.items() if source == node 
                for edge, metadata in edges.items()]

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
        return [
            (source, target, {"edge": edge, **metadata})  
            for (source, target), edges in self._edges.items()
            if target == node and source != node  # Prevent self-referencing edges
            for edge, metadata in edges.items()
        ]

    def _has_edge(self, source: str, target: str) -> bool:
        """
        Checks if any edge exists between two nodes.

        Args:
            source (str): The starting node.
            target (str): The ending node.

        Returns:
            bool: True if at least one edge exists between the nodes, False otherwise.
        """
        return (source, target) in self._edges and bool(self._edges[(source, target)])

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
        if (source, target) in self._edges:
            if edge:
                self._edges[(source, target)].pop(edge, None)
                if not self._edges[(source, target)]:  # Remove entry if no edges remain
                    del self._edges[(source, target)]
            else:
                del self._edges[(source, target)]  # Remove all edges


    def _remove_node(self, node: str):
        """
        Removes a node from the graph along with all its connected edges.

        Args:
            node (str): The node to remove.

        Example:
            _remove_node("Alice")  # Removes Alice and all edges connected to Alice.
        """
        if node in self._nodes:
            del self._nodes[node]

        # Remove all edges where this node is either source or target
        self._edges = {(s, t): edges for (s, t), edges in self._edges.items() if s != node and t != node}

    def _clear_graph(self):
        """
        Removes all nodes and edges from the graph.

        Example:
            _clear_graph()  # Completely resets the graph.
        """
        self._nodes.clear()
        self._edges.clear()

    def _n_edges(self) -> int:
        """
        Returns the number of edges in the graph.
        """
        return len(self._edges)

    def _copy(self, depth_exceptions: Optional[Dict[str, int]] = None) -> 'SimpleGraphBackend':
        """
        Returns a copy of the graph.
        """
        new_graph_backend = SimpleGraphBackend(depth_exceptions=depth_exceptions or self.depth_exceptions.copy())
        new_graph_backend._nodes = self._nodes.copy()
        new_graph_backend._edges = self._edges.copy()
        return new_graph_backend

