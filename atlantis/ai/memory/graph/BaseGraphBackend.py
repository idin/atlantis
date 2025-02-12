from typing import Optional, List, Tuple, Dict, Union
from collections import deque

class BaseGraphBackend:
    """
    Abstract base class for handling graph storage and reasoning.
    This class provides high-level graph operations and delegates storage-specific implementations
    to the subclass via `_` prefixed methods.
    """

    def __init__(self, depth_exceptions: Optional[Dict[str, int]] = None):
        self.depth_exceptions = depth_exceptions or {}

    def _has_node(self, node: str) -> bool:
        """
        Checks if a node exists in the graph.

        Args:
            node (str): The node to check.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        raise NotImplementedError

    def _add_node(self, node: str):
        """
        Adds a node to the graph if it does not already exist.

        Args:
            node (str): The node to add.
        """
        raise NotImplementedError
    
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
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    def _list_of_edges(self) -> List[Dict[str, any]]:
        """
        Returns a list of all edges in the graph.
        """
        raise NotImplementedError
    
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
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    def _has_edge(self, source: str, target: str) -> bool:
        """
        Checks if any edge exists between two nodes.

        Args:
            source (str): The starting node.
            target (str): The ending node.

        Returns:
            bool: True if at least one edge exists between the nodes, False otherwise.
        """
        raise NotImplementedError
    
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
        raise NotImplementedError

    def _remove_node(self, node: str):
        """
        Removes a node from the graph along with all its connected edges.

        Args:
            node (str): The node to remove.

        Example:
            _remove_node("Alice")  # Removes Alice and all edges connected to Alice.
        """
        raise NotImplementedError

    def _clear_graph(self):
        """
        Removes all nodes and edges from the graph.

        Example:
            _clear_graph()  # Completely resets the graph.
        """
        raise NotImplementedError
    
    def _format_output(self, source: str, target: str, data: Dict, output_format: Tuple[str, ...], depth: Optional[int] = None) -> Tuple:
        """
        Formats the output tuple based on the requested output format.

        Args:
            source (str): The source node.
            target (str): The target (neighbour) node.
            data (Dict): Metadata associated with the edge.
            output_format (Tuple[str, ...]): The requested output format.

        Returns:
            Tuple: A tuple containing requested fields.
        """
        if isinstance(output_format, str):
            return_string = True
            output_format = (output_format,)
        else:
            return_string = False

        metadata = dict(data)
        metadata.setdefault("edge", data.get("edge", "unknown"))

        fields = {
            'source': source,
            'edge': metadata['edge'],
            'target': target,
            'metadata': metadata,
            'depth': depth
        }

        output_result = []
        for field in output_format:
            if field in fields:
                output_result.append(fields[field])
            elif field in metadata:
                output_result.append(metadata[field])
            else:
                raise ValueError(f"Invalid output format field: {field}")

        if return_string:
            return output_result[0]
        else:
            return tuple(output_result)
        
    def _n_edges(self) -> int:
        """
        Returns the number of edges in the graph.
        """
        raise NotImplementedError
    
    def n_edges(self) -> int:
        """
        Returns the number of edges in the graph.
        """
        return self._n_edges()

    def __len__(self) -> int:
        return self._n_edges()
    
    def _copy(self, **kwargs) -> 'BaseGraphBackend':
        """
        Returns a copy of the graph.
        """
        raise NotImplementedError

