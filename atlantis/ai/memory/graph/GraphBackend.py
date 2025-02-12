from .BaseGraphBackend import BaseGraphBackend
from typing import Optional, List, Tuple, Dict, Union
from collections import deque

class GraphBackend(BaseGraphBackend):
    """
    The public API for the graph backend that uses the abstract base class and provides the standard methods that are not supposed to be changed by the subclasses.
    """
    def add_edge(self, source: str, edge: str, target: str, **metadata):
        """
        Adds a directed edge between two nodes.

        Args:
            source (str): The starting node.
            edge (str): The edge type.
            target (str): The ending node.
            **metadata: Additional metadata to associate with the edge.
        """
        if not self._has_node(source):
            self._add_node(source)
        if not self._has_node(target):
            self._add_node(target)
        self._add_edge(source, target, edge, **metadata)

    def get_edges(self, source: str, target: str) -> Dict[str, Dict]:
        """
        Retrieves all edges between two nodes.


        Args:
            source (str): The starting node.
            target (str): The ending node.

        Returns:
            List[Tuple[str, Dict]]: A list of (edge, metadata).

        Example Output:
            {
                "knows": {"confidence": 0.9},
                "colleague_of": {"since": "2021"}
            }
        """
        return self._get_all_edges_between_two_nodes(source, target)

    def remove_edge(self, source: str, target: str, edge: Optional[str] = None):
        """
        Removes an edge between two nodes. If no specific edge type is provided, all edges between
        the nodes are removed.

        Args:
            source (str): The starting node.
            target (str): The ending node.
            edge (Optional[str]): The specific edge type to remove. If None, all edges are removed.
        """
        if self._has_edge(source, target):
            self._remove_edge(source, target, edge)

    def remove_node(self, node: str):
        """
        Removes a node and all its edges.

        Args:
            node (str): The node to remove.
        """
        # Retrieve all edges before modifying the graph
        out_edges = [(node, target, edge) for _, target, edge in self._out_edges(node)]
        in_edges = [(source, node, edge) for source, _, edge in self._in_edges(node)]
        
        # Remove all edges
        for source, target, edge in out_edges:
            self._remove_edge(source, target, edge)
        for source, target, edge in in_edges:
            self._remove_edge(source, target, edge)

        # Remove the node itself
        self._remove_node(node)

    def clear(self):
        """
        Clears all nodes and edges from the graph.

        Example:
            clear()  # Completely resets the graph.
        """
        self._clear_graph()

    def get_neighbours(
        self, node: str, depth: int = 1, direction: str = "outgoing",
        output_format: Optional[Union[str, Tuple[str, ...]]] = ('source', 'target', 'edge'),
        return_as_dict: bool = False
    ) -> List[Tuple]:
        """
        Retrieves all neighbors of a node up to a certain depth using BFS.

        Args:
            node (str): The starting node.
            depth (int): The maximum depth to search.
            direction (str): 'outgoing', 'incoming', or 'both'.
            output_format (tuple): Output format options:
                - ('target', 'edge') (default)
                - ('source', 'edge', 'target')
                - ('source', 'edge', 'target', 'metadata')

        Returns:
            List[Tuple]: List of tuples containing requested fields.
        """
        if not self._has_node(node):
            return []

        queue = deque([(node, 0)])
        visited = {node}
        neighbours = {}

        while queue:
            current_node, _current_depth = queue.popleft()
            if _current_depth >= depth:
                continue

            next_nodes = []
            if direction in ["outgoing", "both"]:
                next_nodes.extend(self._out_edges(current_node))

            if direction in ["incoming", "both"]:
                next_nodes.extend(self._in_edges(current_node))

            for source, target, data in next_nodes:
                edge = data.get("edge", "unknown")
                depth_increment = self.depth_exceptions.get(edge, 1)
                next_depth = _current_depth + depth_increment

                # Determine the correct neighbor based on direction
                if direction == 'incoming':
                    neighbour_nodes = [source]
                elif direction == 'outgoing':
                    neighbour_nodes = [target]
                else:
                    neighbour_nodes = [source, target]

                # Ensure correct tuple ordering
                formatted_entry = self._format_output(source=source, target=target, data=data, output_format=output_format, depth=next_depth)
                key = (source, edge, target)

                if key not in neighbours:
                    neighbours[key] = (next_depth, formatted_entry)
                else:
                    _next_depth, _ = neighbours[key]
                    if next_depth < _next_depth:
                        neighbours[key] = (next_depth, formatted_entry)

                # Mark neighbor as visited before enqueuing to avoid duplicate processing
                for neighbour in neighbour_nodes:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append((neighbour, next_depth))

        if return_as_dict:
            return {key: formatted_entry for key, (_, formatted_entry) in neighbours.items()}
        return [formatted_entry for depth, formatted_entry in neighbours.values()]

    def get_list_of_edges(self) -> List[Dict[str, any]]:
        """
        Returns a list of all edges in the graph in the form of {'source': source, 'edge': edge, 'target': target, **metadata}
        """
        return self._list_of_edges()

    def copy(self, **kwargs) -> 'GraphBackend':
        """
        Returns a copy of the graph.
        """
        return self._copy(**kwargs)

