import pytest
from atlantis.ai.memory.graph import SimpleGraphBackend

@pytest.fixture
def graph():
    """Fixture to create a fresh SimpleGraphBackend instance for each test."""
    return SimpleGraphBackend()

@pytest.fixture
def graph_with_depth_exception():
    """Fixture to create a fresh SimpleGraphBackend instance for each test."""
    return SimpleGraphBackend(depth_exceptions={'is_subject_of': 0, 'is_object_of': 0})


def test_simple_graph_backend_has_all_methods():
    """Ensure SimpleGraphBackend implements all required methods from GraphBackend."""
    required_methods = [
        "add_edge", "get_edges", "remove_edge", "remove_node", "clear",
        "get_neighbours", "_format_output", "_has_node", "_add_node",
        "_add_edge", "_get_all_edges_between_two_nodes", "_out_edges", "_in_edges",
        "_has_edge", "_remove_edge", "_remove_node", "_clear_graph"
    ]

    missing_methods = [method for method in required_methods if not hasattr(SimpleGraphBackend, method)]
    
    assert not missing_methods, f"SimpleGraphBackend is missing methods: {missing_methods}"


def test_add_edge_and_get_relationships(graph):
    """Test adding edges and retrieving relationships between nodes."""
    graph.add_edge(source="Paris", target="France", edge="capital_of")
    graph.add_edge(source="Paris", target="France", edge="famous_for", description="Eiffel Tower")

    relationships = graph.get_edges(source="Paris", target="France")

    assert isinstance(relationships, dict)
    assert len(relationships) == 2  # Ensure both edges exist

    assert "capital_of" in relationships
    assert relationships["capital_of"] == {"edge": "capital_of"}

    assert "famous_for" in relationships
    assert relationships["famous_for"] == {"edge": "famous_for", "description": "Eiffel Tower"}


def test_remove_edge(graph):
    """Test removing specific edges."""
    graph.add_edge(source="Paris", target="France", edge="capital_of")
    graph.add_edge(source="Paris", target="France", edge="famous_for", description="Eiffel Tower")

    # Remove only the "capital_of" edge
    graph.remove_edge(source="Paris", edge="capital_of", target="France")
    relationships = graph.get_edges(source="Paris", target="France")

    assert isinstance(relationships, dict)
    assert len(relationships) == 1  # Only "famous_for" should remain

    assert "capital_of" not in relationships  # Ensure it was removed
    assert "famous_for" in relationships

    assert relationships["famous_for"] == {"edge": "famous_for", "description": "Eiffel Tower"}


def test_remove_all_edges_between_nodes(graph):
    """Test removing all edges between two nodes."""
    graph.add_edge(source="Paris", target="France", edge="capital_of")
    graph.add_edge(source="Paris", target="France", edge="famous_for", description="Eiffel Tower")

    # Remove all edges between "Paris" and "France"
    graph.remove_edge(source="Paris", target="France")
    relationships = graph.get_edges(source="Paris", target="France")

    assert isinstance(relationships, dict)
    assert len(relationships) == 0  # Should be empty after removal


def test_get_edges_from_nonexistent_nodes(graph):
    """Test retrieving edges when no edges exist between nodes."""
    relationships = graph.get_edges(source="Paris", target="France")

    assert isinstance(relationships, dict)
    assert len(relationships) == 0  # Should return an empty dictionary


def test_add_duplicate_edge_with_different_metadata(graph):
    """Test adding an edge with the same type but different metadata and retrieving it."""
    graph.add_edge(source="Paris", target="France", edge="famous_for", description="Eiffel Tower")
    graph.add_edge(source="Paris", target="France", edge="famous_for", year="1889")

    relationships = graph.get_edges(source="Paris", target="France")

    assert "famous_for" in relationships
    assert relationships["famous_for"] == {"edge": "famous_for", "year": "1889"}  # Latest metadata should overwrite

def test_get_neighbours_outgoing(graph):
    """Test retrieving outgoing neighbours from a node."""
    graph.add_edge("Alice", "knows", "Bob")
    graph.add_edge("Alice", "works_with", "Charlie")
    graph.add_edge("Charlie", "friend_of", "Alice")

    neighbours = graph.get_neighbours("Alice", direction="outgoing", output_format=('source', 'edge', 'target'))
    edge1 = ("Alice", "knows", "Bob")
    edge2 = ("Alice", "works_with", "Charlie")
    edge3 = ("Charlie", "friend_of", "Alice")

    assert len(neighbours) == 2
    assert edge1 in neighbours
    assert edge2 in neighbours
    assert edge3 not in neighbours

    neighbours = graph.get_neighbours("Alice", direction="outgoing", output_format=('target',))
    assert len(neighbours) == 2
    assert ("Bob",) in neighbours
    assert ("Charlie",) in neighbours


def test_get_neighbours_incoming(graph):
    """Test retrieving incoming neighbours from a node."""
    graph.add_edge("Alice", "knows", "Bob")
    graph.add_edge("Charlie", "friend_of", "Alice")

    neighbours = graph.get_neighbours("Alice", direction="incoming", output_format=('source', 'edge', 'target'))
    edge1 = ("Charlie", "friend_of", "Alice")
    edge2 = ("Alice", "knows", "Bob")


    assert len(neighbours) == 1
    assert edge1 in neighbours
    assert edge2 not in neighbours

def test_get_neighbours_both_directions(graph):
    """Test retrieving neighbours in both directions."""
    graph.add_edge("Alice", "knows", "Bob")
    graph.add_edge("Bob", "colleague_of", "Alice")
    graph.add_edge("Charlie", "friend_of", "Alice")

    # Charlie --friend_of--> Alice --knows--> Bob --colleague_of--> Alice

    neighbours = graph.get_neighbours("Alice", direction="both", output_format=('source', 'edge', 'target'))
    edge1 = ("Charlie", "friend_of", "Alice")
    edge2 = ("Alice", "knows", "Bob")
    edge3 = ("Bob", "colleague_of", "Alice")



    assert isinstance(neighbours, list)
    assert len(neighbours) == 3
    assert edge1 in neighbours
    assert edge2 in neighbours
    assert edge3 in neighbours



def test_get_neighbours_with_depth(graph):
    """Test retrieving neighbours with depth > 1."""
    graph.add_edge("Alice", "knows", "Bob")
    graph.add_edge("Bob", "works_with", "Charlie")
    graph.add_edge("Charlie", "friend_of", "David")

    neighbours = graph.get_neighbours("Alice", depth=2, output_format=('edge', 'target'))

    assert isinstance(neighbours, list)
    assert len(neighbours) == 2
    assert ("knows", "Bob") in neighbours
    assert ("works_with", "Charlie") in neighbours

    neighbours = graph.get_neighbours("Alice", depth=2, output_format='target')
    assert isinstance(neighbours, list)
    assert len(neighbours) == 2
    assert "Bob" in neighbours
    assert "Charlie" in neighbours
    assert "Alice" not in neighbours

    neighbours = graph.get_neighbours("Alice", depth=2, output_format=('source', 'edge', 'target'))
    assert isinstance(neighbours, list)
    assert len(neighbours) == 2
    assert ("Alice", "knows", "Bob") in neighbours
    assert ("Bob", "works_with", "Charlie") in neighbours
    assert ("Charlie", "friend_of", "David") not in neighbours


    neighbours = graph.get_neighbours("Alice", depth=2, output_format='source')
    assert isinstance(neighbours, list)
    assert len(neighbours) == 2
    assert "Alice" in neighbours
    assert "Bob" in neighbours
    assert "Charlie" not in neighbours


def test_get_neighbours_no_connections(graph):
    """Test retrieving neighbours when node has no connections."""
    graph.add_edge("Alice", "knows", "Bob")

    neighbours = graph.get_neighbours("Charlie")

    assert isinstance(neighbours, list)
    assert len(neighbours) == 0  # Charlie has no connections


def test_get_neighbours_empty_graph(graph):
    """Test retrieving neighbours from an empty graph."""
    neighbours = graph.get_neighbours("Alice")

    assert isinstance(neighbours, list)
    assert len(neighbours) == 0  # No nodes exist in the graph


def test_get_neighbours_custom_output_format(graph):
    """Test retrieving neighbours with a custom output format."""
    graph.add_edge("Alice", "knows", "Bob", strength=5)

    neighbours = graph.get_neighbours("Alice", output_format=("source", "edge", "target", "metadata"))

    assert isinstance(neighbours, list)
    assert len(neighbours) == 1
    assert neighbours[0] == ("Alice", "knows", "Bob", {"edge": "knows", "strength": 5})

def test_depth_without_exceptions(graph):
    """Test depth exceptions."""
    graph.add_edge("Alice", "knows", "Bob")
    graph.add_edge("Bob", "is_friend_of", "Charlie")
    graph.add_edge("Alice", "is_subject_of", "Fact1")
    graph.add_edge("Bob", "is_object_of", "Fact1")
    graph.add_edge("Fact1", "is", "Questionable")
    graph.add_edge("Questionable", "is", "Important")
    neighbours_at_depth_1 = graph.get_neighbours("Alice", depth=1, direction='outgoing', output_format=('source', 'edge', 'target'))
    """
    neighbours should be:
        Alice --knows--> Bob
        Alice --is_subject_of--> Fact1
    """
    assert len(neighbours_at_depth_1) == 2
    assert ("Alice", "knows", "Bob") in neighbours_at_depth_1
    assert ("Alice", "is_subject_of", "Fact1") in neighbours_at_depth_1

    neighbours_at_depth_2 = graph.get_neighbours("Alice", depth=2, direction='outgoing', output_format=('source', 'edge', 'target'))
    """
    neighbours_at_depth_2 should be:
        Alice --knows--> Bob (depth = 1)
        Bob --is_friend_of--> Charlie (depth = 2)
        Bob --is_object_of--> Fact1 (depth = 2)
        Alice --is_subject_of--> Fact1 (depth = 1)
        Fact1 --is--> Questionable (depth = 2)

    """
    assert len(neighbours_at_depth_2) == 5
    assert ("Alice", "knows", "Bob") in neighbours_at_depth_2
    assert ("Alice", "is_subject_of", "Fact1") in neighbours_at_depth_2
    assert ("Bob", "is_friend_of", "Charlie") in neighbours_at_depth_2
    assert ("Bob", "is_object_of", "Fact1") in neighbours_at_depth_2
    assert ("Fact1", "is", "Questionable") in neighbours_at_depth_2

def test_depth_with_exceptions(graph_with_depth_exception):
    """Test depth exceptions."""
    graph_with_depth_exception.add_edge("Alice", "knows", "Bob")
    graph_with_depth_exception.add_edge("Bob", "is_friend_of", "Charlie")
    graph_with_depth_exception.add_edge("Alice", "is_subject_of", "Fact1")
    graph_with_depth_exception.add_edge("Bob", "is_object_of", "Fact1")
    graph_with_depth_exception.add_edge("Fact1", "is", "Questionable")

    neighbours_at_depth_1_with_depth_exception = graph_with_depth_exception.get_neighbours("Alice", depth=1, direction='outgoing', output_format=('source', 'edge', 'target', 'depth'))
    """
    neighbours_at_depth_1_with_depth_exception should be:
        Alice --knows--> Bob (depth = 1)
        Alice --is_subject_of--> Fact1 (depth = 0)
        Fact1 --is--> Questionable (depth = 1)
    """
    assert len(neighbours_at_depth_1_with_depth_exception) == 3
    assert ("Alice", "knows", "Bob", 1) in neighbours_at_depth_1_with_depth_exception
    assert ("Alice", "is_subject_of", "Fact1", 0) in neighbours_at_depth_1_with_depth_exception
    assert ("Fact1", "is", "Questionable", 1) in neighbours_at_depth_1_with_depth_exception


    neighbours_at_depth_2_with_depth_exception = graph_with_depth_exception.get_neighbours("Alice", depth=2, direction='outgoing', output_format=('source', 'edge', 'target'))
    """
    neighbours_at_depth_2_with_depth_exception should be:
        Alice --knows--> Bob (depth = 1)
        Bob --is_friend_of--> Charlie (depth = 2)
        Bob --is_object_of--> Fact1 (depth = 1 + 0)
        Alice --is_subject_of--> Fact1 (depth = 1 + 0)
        Fact1 --is--> Questionable (depth = 2)
        Questionable --is--> Important (depth = 3) 
    """
    assert len(neighbours_at_depth_2_with_depth_exception) == 5
    assert ("Alice", "knows", "Bob") in neighbours_at_depth_2_with_depth_exception
    assert ("Bob", "is_friend_of", "Charlie") in neighbours_at_depth_2_with_depth_exception
    assert ("Alice", "is_subject_of", "Fact1") in neighbours_at_depth_2_with_depth_exception
    assert ("Fact1", "is", "Questionable") in neighbours_at_depth_2_with_depth_exception
    assert ("Bob", "is_object_of", "Fact1") in neighbours_at_depth_2_with_depth_exception
    assert ("Questionable", "is", "Important") not in neighbours_at_depth_2_with_depth_exception

def test_depth_without_exception_incoming(graph):
    """Test depth without exception incoming."""
    graph.add_edge("Alice", "knows", "Bob")
    graph.add_edge("Bob", "is_friend_of", "Charlie")
    graph.add_edge("Alice", "is_subject_of", "Fact1")
    graph.add_edge("Bob", "is_object_of", "Fact1")
    graph.add_edge("Fact1", "is", "Questionable")

    neighbours = graph.get_neighbours("Alice", depth=1, direction='incoming', output_format=('source', 'edge', 'target'))
    assert len(neighbours) == 0

    neighbours = graph.get_neighbours("Bob", depth=1, direction='incoming', output_format=('source', 'edge', 'target'))
    assert len(neighbours) == 1
    assert ("Alice", "knows", "Bob") in neighbours

    neighbours = graph.get_neighbours("Charlie", depth=1, direction='incoming', output_format=('source', 'edge', 'target'))
    assert len(neighbours) == 1
    assert ("Bob", "is_friend_of", "Charlie") in neighbours

    neighbours = graph.get_neighbours("Charlie", depth=2, direction='incoming', output_format=('source', 'edge', 'target'))
    assert len(neighbours) == 2
    assert ("Bob", "is_friend_of", "Charlie") in neighbours
    assert ("Alice", "knows", "Bob") in neighbours

    neighbours = graph.get_neighbours("Fact1", depth=2, direction='incoming', output_format=('source', 'edge', 'target'))
    assert len(neighbours) == 3
    assert ("Alice", "is_subject_of", "Fact1") in neighbours
    assert ("Bob", "is_object_of", "Fact1") in neighbours
    assert ("Alice", "knows", "Bob") in neighbours

    neighbours = graph.get_neighbours("Questionable", depth=3, direction='incoming', output_format=('source', 'edge', 'target', 'depth'))
    assert len(neighbours) == 4
    assert ("Fact1", "is", "Questionable", 1) in neighbours # Depth 1
    assert ("Alice", "is_subject_of", "Fact1", 2) in neighbours # Depth 2
    assert ("Bob", "is_object_of", "Fact1", 2) in neighbours # Depth 2
    assert ("Alice", "knows", "Bob", 3) in neighbours # Depth 3

def test_depth_with_exception_incoming(graph_with_depth_exception):
    """Test depth with exception incoming."""

    graph_with_depth_exception.add_edge("Alice", "knows", "Bob")
    graph_with_depth_exception.add_edge("Bob", "is_friend_of", "Charlie")
    graph_with_depth_exception.add_edge("Alice", "is_subject_of", "Fact1")
    graph_with_depth_exception.add_edge("Bob", "is_object_of", "Fact1")
    graph_with_depth_exception.add_edge("Fact1", "is", "Questionable")

    neighbours = graph_with_depth_exception.get_neighbours("Alice", depth=1, direction='incoming', output_format=('source', 'edge', 'target'))
    assert len(neighbours) == 0

    neighbours = graph_with_depth_exception.get_neighbours("Bob", depth=1, direction='incoming', output_format=('source', 'edge', 'target'))
    assert len(neighbours) == 1

    neighbours = graph_with_depth_exception.get_neighbours("Charlie", depth=1, direction='incoming', output_format=('source', 'edge', 'target'))
    assert len(neighbours) == 1
    assert ("Bob", "is_friend_of", "Charlie") in neighbours

    neighbours = graph_with_depth_exception.get_neighbours("Fact1", depth=2, direction='incoming', output_format=('source', 'edge', 'target', 'depth'))
    assert len(neighbours) == 3
    assert ("Alice", "is_subject_of", "Fact1", 0) in neighbours
    assert ("Bob", "is_object_of", "Fact1", 0) in neighbours
    assert ("Alice", "knows", "Bob", 1) in neighbours


    neighbours = graph_with_depth_exception.get_neighbours("Questionable", depth=2, direction='incoming', output_format=('source', 'edge', 'target', 'depth'))
    #assert len(neighbours) == 4
    assert ("Fact1", "is", "Questionable", 1) in neighbours # depth 1
    assert ("Alice", "is_subject_of", "Fact1", 1) in neighbours # depth 1
    assert ("Bob", "is_object_of", "Fact1", 1) in neighbours # depth 1
    assert ("Alice", "knows", "Bob", 2) in neighbours # depth 2

def test_depth_with_exception_bidirectional(graph_with_depth_exception):
    """Test depth with exception bidirectional."""
    graph_with_depth_exception.add_edge("Alice", "knows", "Bob")
    graph_with_depth_exception.add_edge("Bob", "is_friend_of", "Charlie")
    graph_with_depth_exception.add_edge("Alice", "is_subject_of", "Fact1")
    graph_with_depth_exception.add_edge("Bob", "is_object_of", "Fact1")
    graph_with_depth_exception.add_edge("Fact1", "is", "Questionable")

    neighbours = graph_with_depth_exception.get_neighbours("Bob", depth=1, direction='both', output_format=('source', 'edge', 'target', 'depth'))
    assert len(neighbours) == 5
    assert ("Alice", "knows", "Bob", 1) in neighbours
    assert ("Bob", "is_friend_of", "Charlie", 1) in neighbours
    assert ("Bob", "is_object_of", "Fact1", 0) in neighbours
    assert ("Alice", "is_subject_of", "Fact1", 0) in neighbours
    assert ("Fact1", "is", "Questionable", 1) in neighbours

    neighbours = graph_with_depth_exception.get_neighbours("Bob", depth=2, direction='both', output_format=('source', 'edge', 'target', 'depth'))
    assert len(neighbours) == 5
    assert ("Alice", "knows", "Bob", 1) in neighbours
    assert ("Bob", "is_friend_of", "Charlie", 1) in neighbours
    assert ("Bob", "is_object_of", "Fact1", 0) in neighbours
    assert ("Alice", "is_subject_of", "Fact1", 0) in neighbours
    assert ("Fact1", "is", "Questionable", 1) in neighbours
    
