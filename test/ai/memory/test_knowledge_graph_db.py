import pytest
from atlantis.ai.memory.db.DuckDBGraphDB import DuckDBGraphDB
from atlantis.ai.memory.graph.NetworkXGraphBackend import NetworkXGraphBackend
from atlantis.ai.memory import KnowledgeGraphDB


@pytest.fixture
def kg():
    """Fixture to create a fresh in-memory knowledge graph for each test."""
    return KnowledgeGraphDB(
        database=DuckDBGraphDB(),
        graph_backend=NetworkXGraphBackend()
    )


def test_knowledge_graph_db(kg):
    """Basic tests for adding, retrieving, deleting, and clearing facts."""

    # Add facts
    kg.add_fact(subject="Paris", predicate="capital_of", obj="France")
    kg.add_fact(subject="France", predicate="located_in", obj="Europe")
    kg.add_fact(subject="Einstein", predicate="discovered", obj="Theory of Relativity")

    # Test direct retrieval
    paris_facts = kg.get_facts_about(subject="Paris", predicate="capital_of", include_source=True, include_timestamp=True)
    assert [(s, p, o) for s, p, o, _, _ in paris_facts] == [("Paris", "capital_of", "France")]

    einstein_facts = kg.get_facts_about(subject="Einstein", predicate="discovered", include_source=True, include_timestamp=True)
    assert [(s, p, o) for s, p, o, _, _ in einstein_facts] == [("Einstein", "discovered", "Theory of Relativity")]

    assert kg.get_facts_about(subject="France", predicate="located_in") == [("France", "located_in", "Europe")]

    # Test case-insensitive retrieval
    assert kg.get_facts_about(subject="paris", predicate="capital_of") == [("Paris", "capital_of", "France")]
    assert kg.get_facts_about(subject="einstein", predicate="discovered") == [("Einstein", "discovered", "Theory of Relativity")]

    # Test getting all facts about a subject
    facts = kg.get_facts_about(subject="Paris")
    assert [(p, o) for _, p, o in facts] == [("capital_of", "France")]

    facts = kg.get_facts_about(subject="France")
    assert [(p, o) for _, p, o in facts] == [("located_in", "Europe")]

    # Test multi-hop reasoning
    inferred = kg.get_facts_about(subject="Paris", depth=2)
    inferred_objects = {fact[2] for fact in inferred}
    assert "Europe" in inferred_objects

    # Test deleting a fact
    kg.remove_fact(fact_id=kg._generate_fact_id(subject="Einstein", predicate="discovered", obj="Theory of Relativity"))
    assert kg.get_facts_about(subject="Einstein", predicate="discovered") == []

    # Test clearing knowledge
    kg.clear_knowledge()
    assert kg.get_facts_about() == []


def test_knowledge_graph_db_delete_facts_about(kg):
    """Test deleting facts using remove_facts_about()."""

    kg.add_fact(subject="Paris", predicate="capital_of", obj="France")
    kg.add_fact(subject="France", predicate="located_in", obj="Europe")
    kg.add_fact(subject="Paris", predicate="has_landmark", obj="Eiffel Tower")

    # Delete facts only about Paris
    kg.remove_facts_about(subject="Paris")
    assert kg.get_facts_about(subject="Paris") == []
    assert kg.get_facts_about(subject="France", predicate="located_in") == [("France", "located_in", "Europe")]


def test_knowledge_graph_db_get_facts_about(kg):
    """Test retrieving facts by subject, predicate, and object."""

    kg.add_fact(subject="Paris", predicate="capital_of", obj="France")
    kg.add_fact(subject="France", predicate="located_in", obj="Europe")
    kg.add_fact(subject="Europe", predicate="is_a", obj="Continent")

    # Retrieve by subject
    facts = kg.get_facts_about(subject="Paris")
    assert [(p, o) for _, p, o in facts] == [("capital_of", "France")]

    # Retrieve by object
    facts = kg.get_facts_about(obj="France")
    assert [(s, p) for s, p, _ in facts] == [("Paris", "capital_of")]

    # Retrieve by predicate
    facts = kg.get_facts_about(predicate="capital_of")
    assert [(s, p, o) for s, p, o in facts] == [("Paris", "capital_of", "France")]

    # Mixed retrieval
    facts = kg.get_facts_about(subject="Paris", obj="France")
    assert [(s, p, o) for s, p, o in facts] == [("Paris", "capital_of", "France")]


def test_knowledge_graph_retrieve_facts_about_bidirectional_depth_2(kg):
    """Test bidirectional retrieval with depth=2, ensuring sibling traversal."""

    kg.add_fact(subject="Paris", predicate="capital_of", obj="France")
    kg.add_fact(subject="Berlin", predicate="capital_of", obj="Germany")
    kg.add_fact(subject="France", predicate="located_in", obj="Europe")
    kg.add_fact(subject="Germany", predicate="located_in", obj="Europe")
    kg.add_fact(subject="Europe", predicate="is_a", obj="Continent")

    results = kg.get_facts_about(subject="Paris", depth=2, bidirectional=True)
    assert ("Europe", "is_a", "Continent") in results


def test_include_source_and_timestamp(kg):
    """Test including source and timestamp in retrieval."""

    kg.add_fact(subject="Paris", predicate="capital_of", obj="France", source="historical_data")
    kg.add_fact(subject="France", predicate="located_in", obj="Europe", source="geographical_data")

    facts = kg.get_facts_about(subject="Paris", include_source=True, include_timestamp=True)
    assert all(len(fact) == 5 for fact in facts)

    facts = kg.get_facts_about(subject="Paris", include_source=True, include_timestamp=False)
    assert all(len(fact) == 4 for fact in facts)

    facts = kg.get_facts_about(subject="Paris", include_source=False, include_timestamp=True)
    assert all(len(fact) == 4 for fact in facts)

    facts = kg.get_facts_about(subject="Paris", include_source=False, include_timestamp=False)
    assert all(len(fact) == 3 for fact in facts)

    facts = kg.get_facts_about(subject="Paris")
    assert all(len(fact) == 3 for fact in facts)


if __name__ == "__main__":
    pytest.main()
