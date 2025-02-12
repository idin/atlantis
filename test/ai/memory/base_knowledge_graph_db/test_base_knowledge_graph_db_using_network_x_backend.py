import pytest
from datetime import datetime, timedelta
import time
try:
    from datetime import UTC
except ImportError:
    from datetime import timezone

    UTC = timezone.utc

from atlantis.ai.memory.db import DuckDBGraphDB
from atlantis.ai.memory.graph import NetworkXGraphBackend
from atlantis.ai.memory import KnowledgeGraphDB

@pytest.fixture
def knowledge_graph():
    graph_db = DuckDBGraphDB()
    graph_backend = NetworkXGraphBackend()
    return KnowledgeGraphDB(graph_db, graph_backend)


def test_add_fact_to_db(knowledge_graph):
    fact_id, original_timestamp = knowledge_graph._add_fact_to_db("Earth", "is_a", "Planet", source="NASA")
    
    assert knowledge_graph.database.get_number_of_rows("knowledge") == 1
    fact = knowledge_graph.database.get_one("knowledge", include_fields=["subject", "predicate", "object", "source", "timestamp"])
    subject, predicate, object, source, timestamp = fact
    assert subject == "Earth"
    assert predicate == "is_a"
    assert object == "Planet"
    assert source == "NASA"
    assert isinstance(timestamp, datetime)
    assert timestamp.timestamp() == original_timestamp.timestamp()


def test_add_fact_to_graph(knowledge_graph):
    timestamp = knowledge_graph._add_fact_to_graph("Alice", "knows", "Bob", fact_id='f1', source="SocialNetwork", timestamp=datetime.now())

    assert len(knowledge_graph.graph_backend.get_list_of_edges()) == 1
    edge = knowledge_graph.graph_backend.get_list_of_edges()[0]
    assert edge == {'source': "Alice", 'target': "Bob", 'edge': "knows", 'fact_id': 'f1', 'fact_source': "SocialNetwork", 'timestamp': timestamp}


def test_add_fact_with_relationship_predicate(knowledge_graph):
    timestamp = knowledge_graph.add_fact("Mars", "is_a", "Planet", source="NASA", add_to_graph=None)

    assert len(knowledge_graph) == 1
    assert len(knowledge_graph) == 1  # "is_a" should be added to graph


def test_add_fact_without_relationship_predicate(knowledge_graph):
    timestamp = knowledge_graph.add_fact("Mars", "discovered_by", "Galileo", source="NASA", add_to_graph=False)
    assert len(knowledge_graph) == 1
    assert len(knowledge_graph.graph_backend) == 0 # "discovered_by" should not be added to graph


def test_add_fact_with_force_add(knowledge_graph):
    timestamp = knowledge_graph.add_fact("Mars", "discovered_by", "Galileo", source="NASA", add_to_graph=True)

    assert len(knowledge_graph) == 1
    assert len(knowledge_graph) == 1  # Force-added to graph


def test_get_facts_about(knowledge_graph):
    knowledge_graph.add_fact("Sun", "is_a", "Star", source="NASA")
    knowledge_graph.add_fact("Moon", "orbits", "Earth", source="NASA")

    results = knowledge_graph.get_facts_about(subject="Sun")
    assert len(results) == 1
    assert results[0][:3] == ("Sun", "is_a", "Star")


def test_get_facts_with_filters(knowledge_graph):
    # make sure table exists
    assert knowledge_graph.knowledge_table_name == 'knowledge'
    assert knowledge_graph.database.tables['knowledge'] is not None

    id1, ts1 = knowledge_graph.add_fact("Earth", "is_a", "Planet", source="NASA")
    time.sleep(0.001)
    id2, ts2 = knowledge_graph.add_fact("Pluto", "is_a", "Dwarf Planet", source="IAU")


    assert ts1 < ts2
    results = knowledge_graph.get_facts_about(start_time=ts1, end_time=ts2)
    assert len(results) == 2  # Both fall in range

    results = knowledge_graph.get_facts_about(start_time=ts2)
    assert len(results) == 1  # Only Pluto fact remains


def test_remove_fact(knowledge_graph):
    _fact_id, _timestamp = knowledge_graph.add_fact(subject="Venus", predicate="is_a", obj="Planet", source="NASA")
    fact_id = knowledge_graph.generate_fact_id(subject="Venus", predicate="is_a", obj="Planet", source="NASA", timestamp=_timestamp)
    assert _fact_id == fact_id

    knowledge_graph.remove_fact(fact_id)

    assert knowledge_graph.database.get_number_of_rows("knowledge") == 0
    assert len(knowledge_graph.graph_backend.get_list_of_edges()) == 0

def test_clear_knowledge(knowledge_graph):
    knowledge_graph.add_fact(fact_id='f1', subject="Jupiter", predicate="is_a", obj="Gas Giant", source="NASA")
    knowledge_graph.add_fact(fact_id='f2', subject="Saturn", predicate="is_a", obj="Gas Giant", source="NASA")

    knowledge_graph.clear()

    assert knowledge_graph.database.get_number_of_rows("knowledge") == 0
    assert len(knowledge_graph.graph_backend.get_list_of_edges()) == 0
