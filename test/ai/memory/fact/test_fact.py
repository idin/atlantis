import pytest
from datetime import datetime, date
from atlantis.ai.memory import Fact


@pytest.fixture
def fact():
    """Fixture to create a sample fact."""
    return Fact(
        id="fact_123",
        subject="Paris",
        predicate="capital_of",
        obj="France",
        source="historical_data",
        timestamp=datetime(2023, 5, 1, 12, 0, 0)
    )


def test_fact_has_all_methods():
    """Test that the Fact class has all required methods."""
    required_methods = {
        "to_tuple",
        "to_dict",
        "to_type",
        "from_tuple",
        "from_dict",
        "from_type",
        "convert",
        "__contains__",
        "equals",
        "__eq__",
        "__hash__"
    }
    fact_methods = set(dir(Fact))
    missing_methods = required_methods - fact_methods

    assert not missing_methods, f"Fact class is missing methods: {missing_methods}"


def test_fact_to_tuple(fact):
    """Test converting Fact to tuple."""
    assert fact.to_tuple() == ("fact_123", "Paris", "capital_of", "France", "historical_data")
    assert fact.to_tuple(include_timestamp=True) == ("fact_123", "Paris", "capital_of", "France", "historical_data", datetime(2023, 5, 1, 12, 0, 0))


def test_fact_to_dict(fact):
    """Test converting Fact to dictionary."""
    assert fact.to_dict() == {
        "id": "fact_123",
        "subject": "Paris",
        "predicate": "capital_of",
        "obj": "France"
    }

    assert fact.to_dict(include_source=True) == {
        "id": "fact_123",
        "subject": "Paris",
        "predicate": "capital_of",
        "obj": "France",
        "source": "historical_data"
    }


def test_fact_from_tuple():
    """Test creating Fact from tuple."""
    fact = Fact.from_tuple(
        tuple_data=("fact_123", "Paris", "capital_of", "France", "historical_data", datetime(2023, 5, 1, 12, 0, 0))
    )
    assert fact.timestamp is None

    fact = Fact.from_tuple(
        tuple_data=("fact_123", "Paris", "capital_of", "France", "historical_data", datetime(2023, 5, 1, 12, 0, 0)),
        include_timestamp=True
    )
    assert fact.timestamp == datetime(2023, 5, 1, 12, 0, 0)


def test_fact_from_dict():
    """Test creating Fact from dictionary."""
    fact = Fact.from_dict(
        dict_data={
            "id": "fact_123",
            "subject": "Paris",
            "predicate": "capital_of",
            "obj": "France",
            "source": "historical_data",
            "timestamp": datetime(2023, 5, 1, 12, 0, 0)
        }
    )
    assert fact.timestamp is None

    fact = Fact.from_dict(
        dict_data={
            "id": "fact_123",
            "subject": "Paris",
            "predicate": "capital_of",
            "obj": "France",
            "source": "historical_data",
            "timestamp": datetime(2023, 5, 1, 12, 0, 0)
        },
        include_timestamp=True
    )
    assert fact.timestamp == datetime(2023, 5, 1, 12, 0, 0)


def test_fact_contains(fact):
    """Test Fact containment check."""
    assert "Paris" in fact
    assert "capital_of" in fact
    assert "France" in fact
    assert fact.timestamp in fact
    assert fact.date in fact
    assert str(fact.date) == "2023-05-01"
    assert str(fact.date) in fact
    assert str(fact.timestamp) == "2023-05-01 12:00:00"
    assert str(fact.timestamp) in fact
    assert fact.timestamp.isoformat() in fact


def test_fact_equals(fact):
    """Test Fact equality comparison."""
    same_fact = Fact(
        id="fact_123",
        subject="Paris",
        predicate="capital_of",
        obj="France",
        source="historical_data",
        timestamp=datetime(2023, 5, 1, 12, 0, 0)
    )
    assert fact.equals(same_fact)
    assert fact == same_fact


def test_fact_hash():
    """Test Fact hashing for dictionary keys."""
    fact_dict = {fact: "exists"}
    assert fact in fact_dict


def test_fact_convert(fact):
    """Test Fact conversion to different formats."""
    tuple_fact = fact.to_type(to_type="tuple")
    assert tuple_fact == ("fact_123", "Paris", "capital_of", "France", "historical_data")

    dict_fact = fact.to_type(to_type="dict")
    assert dict_fact == {
        "id": "fact_123",
        "subject": "Paris",
        "predicate": "capital_of",
        "obj": "France",
        "source": "historical_data"
    }

    fact_copy = fact.to_type(to_type="fact")
    assert fact_copy == fact


if __name__ == "__main__":
    pytest.main()
