import pytest
import duckdb
from atlantis.ai.memory.db import DuckDBGraphDB
from datetime import datetime, date

try:
    from datetime import UTC
except ImportError:

    from datetime import timezone
    UTC = timezone.utc

@pytest.fixture
def db():
    """Fixture to create a fresh DuckDBGraphDB instance for each test."""
    db_instance = DuckDBGraphDB()
    db_instance.create_table(
        table_name="facts", 
        columns={
            "id": "integer",
            "subject": "text",
            "relationship": "text",
            "object": "text",
            "timestamp": "datetime"
        }
    )
    return db_instance


def test_create_table_valid_schema(db):
    """Test table creation with a valid schema."""
    db.create_table("test_table", {
        "name": "text",
        "age": "integer",
        "score": "float",
        "is_active": "boolean",
        "birthdate": "date",
        "created_at": "datetime"
    })
    assert db.get_rows("test_table") == []  # Ensure table is empty
    assert db.tables["test_table"]["number_of_rows_added"] == 0
    assert db.tables["test_table"]["number_of_rows_accessed"] == 0
    assert db.tables["test_table"]["number_of_rows_deleted"] == 0


def test_create_table_invalid_schema(db):
    """Test that creating a table with an invalid type raises an error."""
    with pytest.raises(ValueError, match="Unsupported column type: unknown_type"):
        db.create_table("invalid_table", {
            "name": "text",
            "invalid_col": "unknown_type"
        })


def test_add_and_get_all(db):
    """Test adding a row and retrieving it."""
    db.add_row("facts", id=1, subject="Paris", relationship="capital_of", object="France", timestamp=datetime.now(UTC))


    results = db.get_rows("facts")
    assert len(results) == 1
    assert results[0][:4] == (1, "Paris", "capital_of", "France")  # Ignore timestamp comparison

def test_range_query(db):
    """Test range query."""

    t1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    t2 = datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC) # <- Paris
    t3 = datetime(2024, 12, 1, 0, 0, 0, tzinfo=UTC)
    t4 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC) # <- Berlin
    t5 = datetime(2025, 6, 1, 0, 0, 0, tzinfo=UTC)

    assert t1 < t2 < t3 < t4 < t5

    db.add_row("facts", id=1, subject="Paris", relationship="capital_of", object="France", timestamp=t2)
    db.add_row("facts", id=2, subject="Berlin", relationship="capital_of", object="Germany", timestamp=t4)
    assert db.tables["facts"]["number_of_rows_added"] == 2
    assert db.tables["facts"]["number_of_rows_accessed"] == 0
    assert db.tables["facts"]["number_of_rows_deleted"] == 0


    from_t1_to_t5 = db.get_rows("facts", timestamp=(t1, t5))
    assert len(from_t1_to_t5) == 2
    assert db.tables["facts"]["number_of_rows_accessed"] == 2
    assert db.tables["facts"]["number_of_rows_deleted"] == 0
    from_t1_to_t3 = db.get_rows("facts", timestamp=(t1, t3))
    assert len(from_t1_to_t3) == 1
    assert from_t1_to_t3[0][:4] == (1, "Paris", "capital_of", "France")
    assert db.tables["facts"]["number_of_rows_accessed"] == 3
    assert db.tables["facts"]["number_of_rows_deleted"] == 0

    from_t2_to_t4 = db.get_rows("facts", timestamp=(t2, t4))
    assert len(from_t2_to_t4) == 2

    from_t3_to_t5 = db.get_rows("facts", timestamp=(t3, t5))
    assert len(from_t3_to_t5) == 1
    assert from_t3_to_t5[0][:4] == (2, "Berlin", "capital_of", "Germany")

    before_t5 = db.get_rows("facts", timestamp=(None, t5))
    assert len(before_t5) == 2

    after_t1 = db.get_rows("facts", timestamp=(t1, None))
    assert len(after_t1) == 2

    before_t4 = db.get_rows("facts", timestamp=(None, t4))
    assert len(before_t4) == 2

    after_t2 = db.get_rows("facts", timestamp=(t2, None))
    assert len(after_t2) == 2
    
    before_t3 = db.get_rows("facts", timestamp=(None, t3))
    assert len(before_t3) == 1
    assert before_t3[0][:4] == (1, "Paris", "capital_of", "France")

    after_t3 = db.get_rows("facts", timestamp=(t3, None))
    assert len(after_t3) == 1
    assert after_t3[0][:4] == (2, "Berlin", "capital_of", "Germany")
    
def test_get_all_with_conditions(db):
    """Test filtering in `get_all()`."""
    db.add_row("facts", id=1, subject="Paris", relationship="capital_of", object="France", timestamp=datetime.now(UTC))
    db.add_row("facts", id=2, subject="Berlin", relationship="capital_of", object="Germany", timestamp=datetime.now(UTC))

    # Retrieve by subject
    results = db.get_rows("facts", subject="Paris")
    assert len(results) == 1
    assert results[0][1] == "Paris"

    # Retrieve by relationship
    results = db.get_rows("facts", relationship="capital_of")
    assert len(results) == 2  # Both Paris and Berlin should match


def test_get_all_with_range_condition(db):
    """Test retrieving rows within a range."""
    db.add_row("facts", id=1, subject="A", relationship="test", object="X", timestamp=datetime.now(UTC))
    db.add_row("facts", id=2, subject="B", relationship="test", object="Y", timestamp=datetime.now(UTC))
    db.add_row("facts", id=3, subject="C", relationship="test", object="Z", timestamp=datetime.now(UTC))


    results = db.get_rows("facts", id=(1, 2))  # Should return records with id 1 and 2
    assert len(results) == 2
    assert set(r[1] for r in results) == {"A", "B"}  # Should contain only A and B


def test_delete_all(db):
    """Test deleting rows with `delete_all()`."""
    db.add_row("facts", id=1, subject="Paris", relationship="capital_of", object="France", timestamp=datetime.now(UTC))
    db.add_row("facts", id=2, subject="Berlin", relationship="capital_of", object="Germany", timestamp=datetime.now(UTC))
    assert db.tables["facts"]["number_of_rows_added"] == 2
    assert db.tables["facts"]["number_of_rows_accessed"] == 0
    assert db.tables["facts"]["number_of_rows_deleted"] == 0


    db.delete_all_rows("facts", subject="Paris")
    assert db.tables["facts"]["number_of_rows_deleted"] == 2
    results = db.get_rows("facts")

    assert len(results) == 1  # Only Berlin should remain
    assert results[0][1] == "Berlin"
    assert db.tables["facts"]["number_of_rows_accessed"] == 1

def test_delete_all(db):
    """Test that `delete_all()` there is no limit parameter."""
    db.add_row("facts", id=1, subject="A", relationship="test", object="X", timestamp=datetime.now(UTC))
    db.add_row("facts", id=2, subject="B", relationship="test", object="Y", timestamp=datetime.now(UTC))
    db.add_row("facts", id=3, subject="C", relationship="test", object="Z", timestamp=datetime.now(UTC))


    deleted_rows = db.delete_rows(table_name="facts", relationship="test")  # Only delete 2 rows
    assert deleted_rows is None or deleted_rows == 2 or len(deleted_rows) == 2
    results = db.get_rows(table_name="facts")

    assert len(results) == 0
    assert db.tables["facts"]["number_of_rows_deleted"] == 2 or deleted_rows is None
    assert db.tables["facts"]["number_of_rows_added"] == 3
    assert db.tables["facts"]["number_of_rows_accessed"] == 0

def test_clear(db):
    """Test clearing all data from a table."""
    db.add_row(table_name="facts", id=1, subject="Paris", relationship="capital_of", object="France", timestamp=datetime.now(UTC))


    db.delete_all_rows("facts")
    assert db.get_rows("facts") == []  # Should be empty


if __name__ == "__main__":
    pytest.main()
