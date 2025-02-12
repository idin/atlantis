import pytest
from atlantis.ai.memory import Fact


def test_fact_as_subject_of_another_fact():
    """Test a Fact being used as the subject of another Fact."""
    f1 = Fact(id="f1", subject="Paris", predicate="is_capital_of", obj="France")
    f2 = Fact(id="f2", subject=f1, predicate="is", obj="correct")

    # Direct containment
    assert "Paris" in f1
    assert f1 in f2

    # Nested containment
    assert "Paris" in f2
    assert "paris" in f2  # Case-insensitive check
    assert "is" in f2
    assert "capital" not in f1
    assert "capital" not in f2
    assert "correct" in f2
    assert "france" in f1
    assert "france" in f2


def test_fact_as_object_of_another_fact():
    """Test a Fact being used as the object of another Fact."""
    f1 = Fact(id="f1", subject="Earth", predicate="is_part_of", obj="Solar System")
    f2 = Fact(id="f2", subject="FactCheck", predicate="verified", obj=f1)

    # Direct containment
    assert "Earth" in f1
    assert f1 in f2

    # Nested containment
    assert "Earth" in f2
    assert "solar system" in f1
    assert "solar system" in f2
    assert "is_part_of" in f2
    assert "FactCheck" in f2
    assert "galaxy" not in f2


def test_fact_as_subject_and_object_of_another_fact():
    """Test a Fact being used as both the subject and object of another Fact."""
    f1 = Fact(id="f1", subject="AI", predicate="is_capable_of", obj="learning")
    f2 = Fact(id="f2", subject="research", predicate="proves", obj=f1)
    f3 = Fact(id="f3", subject=f1, predicate="is", obj=f2)

    # Direct containment
    assert "AI" in f1
    assert "learning" in f1
    assert f1 in f2
    assert f1 in f3
    assert f2 in f3

    # Nested containment
    assert "AI" in f3
    assert "learning" in f3
    assert "is_capable_of" in f3
    assert "research" in f3
    assert "proves" in f3
    assert "is" in f3

    # Negative checks
    assert "random" not in f3
    assert "unknown" not in f3


if __name__ == "__main__":
    pytest.main()
