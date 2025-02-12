import pytest
from atlantis.ai import KnowledgeGraphMemory  # ✅ Correct import


# ✅ Step 1: Create a new KnowledgeGraphMemory instance
def test_knowledge_graph_memory():
    kg = KnowledgeGraphMemory()

    # ✅ Step 2: Add Facts
    kg.add_fact("Paris", "capital_of", "France")
    kg.add_fact("Einstein", "discovered", "Theory of Relativity")
    kg.add_fact("Python", "is_a", "Programming Language")

    # ✅ Step 3: Retrieve a Single Fact
    assert kg.get_fact("Paris", "capital_of") == "France"
    assert kg.get_fact("Einstein", "discovered") == "Theory of Relativity"
    assert kg.get_fact("Python", "is_a") == "Programming Language"

    # ✅ Step 4: Retrieve Facts About a Subject
    assert kg.get_facts_about("Paris") == {"capital_of": "France"}
    assert kg.get_facts_about("Einstein") == {"discovered": "Theory of Relativity"}
    assert kg.get_facts_about("Python") == {"is_a": "Programming Language"}

    # ✅ Step 5: Retrieve the Entire Knowledge Graph
    expected_graph = {
        "Paris": {"capital_of": "France"},
        "Einstein": {"discovered": "Theory of Relativity"},
        "Python": {"is_a": "Programming Language"},
    }
    assert kg.get_all_facts() == expected_graph


# ✅ Run the test manually (if running outside `pytest`)
if __name__ == "__main__":
    test_knowledge_graph_memory()
    print("All tests passed!")
