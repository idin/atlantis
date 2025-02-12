import pytest
from atlantis.ai import FactRetriever


def test_extract_facts():
    retriever = FactRetriever()

    # Test 1: Basic Sentence Extraction
    text = "Paris is the capital of France."
    extracted_facts = retriever.extract_facts(text)
    assert any(fact[0] == "Paris" and fact[1] == "be" and "capital" in fact[2] for fact in extracted_facts)

    # Test 2: Multiple Sentences
    text = """
    Einstein discovered the Theory of Relativity.
    Tesla invented alternate current.
    Apple is headquartered in California.
    """
    extracted_facts = retriever.extract_facts(text)

    assert any(fact[0] == "Einstein" and fact[1] == "discover" and "Theory of Relativity" in fact[2] for fact in extracted_facts)
    assert any(fact[0] == "Tesla" and fact[1] == "invent" and "alternate current" in fact[2] for fact in extracted_facts)
    print(extracted_facts)
    assert any(fact[0] == "Apple" and fact[1] == "be" and "headquartered in California" in fact[2] for fact in extracted_facts)

    # Test 3: Complex Sentence Structure (Passive Voice)
    text = "The Mona Lisa was painted by Leonardo da Vinci in the 16th century."
    extracted_facts = retriever.extract_facts(text)

    # Expect BOTH facts: 
    # - Mona Lisa was painted
    # - Leonardo da Vinci painted Mona Lisa
    assert any(fact[0] in ["Mona Lisa", "Lisa"] and fact[1] == "be" and "painted" in fact[2] for fact in extracted_facts)
    assert any(fact[0] == "Leonardo da Vinci" and fact[1] == "paint" and "Mona Lisa" in fact[2] for fact in extracted_facts)


if __name__ == "__main__":
    test_extract_facts()
    print("All fact extraction tests passed!")
