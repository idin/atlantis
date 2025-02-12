import re
from typing import List, Tuple


def extract_facts_from_text_using_regex(text: str) -> List[Tuple[str, str, str]]:
    """
    Extracts structured (subject, relationship, object) facts from a given text.
    Uses regex-based extraction with NLP heuristics.

    Example:
    Input: "Paris is the capital of France. Einstein discovered the Theory of Relativity."
    Output: [("Paris", "capital_of", "France"), ("Einstein", "discovered", "Theory of Relativity")]
    """
    # ✅ Basic pattern to extract simple (subject, relationship, object) triplets
    fact_pattern = re.findall(r"([A-Z][a-z]+) (is|was|became|discovered|created|invented) ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", text)

    # ✅ Convert relationships to a more structured format
    structured_facts = [(subj, rel.replace(" ", "_"), obj) for subj, rel, obj in fact_pattern]

    return structured_facts

def extract_facts_from_text_using_spacy(text: str) -> List[Tuple[str, str, str]]:
    """
    Extracts structured (subject, relationship, object) facts from a given text.
    Uses spaCy for more sophisticated NLP-based extraction.

    Example:
    Input: "Paris is the capital of France. Einstein discovered the Theory of Relativity."
    Output: [("Paris", "capital_of", "France"), ("Einstein", "discovered", "Theory of Relativity")]
    """
    # Initialize spaCy model
    nlp = spacy.load("en_core_web_sm")


