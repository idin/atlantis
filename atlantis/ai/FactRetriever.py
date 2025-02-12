import spacy
from typing import List, Tuple


class FactRetriever:
    """
    Extracts structured (subject, relationship, object) facts from text using SpaCy.
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_facts(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extracts (subject, relationship, object) triplets from the given text.

        Example:
            Input: "The Mona Lisa was painted by Leonardo da Vinci in the 16th century."
            Output: [("Mona Lisa", "be", "painted"), ("Leonardo da Vinci", "paint", "Mona Lisa")]
        """
        doc = self.nlp(text)
        facts = []

        for sent in doc.sents:
            subject, relation, obj = None, None, None

            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass"):  # Subject
                    subject = token.text

                if token.dep_ == "ROOT":  # Main verb (relationship)
                    relation = token.text if token.lemma_ in ["headquarter"] else token.lemma_

                if token.dep_ in ("attr", "dobj"):  # Object or attribute
                    obj = " ".join([child.text for child in token.subtree if child.pos_ != "DET"])

                # Capture prepositional objects (e.g., "headquartered in California")
                if token.dep_ == "prep" and subject and relation:
                    prep_phrase = " ".join([token.text] + [child.text for child in token.subtree if child.dep_ == "pobj"])
                    obj = f"{relation} {prep_phrase}"
                    relation = "be" if relation == "is" else relation

                if subject and relation and obj:
                    facts.append((subject, relation, obj))
                    break  # Stop after first valid fact per sentence

        return facts
