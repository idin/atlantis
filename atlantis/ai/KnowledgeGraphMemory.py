import ast
from langchain_core.memory import BaseMemory
from typing import Dict, Any, List
from atlantis.ai.memory.KnowledgeGraphDB import KnowledgeGraphDB  # Adjust the import based on your actual file structure

class KnowledgeGraphMemory(BaseMemory):
    """
    A LangChain-compatible memory module using KnowledgeGraphDB.
    """

    kg: Any = None
    llm: Any = None  

    def __init__(self, llm, db_path: str = ":memory:"):
        """
        Initializes KnowledgeGraphDB as a LangChain memory module.
        """
        super().__init__()
        object.__setattr__(self, "kg", KnowledgeGraphDB(db_path=db_path))  
        object.__setattr__(self, "llm", llm)  
        self._fields = ["kg", "llm"]  

    @property
    def memory_variables(self) -> List[str]:
        """Required property for LangChain memory modules."""
        return ["history"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically retrieves relevant facts based on the user's query,
        considering both subjects and objects.
        """
        query = inputs.get("input", "").strip()
        if not query:
            return {"history": ""}

        # Ask OpenAI to extract relevant subjects and objects from the query
        prompt = f"Extract key subject and object phrases from this query that are relevant for recalling knowledge:\n\n'{query}'\n\nReturn a Python list of strings."
        response = self.llm.invoke(prompt).content

        # Ensure the response is a properly formatted Python list
        try:
            relevant_terms = ast.literal_eval(response)  
        except (SyntaxError, ValueError):
            relevant_terms = []

        # Retrieve relevant knowledge
        retrieved_facts = []
        for term in relevant_terms:
            # Search for facts where the term appears as either a subject or an object
            subject_facts = self.kg.get_facts_about(term)  
            object_facts = self.kg.db.execute(
                "SELECT subject, relationship FROM knowledge WHERE object = ?", (term,)
            ).fetchall()  

            # Format retrieved facts
            retrieved_facts.extend([f"{term} {rel} {obj}" for rel, obj in subject_facts])
            retrieved_facts.extend([f"{subj} {rel} {term}" for subj, rel in object_facts])

        return {"history": "\n".join(retrieved_facts)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Uses AI to extract structured facts from user input and stores them in the Knowledge Graph.
        """
        input_text = inputs.get("input", "")
        output_text = outputs.get("output", "")

        # Query OpenAI to extract structured triples (subject, relationship, object)
        prompt = f"Extract structured knowledge triples (subject, relationship, object) from this sentence:\n\n'{input_text}'\n\nReturn a list of tuples like: [('Idin', 'loves', 'playing bass'), ('Idin', 'enjoys', 'writing code')]."
        response = self.llm.invoke(prompt)

        # Ensure the response is in the correct format
        try:
            extracted_facts = ast.literal_eval(response.content)  
        except (SyntaxError, ValueError):
            extracted_facts = []

        # Store extracted facts in the Knowledge Graph
        for fact in extracted_facts:
            if len(fact) == 3:
                subject, relationship, obj = fact
                self.kg.add_fact(subject, relationship, obj)

        # Store the AI's response as a separate fact
        self.kg.add_fact("AI", "responded_with", output_text)

    def clear(self) -> None:
        """
        Clears all stored knowledge.
        """
        self.kg.clear_knowledge()
