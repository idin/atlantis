import openai
from typing import List, Dict, Any, Optional
from .KnowledgeGraphMemory import KnowledgeGraphMemory  # ✅ Import the tested Knowledge Graph Memory


class OpenAIWrapper:
    """
    A simple wrapper for OpenAI's API that integrates with KnowledgeGraphMemory.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.5):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.memory: List[Dict[str, str]] = []  # Conversation history
        self.knowledge_graph = KnowledgeGraphMemory()  # ✅ Integrated Knowledge Graph Memory

    def _call_openai(self, prompt: str, extract_facts: bool = True) -> str:
        """
        Calls OpenAI API and returns the response text.
        """
        response = openai.ChatCompletion.create(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            messages=self.memory + [{"role": "user", "content": prompt}]
        )
        message = response["choices"][0]["message"]["content"]

        # ✅ Only store in memory if this is a normal user query, NOT a fact extraction call
        if extract_facts:
            self.memory.append({"role": "user", "content": prompt})
            self.memory.append({"role": "assistant", "content": message})

            # ✅ Extract facts only if it's a user query
            self._extract_and_store_facts(prompt, message)

        return message

    def query(self, prompt: str) -> str:
        """
        Queries the AI and returns a response, first checking the knowledge graph.
        """
        # ✅ Step 1: Check the knowledge graph for relevant info
        known_facts = self._retrieve_relevant_facts(prompt)
        if known_facts:
            return known_facts  # ✅ Return stored knowledge instead of calling OpenAI

        # ✅ Step 2: If no relevant info, call OpenAI
        return self._call_openai(prompt)

    def clear_memory(self):
        """
        Clears conversation history.
        """
        self.memory = []

    def _extract_and_store_facts(self, user_prompt: str, ai_response: str):
        """
        Extracts structured facts from the AI response and stores them in the knowledge graph.
        """
        fact_extraction_prompt = f"""
        Extract key facts from the following text in the format (subject, relationship, object):

        "{ai_response}"

        Example output:
        - (Paris, capital_of, France)
        - (Einstein, discovered, Theory of Relativity)
        - (Python, is_a, Programming Language)
        """
        # ✅ Prevent fact extraction from modifying memory
        extracted_facts = self._call_openai(fact_extraction_prompt, extract_facts=False)

        # ✅ Step 3: Parse and store extracted facts
        for line in extracted_facts.split("\n"):
            line = line.strip()
            if line.startswith("- (") and line.endswith(")"):
                parts = line[3:-1].split(", ")
                if len(parts) == 3:
                    self.knowledge_graph.add_fact(parts[0], parts[1], parts[2])

    def _retrieve_relevant_facts(self, user_prompt: str) -> Optional[str]:
        """
        ✅ Checks if the knowledge graph has relevant information for the user query.
        """
        words = user_prompt.split()
        for word in words:
            facts = self.knowledge_graph.get_facts_about(word)
            if facts:
                return f"Known facts about {word}: {facts}"
        return None
