from .Fact import Fact
from typing import Union, List, Tuple, Dict

class FactsCollection:
    def __init__(self, facts: Union[List[Union[Tuple, Dict, Fact]], Dict[str, Union[Tuple, Dict, Fact]]], include_source: bool = True, include_timestamp: bool = False  ):
        self._process_facts(facts, include_source=include_source, include_timestamp=include_timestamp)

    def _process_facts(self, facts: Union[List[Union[Tuple, Dict, Fact]], Dict[str, Union[Tuple, Dict, Fact]]], include_source: bool = True, include_timestamp: bool = False) -> List[Fact]:
        if isinstance(facts, list):
            self.facts = {}
            for fact in facts:
                _fact = Fact.from_any_type(fact, to_type="fact", include_source=include_source, include_timestamp=include_timestamp)
                self.facts[_fact.fact_id] = _fact
        elif isinstance(facts, dict):
            self.facts = {key: Fact.from_any_type(fact, to_type="fact", include_source=include_source, include_timestamp=include_timestamp) for key, fact in facts.items()}
        else:
            raise ValueError("Invalid facts type")
        
