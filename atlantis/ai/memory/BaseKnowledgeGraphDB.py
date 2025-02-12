import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Union, Literal, Callable
from .Fact import Fact
from .graph.GraphBackend import GraphBackend
from .db.GraphDB import GraphDB

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc


class BaseKnowledgeGraphDB:
    """
    A base knowledge graph database that supports fact storage and retrieval.

    - Every fact is stored in `GraphDB`.
    - Facts are stored in `GraphBackend` only if:
        - `add_to_graph=True`
        - `add_to_graph=None` and the predicate is in `relationship_predicates`

    This class has the following methods:
        - `_build_fact_id_function()`
        - `_initialize_knowledge_table()`
        - `__len__()`
        - `_is_fact_id()`
        - `_add_fact_to_db()`
        - `_add_fact_to_graph()`
        - `_get_fact()`
        - `_remove_fact()`
        - `_facts_as_dict()`
        - `_facts_as_fact()`
        - `_get_fact()`
        - `_remove_fact()`
    """

    DEFAULT_RELATIONSHIP_PREDICATES = {
        "is_a", "related_to", "knows", "friend_of",
        "parent_of", "child_of", "works_with", "lives_in", "part_of",
        "is_subject_of", "is_object_of"
    }

    def __init__(
            self, database: GraphDB, graph_backend: GraphBackend, relationship_predicates: Optional[set] = None, 
            knowledge_table_name: Optional[str] = "knowledge",
            fact_id_key: Optional[tuple] = ("subject", "predicate", "obj", "source")
        ):
        """
        Initializes the knowledge graph.

        Args:
            graph_db (GraphDB): The database backend for structured storage.
            graph_backend (GraphBackend): The graph backend for relationships.
            relationship_predicates (set, optional): The relationship predicates to use. If not provided, the default relationship predicates will be used.
        """
        self.database = database
        # if knowledge_table_name is in the database raise an error
        if knowledge_table_name in self.database.tables:
            raise ValueError(f"Table {knowledge_table_name} already exists in the database.")
        self.knowledge_table_name = knowledge_table_name
        self.graph_backend = graph_backend
        self.relationship_predicates = relationship_predicates or set(self.DEFAULT_RELATIONSHIP_PREDICATES)
        self._initialize_knowledge_table()
        self.fact_id_key = fact_id_key

        # make sure object is renamed to obj in fact_id_key
        if "object" in fact_id_key:
            fact_id_key = fact_id_key.replace("object", "obj")
        self._generate_fact_id = self._build_fact_id_function(fact_id_key=fact_id_key)
        
    @staticmethod
    def _build_fact_id_function(fact_id_key: tuple) -> Callable:
        def generate_fact_id(**kwargs) -> str:
            string = '|'.join([f'{kwargs[key]}' for key in fact_id_key])
            return f"fact_{hashlib.sha256(string.encode()).hexdigest()}"
        return generate_fact_id

    def _initialize_knowledge_table(self, primary_keys: Optional[list] = None):
        """
        Ensures that the 'knowledge' table exists in the database using `create_table()`.
        """
        if primary_keys is None:
            other_kwargs = {}
        else:
            other_kwargs = {"primary_keys": primary_keys}
        
        self.database.create_table(
            table_name=self.knowledge_table_name,
            columns={
                "fact_id": "text",
                "subject": "text",
                "subject_lower": "text",
                "predicate": "text",
                "predicate_lower": "text",
                "object": "text",
                "object_lower": "text",
                "timestamp": "datetime",
                "source": "text"

            },
            **other_kwargs
        )

    def __len__(self) -> int:
        """
        Returns the number of facts in the database.
        """
        return self.database.get_number_of_rows(table_name=self.knowledge_table_name)
    
    @staticmethod
    def _is_fact_id(subject_or_obj: str) -> bool:
        return subject_or_obj.startswith("fact_")

    def _add_fact_to_db(
        self,
        subject: str, 
        predicate: str, 
        obj: str, 
        fact_id: Optional[str] = None,
        source: Optional[str] = None, 
        timestamp: Optional[datetime] = None

    ) -> Tuple[str, datetime]:
        """
        Adds a fact to the database.

        Every fact is always stored in `GraphDB`.

        Args:
            subject (str): The subject of the fact.
            predicate (str): The predicate of the fact.
            obj (str): The object of the fact.
            source (str, optional): The source of the fact.
            timestamp (datetime, optional): The timestamp of the fact.

        Returns:
            datetime: The timestamp when the fact was stored.
        """
        if not isinstance(subject, str):
            raise TypeError(f"Subject should be a string but it is a {subject} of type {type(subject)}.")
        if not isinstance(obj, str):
            raise TypeError(f"Object should be a string but it is a {obj} of type {type(obj)}.")
        if not isinstance(predicate, str):
            raise TypeError(f"Predicate should be a string but it is a {predicate} of type {type(predicate)}.")
        if not isinstance(source, str) and source is not None:
            raise TypeError(f"Source should be a string but it is a {source} of type {type(source)}.")
        if not isinstance(timestamp, datetime) and timestamp is not None:
            raise TypeError(f"Timestamp should be a datetime but it is a {timestamp} of type {type(timestamp)}.")

        if fact_id is None:
            fact_id = self.generate_fact_id(subject=subject, predicate=predicate, obj=obj, source=source, timestamp=timestamp)
        if timestamp is None:
            timestamp = datetime.now(UTC)

        self.database._add_row(
            table_name="knowledge",
            fact_id=fact_id,
            subject=subject,
            subject_lower=subject.lower(),
            predicate=predicate,
            predicate_lower=predicate.lower(),
            object=obj,
            object_lower=obj.lower(),
            source=source,
            timestamp=timestamp
        )

        return fact_id, timestamp

    def _add_fact_to_graph(
        self, 
        subject: str, 
        predicate: str, 
        obj: str, 
        fact_id: str,
        source: Optional[str] = None, 
        timestamp: Optional[datetime] = None
    ) -> Optional[datetime]:
        """
        Adds a fact to the graph backend.
        If it is a regular fact: Subject is Node1, Object is Node2, Predicate is Edge.
        If it is a fact about a fact: 
            if fact_1 is subject_1 predicate_1 object_1
            and fact_2 is fact_1 predicate_2 object_2

            the subject_1 is a node, predicate_1 is an edge, object_1 is a node.
            subject_1 is_subject_of fact_1 is another connection: is_subject_of is the edge and fact_1 is the object.
            fact_1 is the node, predicate_2 is the edge, object_2 is the object.

        A fact is only stored in `GraphBackend` if:
          - The predicate is in `relationship_predicates`, OR
          - `force_add=True` is explicitly set.

        Args:
            subject (str): The subject of the fact.
            predicate (str): The predicate of the fact.
            obj (str): The object of the fact.
            fact_id (str): The ID of the fact.
            source (str, optional): The source of the fact.
            timestamp (datetime, optional): The timestamp of the fact.


        Returns:
            datetime: The timestamp if added, otherwise `None`.
        """
        if not isinstance(subject, str):
            raise TypeError(f"Subject should be a string but it is a {subject} of type {type(subject)}.")
        if not isinstance(obj, str):
            raise TypeError(f"Object should be a string but it is a {obj} of type {type(obj)}.")
        if not isinstance(predicate, str):
            raise TypeError(f"Predicate should be a string but it is a {predicate} of type {type(predicate)}.")
        if not isinstance(source, str):
            raise TypeError(f"Source should be a string but it is a {source} of type {type(source)}.")
        if not isinstance(timestamp, datetime):
            raise TypeError(f"Timestamp should be a datetime but it is a {timestamp} of type {type(timestamp)}.")
        
        # we have to use fact_source to store the source of the fact to avoid conflicts with the source of the edge
        self.graph_backend.add_edge(source=subject, target=obj, edge=predicate, fact_id=fact_id, fact_source=source, timestamp=timestamp)
        return timestamp

    def _facts_as_dict(self, facts_as_tuples: List[Tuple], include_fields: List[str]) -> List[Dict]:
        return [{field: value for field, value in zip(include_fields, fact)} for fact in facts_as_tuples]

    def _facts_as_fact(self, facts_as_tuples: List[Tuple], include_fields: List[str]) -> List[Fact]:
        return [Fact(**{field: value for field, value in zip(include_fields, fact)}) for fact in facts_as_tuples]   
    
    def _get_fact(self, fact_id: str, return_type: Literal["tuple", "dict", "fact"] = "fact"):
        include_fields = ['fact_id', 'subject', 'predicate', 'object', 'source', 'timestamp']
        fact = self.database.get_one(table_name="knowledge", fact_id=fact_id, include_fields=include_fields)
        if fact is None:
            raise KeyError(f"Fact with ID {fact_id} not found.")

        elif isinstance(fact, list):
            if len(fact) == 1:
                fact = fact[0]
            else:
                raise ValueError(f"Multiple facts found with ID {fact_id}.")
        if not isinstance(fact, tuple):
            raise TypeError(f"Fact should be a tuple but it is a {fact} of type {type(fact)}.")
        if return_type == "tuple":
            return fact
        elif return_type == "dict":
            return {field: value for field, value in zip(include_fields, fact)}
        elif return_type == "fact":
            return Fact(id=fact[0], subject=fact[1], predicate=fact[2], obj=fact[3], source=fact[4], timestamp=fact[5])

    def _remove_fact(self, fact_id: str):
        """
        Removes a fact from both the database and the graph backend.
        """
        fact = self._get_fact(fact_id, return_type='fact')
        if fact is None:
            raise KeyError(f"Fact with ID {fact_id} not found.")
        self.database._delete_rows(table_name="knowledge", fact_id=fact_id)
        self.graph_backend.remove_edge(source=fact.subject, target=fact.obj, edge=fact.predicate)
