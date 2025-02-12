import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Union, Set, Literal
from .db import GraphDB, DuckDBGraphDB
from .graph import GraphBackend, NetworkXGraphBackend
from .BaseKnowledgeGraphDB import BaseKnowledgeGraphDB
from .Fact import Fact

try:
    from datetime import UTC
except ImportError:

    from datetime import timezone
    UTC = timezone.utc


class KnowledgeGraphDB(BaseKnowledgeGraphDB):
    """
    A knowledge graph database that supports fact storage and retrieval.

    This class has the following methods:
        - `generate_fact_id()`
        - `add_fact()`
        - `get_facts_about()`
        - `clear()`
        - `remove_facts_about()`
        - `is_relationship()`
        - `remove_fact()`
    """

    DEFAULT_DEPTH_EXCEPTIONS = {'is_subject_of': 0, 'is_object_of': 0}

    def __init__(
            self, 
            database: Optional[GraphDB] = None, 
            graph_backend: Optional[GraphBackend] = None, 
            relationship_predicates: Optional[set] = None, 
            depth_exceptions: Optional[dict] = None,
            fact_id_key: Optional[tuple] = ("subject", "predicate", "obj", "source")
        ):
        """
        Initializes the knowledge graph.

        Args:
            graph_db (GraphDB): The database backend for structured storage.
            graph_backend (GraphBackend): The graph backend for relationships.
            relationship_predicates (set, optional): The relationship predicates to use. If not provided, the default relationship predicates will be used.
        """
        if database is None:
            database = DuckDBGraphDB()
        
        if graph_backend is None:
            graph_backend = NetworkXGraphBackend(depth_exceptions=depth_exceptions or self.DEFAULT_DEPTH_EXCEPTIONS)
        else:
            if depth_exceptions:
                graph_backend = graph_backend.copy(depth_exceptions=depth_exceptions)

        super().__init__(
            database=database, 
            graph_backend=graph_backend, 
            relationship_predicates=relationship_predicates,
            fact_id_key=fact_id_key
        )

        # make sure table exists
        if self.knowledge_table_name not in self.database.tables:
            raise RuntimeError(f"Table {self.knowledge_table_name} does not exist. Please create it first.")
    
    def generate_fact_id(self, **kwargs) -> str:
        return self._generate_fact_id(**kwargs)

    def add_fact(
        self, 
        subject: str, 
        predicate: str, 
        obj: str, 
        fact_id: Optional[str] = None,
        source: Optional[str] = None, 
        timestamp: Optional[datetime] = None, 
        add_to_graph: Optional[bool] = None
    ) -> datetime:
        """
        Adds a fact to the database and optionally to the graph backend.

        - Calls `add_fact_to_db()` first.
        - Calls `add_fact_to_graph()` if needed.

        Args:
            subject (str): The subject of the fact.
            predicate (str): The predicate of the fact.
            obj (str): The object of the fact.
            fact_id (str): The ID of the fact.
            source (str, optional): The source of the fact.
            timestamp (datetime, optional): The timestamp of the fact.


            add_to_graph (Optional[bool]): Determines whether to store in `GraphBackend`.

                - `True`: Always store in `GraphBackend`.
                - `False`: Never store in `GraphBackend`.
                - `None`: Store in `GraphBackend` only if `predicate` is in `relationship_predicates`.

        Returns:
            datetime: The timestamp when the fact was stored.
        """
        if not isinstance(subject, str):
            raise TypeError(f"Subject should be a string but it is a {subject} of type {type(subject)}.")
        if not isinstance(obj, str):
            raise TypeError(f"Object should be a string but it is a {obj} of type {type(obj)}.")
        if not isinstance(predicate, str):
            raise TypeError(f"Predicate should be a string but it is a {predicate} of type {type(predicate)}.")
            
        fact_id, timestamp = self._add_fact_to_db(fact_id=fact_id, subject=subject, predicate=predicate, obj=obj, source=source, timestamp=timestamp)

        if add_to_graph is None:
            add_to_graph = self.is_relationship(subject=subject, predicate=predicate, obj=obj)

        if add_to_graph:
            self._add_fact_to_graph(subject=subject, predicate=predicate, obj=obj, fact_id=fact_id, source=source, timestamp=timestamp)

        return fact_id, timestamp

    def get_facts_about(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        source: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        depth: Optional[int] = 1,
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
        include_fact_id: bool = False,
        include_source: bool = False,
        include_timestamp: bool = False,
        output_type: Literal["tuple", "dict", "fact"] = "tuple"
    ) -> List[Union[Tuple, Dict, Fact]]:
        """
        Retrieves facts from the database based on the provided filters.

        Args:
            subject (str, optional): The subject of the fact.   
            predicate (str, optional): The predicate of the fact.
            obj (str, optional): The object of the fact.
            source (str, optional): The source of the fact.
            start_time (datetime, optional): The start time of the fact.
            end_time (datetime, optional): The end time of the fact.
            include_fact_id (bool, optional): If `True`, include the fact ID in the results.
            include_source (bool, optional): If `True`, include the source in the results.
            include_timestamp (bool, optional): If `True`, include the timestamp in the results.
            output_type (Literal["tuple", "dict", "fact"], optional): The type of output to return.

        Returns:
            List[Union[Tuple, Dict, Fact]]: A list of facts matching the conditions.
        """

        # Query Conditions
        query_conditions = {}
        if subject is not None:
            query_conditions["subject_lower"] = subject.lower()
        if predicate is not None:
            query_conditions["predicate_lower"] = predicate.lower()
        if obj is not None:
            query_conditions["object_lower"] = obj.lower()
        if start_time is not None and end_time is not None:
            query_conditions["timestamp"] = (start_time, end_time)
        elif start_time is not None:
            query_conditions["timestamp"] = (start_time, None)
        elif end_time is not None:
            query_conditions["timestamp"] = (None, end_time)
        if source is not None:
            query_conditions["source"] = source

        # Include Fields
        include_fields = ['fact_id'] if include_fact_id else []
        include_fields += ["subject", "predicate", "object"]
        include_fields += ["source"] if include_source else []
        include_fields += ["timestamp"] if include_timestamp else []

        subject_index = include_fields.index("subject")
        predicate_index = include_fields.index("predicate")
        object_index = include_fields.index("object")

        facts_as_tuples = self.database.get_rows(
            table_name=self.knowledge_table_name,
            include_fields=include_fields,
            **query_conditions
        )

        # get the facts from the graph backend
        results = {}

        def edge_to_fact(_source, _edge, _target, _metadata):
            result = {
                'fact_id': _metadata.get("fact_id"),
                'subject': _source,
                'predicate': _edge,
                'object': _target,
                'source': _metadata.get("fact_source"),
                'timestamp': _metadata.get("timestamp")
            }
            return tuple(result[field] for field in include_fields)
        
        format_mapper = {
            'subject': 'source',
            'predicate': 'edge',
            'object': 'target',
            'source': 'fact_source',
            'timestamp': 'timestamp',
            'fact_id': 'fact_id'
        }

        graph_output_format = tuple(format_mapper.get(field, field) for field in include_fields)

        for fact_as_tuple in facts_as_tuples:
            subject = fact_as_tuple[subject_index]
            predicate = fact_as_tuple[predicate_index]
            obj = fact_as_tuple[object_index]
            subject_neighbours = self.graph_backend.get_neighbours(node=subject, depth=depth, direction=direction, output_format=graph_output_format, return_as_dict=True)
            obj_neighbours = self.graph_backend.get_neighbours(node=obj, depth=depth, direction=direction, output_format=graph_output_format, return_as_dict=True)

            results[(subject, predicate, obj)] = fact_as_tuple
            results.update(subject_neighbours)
            results.update(obj_neighbours)

        results = list(results.values())

        if output_type == "tuple":
            return results
        elif output_type == "dict":
            return self._facts_as_dict(facts_as_tuples=results, include_fields=include_fields)
        elif output_type == "fact":
            return self._facts_as_fact(facts_as_tuples=results, include_fields=include_fields)
        
    def clear(self):
        """
        Clears all stored knowledge from both the database and the graph backend.
        """
        rows = self.database.delete_all_rows(table_name=self.knowledge_table_name)
        edges = self.graph_backend.clear()

        return rows, edges
    
    def remove_fact(self, fact_id: str):
        """
        Removes a fact from the database and the graph backend.
        """
        subject, predicate, obj = self.database.get_one(table_name=self.knowledge_table_name, fact_id=fact_id, include_fields=["subject", "predicate", "object"])
        self.database.delete_rows(table_name=self.knowledge_table_name, fact_id=fact_id)
        self.graph_backend.remove_edge(source=subject, target=obj, edge=predicate)

    def remove_facts_about(
        self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None,
        source: Optional[str] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ):
        """
        Removes all facts matching the given conditions.

        Args:
            subject (Optional[str], optional): The subject to filter by.
            predicate (Optional[str], optional): The predicate to filter by.
            obj (Optional[str], optional): The object to filter by.
        """
        subject_lower = subject.lower() if subject else None
        predicate_lower = predicate.lower() if predicate else None
        obj_lower = obj.lower() if obj else None

        facts = self.database.get_rows(
            table_name=self.knowledge_table_name,
            include_fields=["subject", "predicate", "object"],
            subject_lower=subject_lower,
            predicate_lower=predicate_lower,
            object_lower=obj_lower,
            source=source,
            start_time=start_time,
            end_time=end_time

        )

        for subject_value, predicate_value, object_value in facts:
            self.graph_backend.remove_edge(subject=subject_value, obj=object_value, edge=predicate_value)

        self.database.delete_rows(
            table_name=self.knowledge_table_name, 
            subject_lower=subject_lower,
            predicate_lower=predicate_lower,
            object_lower=obj_lower,
            source=source,
            start_time=start_time,
            end_time=end_time
        )

    def is_relationship(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj: Optional[str] = None) -> bool:
        """

        Checks if the fact is a relationship.

        Args:
            subject: The subject of the fact.
            predicate: The predicate of the fact.
            obj: The object of the fact.


        Returns:
            bool: True if the fact is a relationship, False otherwise.
        """
        if not isinstance(predicate, str):
            raise TypeError(f"Predicate: {predicate} should be a string but it is of the type {type(predicate)}.")
        return predicate in self.relationship_predicates
    