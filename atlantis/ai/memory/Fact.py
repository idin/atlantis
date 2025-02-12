from datetime import datetime, date
from typing import Optional, Dict, Tuple, Union, Literal, Any
from dataclasses import dataclass

try:
    from datetime import UTC
except ImportError:

    from datetime import timezone
    UTC = timezone.utc


@dataclass(frozen=True)
class Fact:
    """
    A structured representation of a fact in the knowledge graph.
    This class has the following methods:
    - to_tuple
    - to_dict
    - to_type
    - from_tuple
    - from_dict
    - from_type
    - convert
    - __contains__
    - equals
    - __eq__
    - __hash__
    """

    id: str
    subject: str
    predicate: str
    obj: str
    source: Optional[str] = None
    timestamp: Optional[Union[datetime, date]] = None

    def __getitem__(self, item: str) -> Union[str, datetime, date]:
        return getattr(self, item)

    @property
    def date(self) -> date:
        return self.timestamp.date() if isinstance(self.timestamp, datetime) else self.timestamp
    
    def copy(self, lower: bool = False, **kwargs) -> "Fact":
        # the kwargs overwrites the existing values
        d = self.__dict__.copy()
        d.update(kwargs)
        if lower:
            d = {key: value.lower() if isinstance(value, str) else value for key, value in d.items()}
        return self.__class__(**d)

    def to_tuple(self, include_source: bool = True, include_timestamp: bool = False, timestamp_as_date: bool = False, lower: bool = False) -> Tuple:
        """

        Converts the fact into a tuple format.

        Args:
            include_source (bool): Whether to include the source field. Defaults to True.
            include_timestamp (bool): Whether to include the timestamp field. Defaults to False.

        Returns:
            Tuple: The fact in tuple format.
        """
        result = (self.id, self.subject, self.predicate, self.obj)
        if include_source:
            result += (self.source,)
        if include_timestamp:
            result += (self.date,) if timestamp_as_date else (self.timestamp,)

        if lower:
            result = tuple(item.lower() if isinstance(item, str) else item for item in result)
        return result
    
    as_tuple = to_tuple

    def to_dict(self, include_source: bool = False, include_timestamp: bool = False, timestamp_as_date: bool = False, lower: bool = False) -> Dict[str, Union[str, datetime]]:
        """
        Converts the fact into a dictionary format.

        Args:
            include_source (bool): Whether to include the source field. Defaults to False.
            include_timestamp (bool): Whether to include the timestamp field. Defaults to False.

        Returns:
            Dict[str, Union[str, datetime]]: The fact in dictionary format.
        """
        result = {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "obj": self.obj
        }
        if include_source:
            result["source"] = self.source
        if include_timestamp:
            result["timestamp"] = self.date if timestamp_as_date else self.timestamp

        if lower:
            result = {key: value.lower() if isinstance(value, str) else value for key, value in result.items()}
        return result

    as_dict = to_dict

    @classmethod
    def from_tuple(
        cls,
        tuple_data: Tuple[str, str, str, Optional[str], Optional[datetime]],
        include_source: bool = True,
        include_timestamp: bool = False,
        timestamp_as_date: bool = False
    ) -> "Fact":
        """
        Creates a Fact instance from a tuple.


        Args:
            tuple_data (Tuple): The tuple data representing a fact.
            include_source (bool): Whether the tuple includes a source field.
            include_timestamp (bool): Whether the tuple includes a timestamp field.

        Returns:
            Fact: A new Fact instance.
        """
        fact_id, subject, predicate, obj = tuple_data[:4]

        if include_source and include_timestamp:
            source = tuple_data[4]
            timestamp = tuple_data[5]
        elif include_source:
            source = tuple_data[4]
            timestamp = None
        elif include_timestamp:
            timestamp = tuple_data[4]
            source = None
        else:
            source = None
            timestamp = None

        if timestamp_as_date and timestamp:
            timestamp = timestamp.date()

        return cls(
            id=fact_id,
            subject=subject,
            predicate=predicate,
            obj=obj,
            source=source,
            timestamp=timestamp
        )

    @classmethod
    def from_dict(
        cls,
        dict_data: Dict[str, Union[str, datetime]],
        include_source: bool = True,
        include_timestamp: bool = False,
        timestamp_as_date: bool = False
    ) -> "Fact":
        """
        Creates a Fact instance from a dictionary.


        Args:
            dict_data (Dict): The dictionary data representing a fact.
            include_source (bool): Whether to include the source field.
            include_timestamp (bool): Whether to include the timestamp field.

        Returns:
            Fact: A new Fact instance.
        """
        dict_data = dict_data.copy()
        source = dict_data.get("source") if include_source else None
        timestamp = dict_data.get("timestamp") if include_timestamp else None
        if timestamp_as_date and timestamp:
            timestamp = timestamp.date()

        return cls(
            id=dict_data["id"],
            subject=dict_data["subject"],
            predicate=dict_data["predicate"],
            obj=dict_data["obj"],
            source=source,
            timestamp=timestamp
        )

    @classmethod
    def from_any_type(
        cls,
        data: Union[Tuple, Dict, "Fact"],
        include_source: bool = True,
        include_timestamp: bool = False,
        timestamp_as_date: bool = False,
        lower: bool = False
    ) -> "Fact":
        if isinstance(data, Fact):
            return data
        elif isinstance(data, tuple):
            return cls.from_tuple(data, include_source=include_source, include_timestamp=include_timestamp, timestamp_as_date=timestamp_as_date, lower=lower)
        elif isinstance(data, dict):
            return cls.from_dict(data, include_source=include_source, include_timestamp=include_timestamp, timestamp_as_date=timestamp_as_date, lower=lower)
        else:
            raise ValueError("Invalid data type")
        
    def to_type(
        self,
        to_type: Literal["tuple", "dict", "fact"] = "tuple",
        include_source: bool = True,
        include_timestamp: bool = False,
        timestamp_as_date: bool = False,
        lower: bool = False
    ) -> Union[Tuple, Dict, "Fact"]:
        if to_type == "tuple":
            return self.to_tuple(include_source=include_source, include_timestamp=include_timestamp, timestamp_as_date=timestamp_as_date, lower=lower) 


        elif to_type == "dict":
            return self.to_dict(include_source=include_source, include_timestamp=include_timestamp, timestamp_as_date=timestamp_as_date, lower=lower)
        
        elif to_type == "fact":
            if lower:
                return self.copy(lower=True)
            return self
        else:
            raise ValueError("Invalid type")
        
    @classmethod
    def convert(
        cls,
        data: Union[Tuple, Dict, "Fact"],
        to_type: Literal["tuple", "dict", "fact"] = "tuple",
        include_source: bool = True,
        include_timestamp: bool = False,
        timestamp_as_date: bool = False,
        lower: bool = False
    ) -> Union[Tuple, Dict, "Fact"]:
        if isinstance(data, Fact):
            return data.to_type(to_type, include_source=include_source, include_timestamp=include_timestamp, timestamp_as_date=timestamp_as_date, lower=lower)


        from_type = "tuple" if isinstance(data, tuple) else "dict" if isinstance(data, dict) else "fact"
        if from_type == to_type:
            return data
        else:
            fact = cls.from_any_type(data, from_type=from_type, include_source=include_source, include_timestamp=include_timestamp, timestamp_as_date=timestamp_as_date, lower=lower)
            return fact.to_type(to_type, include_source=include_source, include_timestamp=include_timestamp, timestamp_as_date=timestamp_as_date, lower=lower)
            
    def __contains__(self, item) -> bool:
        if isinstance(item, str):
            strings = set(x for x in self.to_tuple(include_source=False, include_timestamp=False, lower=True) if isinstance(x, str))
            if item.lower() in strings:
                return True

            if isinstance(self.subject, Fact):
                if item in self.subject:
                    return True
            if isinstance(self.obj, Fact):
                if item in self.obj:
                    return True
            # if the item looks like a datetime or date, convert it to datetime or date
            # make sure if it's like a date, it's just converted to date, not datetime
            # first try timestamp though
            if self.timestamp:
                if item == self.timestamp.isoformat():
                    return True
                if item == str(self.timestamp):
                    return True
                if item == self.date.isoformat():
                    return True
                if item == str(self.date):
                    return True
            
            return False
        
        elif isinstance(item, (Fact, tuple, dict)):
            item = self.from_any_type(item, include_source=False, include_timestamp=False, lower=True)
            if isinstance(self.subject, Fact):
                if item in self.subject:
                    return True
                if item.equals(self.subject, include_source=False, include_timestamp=False):
                    return True
            if isinstance(self.obj, Fact):
                if item in self.obj:
                    return True
                if item.equals(self.obj, include_source=False, include_timestamp=False):
                    return True
            return False
        
        elif isinstance(item, datetime):
            if isinstance(self.timestamp, datetime):
                return item == self.timestamp
            else:
                return item == self.date
        
        elif isinstance(item, date):
            # compare it with date
            return item == self.date      

        else:
            raise TypeError("Invalid item type")

    def equals(self, other: Union["Fact", Tuple, Dict], include_source: bool = True, include_timestamp: bool = False, lower: bool = True) -> bool:
        if not isinstance(other, (Fact, tuple, dict)):
            raise ValueError(f"{other} of type {type(other)} is not a Fact, tuple or dict")
        other = self.from_any_type(other, include_source=include_source, include_timestamp=include_timestamp, lower=lower)
        self_tuple = self.to_tuple(include_source=include_source, include_timestamp=include_timestamp, lower=lower)

        other_tuple = other.to_tuple(include_source=include_source, include_timestamp=include_timestamp, lower=lower)
        return self_tuple == other_tuple

    def __eq__(self, other: "Fact") -> bool:

        """
        Compares two Fact instances for equality.
        Args:
            other (Fact): The other Fact instance.
        Returns:
            bool: True if the facts are equal, False otherwise.
        """
        return self.equals(other, include_source=True, include_timestamp=False, lower=True)

    def __hash__(self) -> int:
        """
        Returns the hash of the fact.

        Returns:
            int: The hash of the fact.
        """
        return hash(self.to_tuple(include_source=True, include_timestamp=False))
