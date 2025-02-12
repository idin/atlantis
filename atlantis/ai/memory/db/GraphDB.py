from .BaseGraphDB import BaseGraphDB
from typing import Dict, Optional, List, Tuple, Union


class GraphDB(BaseGraphDB):
    """
    A graph database that uses a graph backend to store and retrieve data.
    This class is the public API for the graph database that uses the abstract base class.
    This classprovides the standard methods that are not supposed to be changed by the subclasses.
    """
    def __init__(self, **kwargs):
        self.tables = {}

    def create_table(self, table_name: str, columns: Dict[str, str]):
        """
        Creates a table with the given name and columns.
        :param table_name: Name of the table to create.
        :param columns: Dictionary mapping column names to data types.
                        Acceptable types: "text", "integer", "float", "boolean", "date", "datetime".
        """
        table = self._create_table(table_name=table_name, columns=columns)
        self.tables[table_name] = {'columns': columns, 'number_of_rows_added': 0, 'number_of_rows_accessed': 0, 'number_of_rows_deleted': 0}
        if len(self.tables) == 0:
            raise RuntimeError("No tables created. Please create at least one table.")
        return table

    def add_row(self, table_name: str, **kwargs):
        """
        Adds a row to the specified table.
        :param table_name: The name of the table to insert data into.
        :param kwargs: Column values to insert.
        """
        row = self._add_row(table_name=table_name, **kwargs)
        self.tables[table_name]['number_of_rows_added'] += 1
        return row
    
    def get_one(self, table_name: str, include_fields: Optional[List[str]] = None, **kwargs) -> Optional[Tuple]:
        """
        Retrieves a single row matching the given conditions.
        :param table_name: The name of the table to query.
        :param include_fields: Optional list of fields to include in the returned tuples.
        :param kwargs: Conditions as key-value pairs.
        :return: A single tuple representing the row.
        """
        return self._get_one(table_name=table_name, include_fields=include_fields, **kwargs)

    def get_rows(self, table_name: str, include_fields: Optional[List[str]] = None, limit: Optional[int] = None, **kwargs) -> List[Tuple]:
        """
        Retrieves all rows matching the given conditions.
        :param table_name: The name of the table to query.
        :param include_fields: Optional list of fields to include in the returned tuples.
        :param limit: Optional maximum number of rows to return.
        :param kwargs: Conditions as key-value pairs (supports values and ranges).
        :return: List of tuples representing rows.

        """
        rows = self._get_rows(table_name=table_name, include_fields=include_fields, limit=limit, **kwargs)
        if table_name not in self.tables:
            raise KeyError(f"Table {table_name} does not exist. Other tables are: {self.tables.keys()}")
        self.tables[table_name]['number_of_rows_accessed'] += len(rows)
        return rows

    def delete_rows(self, table_name: str, limit: Optional[int] = None, **kwargs) -> Union[int, List[Tuple], None]:
        """
        Deletes all rows matching the given conditions.
        :param table_name: The name of the table to delete data from.
        :param limit: Optional maximum number of rows to delete.
        :param kwargs: Conditions as key-value pairs (supports values and ranges).
        """
        rows = self._delete_rows(table_name=table_name, limit=limit, **kwargs)
        if isinstance(rows, int):
            self.tables[table_name]['number_of_rows_deleted'] += rows
        elif isinstance(rows, list):
            self.tables[table_name]['number_of_rows_deleted'] += len(rows)
        return rows

    def delete_all_rows(self, table_name: str):
        """
        Clears all rows from the specified table.
        :param table_name: The name of the table to clear.
        """
        rows = self._delete_all_rows(table_name=table_name)
        if isinstance(rows, int):
            self.tables[table_name]['number_of_rows_deleted'] += rows
        elif isinstance(rows, list):
            self.tables[table_name]['number_of_rows_deleted'] += len(rows)
        return rows
    
    def get_size_in_bytes(self, table_name: str) -> int:
        """
        Returns the size of the database in bytes.
        :param table_name: The name of the table to get the size of.
        :return: The size of the database in bytes.
        """
        return self._get_size_in_bytes(table_name=table_name)

    def get_number_of_rows(self, table_name: str) -> int:
        """
        Returns the number of rows in the specified table.
        :param table_name: The name of the table to get the number of rows of.
        :return: The number of rows in the specified table.
        """
        return self._get_number_of_rows(table_name=table_name)
