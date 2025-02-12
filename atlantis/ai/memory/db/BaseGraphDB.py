from typing import Dict, Optional, List, Tuple, Union

class BaseGraphDB:
    """
    Abstract database class that provides an interface for creating tables,
    adding data, retrieving data, and deleting data.
    """

    def _create_table(self, table_name: str, columns: Dict[str, str]):
        """
        Creates a table with the given name and columns.
        :param table_name: Name of the table to create.
        :param columns: Dictionary mapping column names to data types.
                        Acceptable types: "text", "integer", "float", "boolean", "date", "datetime".
        """
        self._columns = columns
        raise NotImplementedError

    def _add_row(self, table_name: str, **kwargs):
        """
        Adds a row to the specified table.
        :param table_name: The name of the table to insert data into.
        :param kwargs: Column values to insert.
        """
        raise NotImplementedError
    
    def _get_one(self, table_name: str, include_fields: Optional[List[str]] = None, **kwargs) -> Optional[Tuple]:
        """
        Retrieves a single row matching the given conditions.
        :param table_name: The name of the table to query.
        :param include_fields: Optional list of fields to include in the returned tuples.
        :param kwargs: Conditions as key-value pairs.
        :return: A single tuple representing the row.
        """

        raise NotImplementedError

    def _get_rows(self, table_name: str, include_fields: Optional[List[str]] = None, limit: Optional[int] = None, **kwargs) -> List[Tuple]:
        """
        Retrieves all rows matching the given conditions.
        :param table_name: The name of the table to query.
        :param include_fields: Optional list of fields to include in the returned tuples.
        :param limit: Optional maximum number of rows to return.
        :param kwargs: Conditions as key-value pairs (supports values and ranges).
        :return: List of tuples representing rows.

        """
        raise NotImplementedError

    def _delete_rows(self, table_name: str, limit: Optional[int] = None, **kwargs) -> Union[int, List[Tuple], None]:
        """
        Deletes all rows matching the given conditions.
        :param table_name: The name of the table to delete data from.
        :param limit: Optional maximum number of rows to delete.
        :param kwargs: Conditions as key-value pairs (supports values and ranges).
        """
        raise NotImplementedError

    def _delete_all_rows(self, table_name: str):
        """
        Clears all rows from the specified table.
        :param table_name: The name of the table to clear.
        """
        raise NotImplementedError
    
    def _get_size_in_bytes(self, table_name: str) -> int:
        """
        Returns the size of the database in bytes.
        :param table_name: The name of the table to get the size of.
        :return: The size of the database in bytes.
        """
        raise NotImplementedError

    def _get_number_of_rows(self, table_name: str) -> int:
        """
        Returns the number of rows in the specified table.
        :param table_name: The name of the table to get the number of rows of.
        :return: The number of rows in the specified table.
        """
        raise NotImplementedError
