import duckdb
import os
from typing import Dict, Optional, List, Tuple
from .GraphDB import GraphDB

class DuckDBGraphDB(GraphDB):
    """
    DuckDB implementation of GraphDB, providing an efficient in-memory or file-backed database.
    """

    TYPE_MAPPING = {
        "text": "TEXT",
        "integer": "INTEGER",
        "float": "REAL",
        "boolean": "BOOLEAN",
        "date": "DATE",
        "datetime": "TIMESTAMP"
    }

    def __init__(self, db_path: str = ":memory:"):
        """
        Initializes a DuckDB connection.

        Args:
            db_path (str): Path to the DuckDB database file. Defaults to in-memory.
        """
        self.db_path = db_path
        self.conn = duckdb.connect(self.db_path)
        super().__init__()

    def _create_table(self, table_name: str, columns: Dict[str, str], primary_keys: Optional[List[str]] = None):
        """
        Creates a table in DuckDB.

        Args:
            table_name (str): Name of the table.
            columns (Dict[str, str]): Dictionary mapping column names to data types.
            primary_keys (Optional[List[str]]): List of columns to be set as PRIMARY KEY.
        """
        column_definitions = []
        for col_name, col_type in columns.items():
            if col_type.lower() not in self.TYPE_MAPPING:
                raise ValueError(f"Unsupported column type: {col_type}")
            column_definitions.append(f"{col_name} {self.TYPE_MAPPING[col_type.lower()]}")

        if primary_keys:
            column_definitions.append(f"PRIMARY KEY ({', '.join(primary_keys)})")

        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_definitions)})"
        self.conn.execute(query)

    def _add_row(self, table_name: str, **kwargs):
        """
        Inserts a row into the specified table.

        Args:
            table_name (str): The name of the table.
            kwargs: Column values to insert.
        """
        columns = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?" for _ in kwargs])
        values = tuple(kwargs.values())

        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.conn.execute(query, values)

    def _get_one(self, table_name: str, include_fields: Optional[List[str]] = None, **kwargs) -> Optional[Tuple]:
        """
        Retrieves a single row matching the given conditions.
        """
        return self._get_rows(table_name, include_fields, limit=1, **kwargs)[0]


    def _get_rows(self, table_name: str, include_fields: Optional[List[str]] = None, limit: Optional[int] = None, **kwargs) -> List[Tuple]:
        """
        Retrieves all rows matching the given conditions, including range-based queries.

        Args:
            table_name (str): The name of the table.
            include_fields (Optional[List[str]]): Columns to retrieve. If None, retrieves all.
            limit (Optional[int]): Maximum number of rows to return.
            kwargs: Filter conditions (supports exact values and range queries).

        Returns:
            List[Tuple]: Matching rows.
        """
        conditions = []
        values = []

        for key, value in kwargs.items():
            if isinstance(value, tuple) and len(value) == 2:
                start, end = value
                if start is None and end is None:
                    continue  # Ignore empty range
                elif start is None:
                    conditions.append(f"{key} <= ?")
                    values.append(end)
                elif end is None:
                    conditions.append(f"{key} >= ?")
                    values.append(start)
                else:
                    conditions.append(f"{key} BETWEEN ? AND ?")
                    values.extend([start, end])
            else:
                conditions.append(f"{key} = ?")
                values.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        fields_str = ", ".join(include_fields) if include_fields else "*"

        query = f"SELECT {fields_str} FROM {table_name} WHERE {where_clause}"
        if limit:
            query += f" LIMIT {limit}"

        return self.conn.execute(query, values).fetchall()

    def _delete_rows(self, table_name: str, limit: Optional[int] = None, **kwargs):
        """
        Deletes rows matching conditions, supporting range deletions.

        Args:
            table_name (str): Name of the table.
            kwargs: Conditions (supports values and ranges). Ranges can be:
                - A tuple of (min, max) to define a range.
                - A tuple of (min, None) or (None, max) for open-ended ranges.
            limit (Optional[int]): Maximum number of rows to delete. This is not supported by DuckDB and will be ignored
        """
        conditions = []
        values = []

        for key, value in kwargs.items():
            if isinstance(value, tuple) and len(value) == 2:
                start, end = value
                if start is None and end is None:
                    continue
                elif start is None:
                    conditions.append(f"{key} <= ?")
                    values.append(end)
                elif end is None:
                    conditions.append(f"{key} >= ?")
                    values.append(start)
                else:
                    conditions.append(f"{key} BETWEEN ? AND ?")
                    values.extend([start, end])
            else:
                conditions.append(f"{key} = ?")
                values.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"DELETE FROM {table_name} WHERE {where_clause}"

        try:
            self.conn.execute(query, values)
        except Exception as e:
            print(f"Error deleting rows: {e}, query: {query}, values: {values}")
            raise e

    def _delete_all_rows(self, table_name: str):
        """
        Clears all rows from the specified table.

        Args:
            table_name (str): The name of the table.
        """
        self.conn.execute(f"DELETE FROM {table_name}")

    def _get_size_in_bytes(self) -> int:
        """
        Returns the total size of the database in bytes.

        Returns:
            int: Database size in bytes.
        """
        if self.db_path == ":memory:":
            return self.conn.execute("PRAGMA memory_used").fetchone()[0]  # Corrected for in-memory DB
        return os.path.getsize(self.db_path)  # Get file size for disk-based DB

    def _get_number_of_rows(self, table_name: str) -> int:
        """
        Returns the number of rows in the specified table.
        """
        return self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

