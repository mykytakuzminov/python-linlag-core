from __future__ import annotations


class Matrix:
    """
    A class representing a 2D numerical matrix with support for common arithmetic
    operations and statistical methods.
    """
    def __init__(self, data: list[list[float]]) -> None:
        """
        Initialize the Matrix with a 2D list of floats.

        Args:
            data: A list of lists containing float values.
        """
        self.data = data

    @classmethod
    def zeros(cls, rows: int, cols: int) -> Matrix:
        """
        Create a matrix of the specified shape filled with 0.0.

        Args:
            rows: Number of rows.
            cols: Number of columns.
        """
        new_data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        return cls(new_data)

    @classmethod
    def ones(cls, rows: int, cols: int) -> Matrix:
        """
        Create a matrix of the specified shape filled with 1.0.

        Args:
            rows: Number of rows.
            cols: Number of columns.
        """
        new_data = [[1.0 for _ in range(cols)] for _ in range(rows)]
        return cls(new_data)

    @classmethod
    def identity(cls, n: int) -> Matrix:
        """
        Create an n x n identity matrix.

        Args:
            n: Dimension of the square matrix.
        """
        new_data = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]
        return cls(new_data)

    def __str__(self) -> str:
        """Return a formatted string representation of the matrix."""
        matrix = []
        for row in self.data:
            str_row = "".join([f"{e:^5.1f}" for e in row])
            matrix.append("| " + str_row + " |")
        return "\n".join(matrix)

    def __getitem__(self, el: tuple[int, int]) -> float:
        """
        Retrieve value at index (row, col).

        Args:
            el: A tuple containing (row_index, col_index).
        """
        row, col = el
        self._check_index(row, col)
        return self.data[row][col]

    def __setitem__(self, el: tuple[int, int], val: float) -> None:
        """
        Set value at index (row, col).

        Args:
            el: A tuple containing (row_index, col_index).
            val: The value to set.
        """
        row, col = el
        self._check_index(row, col)
        self.data[row][col] = val

    def __add__(self, obj: Matrix | int | float) -> Matrix:
        """
        Add matrix to another matrix or a scalar.

        Args:
            obj: Scalar (int/float) or another Matrix.

        Raises:
            ValueError: If dimensions do not match for matrix addition.
        """
        new_data: list[list[float]]
        if isinstance(obj, (int, float)):
            new_data = [
                [a + obj for a in row] for row in self.data
            ]

        elif isinstance(obj, Matrix):
            if self.shape != obj.shape:
                raise ValueError(f"Matrix addition requires identical shapes. Got {self.shape} and {obj.shape}.")

            new_data = [
                [a + b for a, b in zip(row_a, row_b)]
                for row_a, row_b in zip(self.data, obj.data)
            ]

        return Matrix(new_data)

    def __sub__(self, obj: Matrix | int | float) -> Matrix:
        """
        Subtract another matrix or a scalar from this matrix.

        Args:
            obj: Scalar (int/float) or another Matrix.

        Raises:
            ValueError: If dimensions do not match.
        """
        new_data: list[list[float]]
        if isinstance(obj, (int, float)):
            new_data = [
                [a - obj for a in row] for row in self.data
            ]

        elif isinstance(obj, Matrix):
            if self.shape != obj.shape:
                raise ValueError(f"Matrix subtraction requires identical shapes. Got {self.shape} and {obj.shape}.")

            new_data = [
                [a - b for a, b in zip(row_a, row_b)]
                for row_a, row_b in zip(self.data, obj.data)
            ]

        return Matrix(new_data)

    def __eq__(self, obj: object) -> bool:
        """Check equality against another object."""
        if not isinstance(obj, Matrix):
            return False

        if self.shape != obj.shape:
            return False

        return all(
            a == b
            for row_a, row_b in zip(self.data, obj.data)
            for a, b in zip(row_a, row_b)
        )

    def __mul__(self, obj: Matrix | int | float) -> Matrix:
        """
        Perform scalar multiplication or dot product matrix multiplication.

        Args:
            obj: Scalar (int/float) or another Matrix.

        Raises:
            ValueError: If dimensions are incompatible for matrix multiplication.
        """
        new_data: list[list[float]]
        if isinstance(obj, (int, float)):
            new_data = [
                [a * obj for a in row] for row in self.data
            ]

        elif isinstance(obj, Matrix):
            if self.shape[1] != obj.shape[0]:
                raise ValueError(f"Incompatible dimensions for multiplication: {self.shape} and {obj.shape}.")

            cols_b = obj.transpose().data

            new_data = [
                [sum(a * b for a, b in zip(row_a, col_b)) for col_b in cols_b]
                for row_a in self.data
            ]

        return Matrix(new_data)

    def __rmul__(self, obj: int | float) -> Matrix:
        """Support reverse scalar multiplication."""
        return self * obj

    def __pow__(self, power: int | float) -> Matrix:
        """Perform element-wise exponentiation."""
        new_data = [
            [a ** power for a in row] for row in self.data
        ]

        return Matrix(new_data)

    @property
    def data(self) -> list[list[float]]:
        """Return the underlying 2D list data."""
        return self._data

    @data.setter
    def data(self, new_data: list[list[float]]) -> None:
        """
        Set and validate the matrix data.

        Raises:
            ValueError: If matrix is empty or has inconsistent row lengths.
        """
        if not len(new_data):
            raise ValueError("Matrix data cannot be empty.")

        cols = len(new_data[0])
        for i, row in enumerate(new_data):
            if len(row) != cols:
                raise ValueError(f"Row {i} has {len(row)} columns, expected {cols}.")

        self._data = new_data

    @property
    def shape(self) -> tuple[int, int]:
        """Return the matrix dimensions as (rows, cols)."""
        rows = len(self.data)
        cols = len(self.data[0])
        return rows, cols

    def transpose(self) -> Matrix:
        """Return the transpose of the matrix."""
        new_data = zip(*self.data)
        t_rows = [list(row) for row in new_data]
        return Matrix(t_rows)

    def sum(self) -> float:
        """Calculate the sum of all elements in the matrix."""
        return sum([sum(row) for row in self.data])

    def mean(self) -> float:
        """Calculate the arithmetic mean of all elements in the matrix."""
        return self.sum() / (self.shape[0] * self.shape[1])

    def _check_index(self, row: int, col: int) -> None:
        """
        Validate indices for getitem/setitem.

        Raises:
            IndexError: If index is out of bounds or negative.
        """
        if row < 0 or col < 0:
            raise IndexError(f"Negative index ({row}, {col}) is not allowed.")

        if row >= self.shape[0] or col >= self.shape[1]:
            raise IndexError(f"Index ({row}, {col}) is out of bounds for matrix with shape {self.shape}.")

