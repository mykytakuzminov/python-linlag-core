from __future__ import annotations


class Matrix:
    ""
    def __init__(self, data: list[list[float]]) -> None:
        ""
        self.data = data

    @classmethod
    def zeros(cls, rows: int, cols: int) -> Matrix:
        ""
        new_data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        return cls(new_data)

    @classmethod
    def ones(cls, rows: int, cols: int) -> Matrix:
        ""
        new_data = [[1.0 for _ in range(cols)] for _ in range(rows)]
        return cls(new_data)

    @classmethod
    def identity(cls, n: int) -> Matrix:
        ""
        new_data = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]
        return cls(new_data)

    def __str__(self) -> str:
        ""
        matrix = []
        for row in self.data:
            str_row = "".join([f"{e:^5.1f}" for e in row])
            matrix.append("| " + str_row + " |")
        return "\n".join(matrix)

    def __getitem__(self, el: tuple[int, int]) -> float:
        ""
        row, col = el
        self._check_index(row, col)
        return self.data[row][col]

    def __setitem__(self, el: tuple[int, int], val: float) -> None:
        ""
        row, col = el
        self._check_index(row, col)
        self.data[row][col] = val

    def __add__(self, obj: Matrix) -> Matrix:
        ""
        if self.shape != obj.shape:
            raise ValueError(f"Matrix addition requires identical shapes. Got {self.shape} and {obj.shape}.")

        new_data = [
            [a + b for a, b in zip(row_a, row_b)]
            for row_a, row_b in zip(self.data, obj.data)
        ]

        return Matrix(new_data)

    def __eq__(self, obj: object) -> bool:
        ""
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
        ""
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
        ""
        return self * obj

    @property
    def data(self) -> list[list[float]]:
        ""
        return self._data

    @data.setter
    def data(self, new_data: list[list[float]]) -> None:
        ""
        if not len(new_data):
            raise ValueError("Matrix data cannot be empty.")

        cols = len(new_data[0])
        for i, row in enumerate(new_data):
            if len(row) != cols:
                raise ValueError(f"Row {i} has {len(row)} columns, expected {cols}.")

        self._data = new_data

    @property
    def shape(self) -> tuple[int, int]:
        ""
        rows = len(self.data)
        cols = len(self.data[0])
        return rows, cols

    def transpose(self) -> Matrix:
        ""
        new_data = zip(*self.data)
        t_rows = [list(row) for row in new_data]
        return Matrix(t_rows)

    def _check_index(self, row: int, col: int) -> None:
        ""
        if row < 0 or col < 0:
            raise IndexError(f"Negative index ({row}, {col}) is not allowed.")

        if row >= self.shape[0] or col >= self.shape[1]:
            raise IndexError(f"Index ({row}, {col}) is out of bounds for matrix with shape {self.shape}.")

