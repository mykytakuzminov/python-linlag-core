class Matrix:
    def __init__(self, data: list[list[float]]) -> None:
        self.data = data

    @classmethod
    def zeros(cls, rows: int, cols: int) -> Matrix:
        new_data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        return cls(new_data)

    @classmethod
    def ones(cls, rows: int, cols: int) -> Matrix:
        new_data = [[1.0 for _ in range(cols)] for _ in range(rows)]
        return cls(new_data)

    @classmethod
    def identity(cls, n: int) -> Matrix:
        new_data = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]
        return cls(new_data)

    def __str__(self) -> str:
        matrix = []
        for row in self.data:
            str_row = "".join([f"{e:^5.1f}" for e in row])
            matrix.append("| " + str_row + " |")
        return "\n".join(matrix)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data!r})"

    def __getitem__(self, el: tuple[int, int]) -> float:
        row, col = el
        self._check_index(row, col)
        return self.data[row][col]

    def __setitem__(self, el: tuple[int, int], val: float) -> None:
        row, col = el
        self._check_index(row, col)
        self.data[row][col] = val

    def __add__(self, obj: Matrix | int | float) -> Matrix:
        new_data: list[list[float]]
        if isinstance(obj, (int, float)):
            new_data = [[a + obj for a in row] for row in self.data]

        elif isinstance(obj, Matrix):
            if self.shape != obj.shape:
                raise ValueError("Different dimensions")

            new_data = [
                [a + b for a, b in zip(row_a, row_b, strict=False)]
                for row_a, row_b in zip(self.data, obj.data, strict=False)
            ]

        return Matrix(new_data)

    def __sub__(self, obj: Matrix | int | float) -> Matrix:
        new_data: list[list[float]]
        if isinstance(obj, (int, float)):
            new_data = [[a - obj for a in row] for row in self.data]

        elif isinstance(obj, Matrix):
            if self.shape != obj.shape:
                raise ValueError("Different dimensions")

            new_data = [
                [a - b for a, b in zip(row_a, row_b, strict=False)]
                for row_a, row_b in zip(self.data, obj.data, strict=False)
            ]

        return Matrix(new_data)

    def __mul__(self, obj: Matrix | int | float) -> Matrix:
        new_data: list[list[float]]
        if isinstance(obj, (int, float)):
            new_data = [[a * obj for a in row] for row in self.data]

        elif isinstance(obj, Matrix):
            if self.shape[1] != obj.shape[0]:
                raise ValueError("Impossible multiplication")

            cols_b = obj.transpose().data

            new_data = [
                [
                    sum(a * b for a, b in zip(row_a, col_b, strict=False))
                    for col_b in cols_b
                ]
                for row_a in self.data
            ]

        return Matrix(new_data)

    def __rmul__(self, obj: int | float) -> Matrix:
        return self * obj

    def __truediv__(self, obj: int | float) -> Matrix:
        if obj == 0:
            raise ValueError("Division by zero")
        new_data = [[a / obj for a in row] for row in self.data]
        return Matrix(new_data)

    def __neg__(self) -> Matrix:
        new_data = [[-a for a in row] for row in self.data]
        return Matrix(new_data)

    def __pow__(self, power: int | float) -> Matrix:
        new_data = [[a**power for a in row] for row in self.data]

        return Matrix(new_data)

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, Matrix):
            return False

        if self.shape != obj.shape:
            return False

        return all(
            a == b
            for row_a, row_b in zip(self.data, obj.data, strict=False)
            for a, b in zip(row_a, row_b, strict=False)
        )

    @property
    def data(self) -> list[list[float]]:
        return self._data

    @data.setter
    def data(self, new_data: list[list[float]]) -> None:
        if not len(new_data):
            raise ValueError("Empty data was given")

        # all rows have the same amount of elements
        cols = len(new_data[0])
        for row in new_data:
            if len(row) != cols:
                raise ValueError("Rows are not equal")

        self._data = new_data

    @property
    def shape(self) -> tuple[int, int]:
        rows = len(self.data)
        cols = len(self.data[0])
        return rows, cols

    def is_square(self) -> bool:
        return self.shape[0] == self.shape[1]

    def is_symmetric(self) -> bool:
        if not self.is_square():
            return False
        return self.data == self.transpose().data

    def transpose(self) -> Matrix:
        new_data = zip(*self.data, strict=False)
        t_rows = [list(row) for row in new_data]
        return Matrix(t_rows)

    def trace(self) -> float:
        if not self.is_square():
            raise ValueError("Not square matrix")
        return sum(self[i, i] for i in range(self.shape[0]))

    def submatrix(self, row: int, col: int) -> Matrix:
        new_data = [
            [self.data[i][j] for j in range(self.shape[1]) if j != col]
            for i in range(self.shape[0])
            if i != row
        ]

        return Matrix(new_data)

    def det(self) -> float:
        if not self.is_square():
            raise ValueError("Not square matrix")

        if self.shape == (1, 1):
            return self[0, 0]

        det: float = sum(
            (-1) ** j * self.data[0][j] * self.submatrix(0, j).det()
            for j in range(self.shape[1])
        )

        return det

    def copy(self) -> Matrix:
        return Matrix([row[:] for row in self.data])

    def total(self) -> float:
        return sum([sum(row) for row in self.data])

    def mean(self) -> float:
        return self.total() / (self.shape[0] * self.shape[1])

    def row(self, index: int) -> list[float]:
        return self.data[index]

    def col(self, index: int) -> list[float]:
        return [row[index] for row in self.data]

    def _check_index(self, row: int, col: int) -> None:
        if row < 0 or col < 0:
            raise IndexError("Negative index")

        if row >= self.shape[0] or col >= self.shape[1]:
            raise IndexError("Index out of bounds")
