from linlag import Matrix


def test_zeros_matrix():
    assert Matrix.zeros(2, 2) == Matrix([[0.0, 0.0], [0.0, 0.0]])
