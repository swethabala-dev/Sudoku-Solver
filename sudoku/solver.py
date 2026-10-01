def is_valid(board, row, col, num):
    """
    Check whether a number can be placed
    in a particular Sudoku cell.
    """

    # Check row
    for current_col in range(9):

        if board[row][current_col] == num:
            return False

    # Check column
    for current_row in range(9):

        if board[current_row][col] == num:
            return False

    # Find 3x3 box
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3

    for current_row in range(box_row, box_row + 3):

        for current_col in range(box_col, box_col + 3):

            if board[current_row][current_col] == num:
                return False

    return True


def solve_sudoku(board):
    """
    Solve a Sudoku board using backtracking.

    Returns True if the board can be solved.
    Returns False if there is no solution.
    """

    # Find an empty cell
    for row in range(9):

        for col in range(9):

            if board[row][col] == 0:

                # Try numbers 1 through 9
                for num in range(1, 10):

                    if is_valid(
                        board,
                        row,
                        col,
                        num
                    ):

                        # Place number
                        board[row][col] = num

                        # Recursively solve
                        if solve_sudoku(board):
                            return True

                        # Undo if it doesn't work
                        board[row][col] = 0

                # No number worked
                return False

    # No empty cells remain
    return True
