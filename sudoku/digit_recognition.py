import cv2
import numpy as np

from sudoku.predict import predict_digit


def preprocess_cell(cell):
    """
    Preprocess one Sudoku cell for detecting
    whether it is empty.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(
        cell,
        cv2.COLOR_BGR2GRAY
    )

    # Apply threshold
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove outer border
    height, width = thresh.shape

    margin = int(
        min(height, width) * 0.15
    )

    thresh = thresh[
        margin:height - margin,
        margin:width - margin
    ]

    return thresh


def is_cell_empty(cell):
    """
    Determine whether a Sudoku cell is empty.
    """

    processed = preprocess_cell(cell)

    # Count white pixels
    white_pixels = cv2.countNonZero(
        processed
    )

    total_pixels = (
        processed.shape[0] *
        processed.shape[1]
    )

    fill_ratio = (
        white_pixels / total_pixels
    )

    return fill_ratio < 0.05


def detect_digits(cells):
    """
    Recognize digits in all 81 Sudoku cells.

    Returns:
        9x9 board

        0 = empty
        1-9 = recognized digit
    """

    board = []

    for row in range(9):

        board_row = []

        for col in range(9):

            cell = cells[row][col]

            # Empty cell
            if is_cell_empty(cell):

                board_row.append(0)

            # Digit cell
            else:

                digit = predict_digit(cell)

                board_row.append(digit)

        board.append(board_row)

    return board