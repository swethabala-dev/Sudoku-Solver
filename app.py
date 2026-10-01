import streamlit as st

from sudoku.image_processing import (
    load_image,
    to_grayscale,
    blur_image,
    threshold_image,
    find_contours,
    find_sudoku_grid,
    draw_grid,
    warp_perspective,
    split_into_cells
)

from sudoku.digit_recognition import detect_digits
from sudoku.solver import solve_sudoku


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Sudoku Solver",
    page_icon="🧩",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🧩 Sudoku Solver")

st.write(
    """
    Upload a picture of a Sudoku puzzle and the app will
    detect the grid, recognize the numbers using a PyTorch
    CNN, and solve it.
    """
)


# --------------------------------------------------
# Upload Sudoku Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a Sudoku image",
    type=["png", "jpg", "jpeg"]
)


# --------------------------------------------------
# Process Image
# --------------------------------------------------

if uploaded_file is not None:

    # --------------------------------------------------
    # Load Image
    # --------------------------------------------------

    image = load_image(uploaded_file)

    st.subheader("Original Image")

    st.image(
        image,
        channels="BGR",
        use_container_width=True
    )


    # --------------------------------------------------
    # Convert to Grayscale
    # --------------------------------------------------

    gray = to_grayscale(image)

    st.subheader("Grayscale")

    st.image(
        gray,
        use_container_width=True
    )


    # --------------------------------------------------
    # Blur Image
    # --------------------------------------------------

    blurred = blur_image(gray)

    st.subheader("Blurred")

    st.image(
        blurred,
        use_container_width=True
    )


    # --------------------------------------------------
    # Threshold Image
    # --------------------------------------------------

    threshold = threshold_image(blurred)

    st.subheader("Threshold")

    st.image(
        threshold,
        use_container_width=True
    )


    # --------------------------------------------------
    # Find Contours
    # --------------------------------------------------

    contours = find_contours(threshold)


    # --------------------------------------------------
    # Find Sudoku Grid
    # --------------------------------------------------

    grid = find_sudoku_grid(contours)


    # --------------------------------------------------
    # Draw Detected Grid
    # --------------------------------------------------

    detected_image = draw_grid(
        image,
        grid
    )

    st.subheader("Detected Sudoku Grid")

    st.image(
        detected_image,
        channels="BGR",
        use_container_width=True
    )


    # --------------------------------------------------
    # Check Grid Detection
    # --------------------------------------------------

    if grid is None:

        st.warning(
            "Could not detect Sudoku grid. "
            "Try another image."
        )

    else:

        st.success(
            "Sudoku grid detected!"
        )


        # --------------------------------------------------
        # Perspective Transformation
        # --------------------------------------------------

        warped = warp_perspective(
            image,
            grid
        )


        if warped is not None:

            st.subheader(
                "Straightened Sudoku"
            )

            st.image(
                warped,
                channels="BGR",
                use_container_width=True
            )


            # --------------------------------------------------
            # Split Sudoku into 81 Cells
            # --------------------------------------------------

            cells = split_into_cells(
                warped
            )

            st.subheader(
                "Sudoku Cells"
            )


            for row in range(9):

                columns = st.columns(9)

                for col in range(9):

                    with columns[col]:

                        st.image(
                            cells[row][col],
                            channels="BGR",
                            use_container_width=True
                        )


            # --------------------------------------------------
            # Recognize Digits with PyTorch CNN
            # --------------------------------------------------

            board = detect_digits(
                cells
            )

            st.subheader(
                "Recognized Sudoku"
            )


            for row in board:

                st.write(
                    " ".join(
                        str(num)
                        if num != 0
                        else "."
                        for num in row
                    )
                )


            # --------------------------------------------------
            # Solve Sudoku
            # --------------------------------------------------

            solved_board = [
                row[:]
                for row in board
            ]


            if solve_sudoku(
                solved_board
            ):

                st.subheader(
                    "Solved Sudoku"
                )


                for row in solved_board:

                    st.write(
                        " ".join(
                            str(num)
                            for num in row
                        )
                    )


            else:

                st.error(
                    "Could not solve the detected Sudoku. "
                    "The CNN may have incorrectly recognized "
                    "one or more digits."
                )