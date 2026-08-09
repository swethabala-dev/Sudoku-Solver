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

# 
# Page Configuration
# 
st.set_page_config(
    page_title="Sudoku Solver",
    page_icon="🧩",
    layout="centered"
)


# 
# Title
# 
st.title("🧩 Sudoku Solver")

st.write(
    """
    Upload a picture of a Sudoku puzzle and the app will
    detect the grid, recognize the numbers, and solve it.

    Current stage:
    - Image preprocessing
    - Sudoku grid detection
    """
)


# 
# Upload Sudoku Image
# 
uploaded_file = st.file_uploader(
    "Upload a Sudoku image",
    type=["png", "jpg", "jpeg"]
)


# 
# Process Image
# 
if uploaded_file is not None:

    # Load image
    image = load_image(uploaded_file)


    # Original image
    st.subheader("Original Image")

    st.image(
        image,
        channels="BGR",
        use_container_width=True
    )


    # Convert to grayscale
    gray = to_grayscale(image)

    st.subheader("Grayscale")

    st.image(
        gray,
        use_container_width=True
    )


    # Blur image
    blurred = blur_image(gray)

    st.subheader("Blurred")

    st.image(
        blurred,
        use_container_width=True
    )


    # Threshold image
    threshold = threshold_image(blurred)

    st.subheader("Threshold")

    st.image(
        threshold,
        use_container_width=True
    )


    # Find contours
    contours = find_contours(threshold)


    # Find Sudoku grid
    grid = find_sudoku_grid(contours)


    # Draw grid
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


    if grid is not None:
        st.success("Sudoku grid detected!")

    else:
        st.warning(
            "Could not detect Sudoku grid. Try another image."
        )
#Display the straigtend sudoku grid

warped = warp_perspective(image, grid)

if warped is not None:

    st.subheader("Straightened Sudoku")

    st.image(
        warped,
        channels="BGR",
        use_container_width=True
    )
# Perspective Transformation

warped = warp_perspective(image, grid)

if warped is not None:

    st.subheader("Straightened Sudoku")

    st.image(
        warped,
        channels="BGR",
        use_container_width=True
    )


    # Split Sudoku into 81 Cells
    
    cells = split_into_cells(warped)

    st.subheader("Sudoku Cells")

    for row in range(9):

        columns = st.columns(9)

        for col in range(9):

            with columns[col]:
                st.image(
                    cells[row][col],
                    channels="BGR",
                    use_container_width=True
                )