import cv2
import numpy as np



def load_image(uploaded_file):
    """
    Converts a Streamlit uploaded file
    into an OpenCV image.
    """

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    return image



def to_grayscale(image):
    """
    Converts image from BGR color
    to grayscale.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return gray



def blur_image(gray):
    """
    Applies Gaussian blur to reduce noise.
    """

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return blurred



def threshold_image(blurred):
    """
    Converts image into binary form.
    Helps highlight Sudoku grid lines.
    """

    threshold = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )


    # Invert image
    threshold = cv2.bitwise_not(
        threshold
    )


    return threshold



def find_contours(threshold):
    """
    Finds all contours in the image.
    """

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    return contours



def find_sudoku_grid(contours):
    """
    Finds the largest 4-sided contour.
    The Sudoku board should be a square.
    """

    largest_area = 0
    sudoku_contour = None


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        if area > largest_area:

            perimeter = cv2.arcLength(
                contour,
                True
            )


            corners = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True
            )


            # Sudoku board should have 4 corners
            if len(corners) == 4:

                largest_area = area
                sudoku_contour = corners


    return sudoku_contour



def draw_grid(image, contour):
    """
    Draws detected Sudoku outline.
    """

    output = image.copy()


    if contour is not None:

        cv2.drawContours(
            output,
            [contour],
            -1,
            (0,255,0),
            3
        )


    return output