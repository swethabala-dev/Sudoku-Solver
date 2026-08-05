import cv2
import numpy as np


def load_image(uploaded_file):
    """
    Converts a Streamlit uploaded file into
    an OpenCV image.
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
    Converts a color image to grayscale.
    """

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def blur_image(gray):
    """
    Applies Gaussian Blur to reduce noise.
    """

    return cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


def threshold_image(blurred):
    """
    Converts the image into black and white
    using adaptive thresholding.

    This helps highlight the Sudoku grid.
    """

    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Invert colors so grid lines become white
    thresh = cv2.bitwise_not(thresh)

    return thresh