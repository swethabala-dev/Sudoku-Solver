import cv2
import torch
import numpy as np
import os

from sudoku.model import DigitCNN


device = torch.device("cpu")


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "digit_cnn.pth"
)


model = DigitCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


def preprocess_digit_for_model(cell):
    """
    Preprocess a Sudoku cell for CNN digit recognition.
    Extracts the digit and centers it in a 28x28 image.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(
        cell,
        cv2.COLOR_BGR2GRAY
    )

    # Remove Sudoku cell border
    height, width = gray.shape

    margin = int(
        min(height, width) * 0.15
    )

    gray = gray[
        margin:height - margin,
        margin:width - margin
    ]

    # Threshold
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Find largest contour
    digit_contour = None
    largest_area = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > largest_area:

            largest_area = area
            digit_contour = contour

    # If no digit was found
    if digit_contour is None:

        return torch.zeros(
            (1, 1, 28, 28),
            dtype=torch.float32
        ).to(device)

    # Get digit bounding box
    x, y, w, h = cv2.boundingRect(
        digit_contour
    )

    digit = binary[
        y:y + h,
        x:x + w
    ]

    # Keep aspect ratio
    scale = 20 / max(w, h)

    new_width = max(
        1,
        int(w * scale)
    )

    new_height = max(
        1,
        int(h * scale)
    )

    digit = cv2.resize(
        digit,
        (new_width, new_height)
    )

    # Create 28x28 canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.float32
    )

    # Center digit
    start_x = (
        28 - new_width
    ) // 2

    start_y = (
        28 - new_height
    ) // 2

    canvas[
        start_y:start_y + new_height,
        start_x:start_x + new_width
    ] = digit

    # Normalize
    canvas = canvas / 255.0

    # Convert to tensor
    tensor = torch.tensor(
        canvas,
        dtype=torch.float32
    )

    # Add batch and channel dimensions
    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return tensor.to(device)

    # Remove outer cell border
    height, width = gray.shape

    margin = int(min(height, width) * 0.15)

    gray = gray[
        margin:height - margin,
        margin:width - margin
    ]

    # Resize to MNIST size
    gray = cv2.resize(
        gray,
        (28, 28)
    )

    # Invert if necessary
    _, gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Normalize to 0-1
    image = gray.astype(
        np.float32
    ) / 255.0

    # Convert to PyTorch tensor
    tensor = torch.tensor(
        image,
        dtype=torch.float32
    )

    # Add dimensions:
    # 28x28 → 1x28x28 → 1x1x28x28
    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


def predict_digit(cell):
    """
    Predict the digit inside a Sudoku cell.
    """

    image = preprocess_digit_for_model(cell)

    with torch.no_grad():

        output = model(image)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

    return prediction