# 🧩 AI Sudoku Solver

An AI-powered Sudoku Solver built with **Python, OpenCV, PyTorch, and Streamlit**. The application uses computer vision to detect a Sudoku puzzle from an uploaded image, recognizes handwritten or printed digits using a convolutional neural network (CNN), and solves the puzzle using a backtracking algorithm.

## 🚀 Features

* 📷 **Image-based Sudoku Detection**
  Upload an image of a Sudoku puzzle and automatically detect the puzzle grid.

* 🔍 **Computer Vision Processing**
  Uses OpenCV for image preprocessing, thresholding, contour detection, perspective transformation, and cell extraction.

* 🧠 **CNN Digit Recognition**
  Uses a PyTorch convolutional neural network to recognize Sudoku digits from individual cells.

* 🧩 **Automated Puzzle Solving**
  Solves recognized Sudoku puzzles using a recursive backtracking algorithm.

* 🌐 **Interactive Streamlit Interface**
  Provides a simple web interface for uploading puzzle images and viewing results.

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **OpenCV**
* **NumPy**
* **Streamlit**
* **CNN / Deep Learning**
* **Computer Vision**
* **Backtracking Algorithms**

## 📁 Project Structure

```text
Sudoku Solver/
│
├── app.py
├── sudoku/
│   ├── digit_recognition.py
│   └── digit_cnn.pth
│
├── requirements.txt
└── README.md
```

## ⚙️ How It Works

The application follows a multi-stage computer vision and AI pipeline:

```text
Upload Sudoku Image
        ↓
Image Preprocessing
        ↓
Grid Detection
        ↓
Perspective Transformation
        ↓
Split Grid into 81 Cells
        ↓
CNN Digit Recognition
        ↓
Generate Sudoku Board
        ↓
Backtracking Solver
        ↓
Display Solved Puzzle
```

### 1. Image Processing

OpenCV processes the uploaded image by:

* Converting the image to grayscale
* Applying thresholding
* Detecting contours
* Identifying the Sudoku grid
* Applying a perspective transformation
* Dividing the grid into 81 individual cells

### 2. Digit Recognition

Each Sudoku cell is passed through a trained **PyTorch CNN** to determine which digit is present.

The model was evaluated on a held-out test set and achieved approximately **99.14% test accuracy**.

### 3. Sudoku Solving

Once the digits are recognized, the puzzle is represented as a 9×9 matrix. A recursive **backtracking algorithm** searches for a valid solution while enforcing Sudoku constraints.

### 4. Streamlit Application

The entire pipeline is integrated into a Streamlit application, allowing users to upload an image and interact with the solver through a web interface.

## 📊 Performance

| Component                        |                                Result |
| -------------------------------- | ------------------------------------: |
| CNN Test Accuracy                |                            **99.14%** |
| Grid Detection Accuracy          |                               **90%** |
| Grid Detection Improvement       | **+35% vs. fixed-threshold baseline** |
| Cell Misclassification Reduction |                               **25%** |

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Create and activate a virtual environment:

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser. Upload a Sudoku image to begin the solving process.

## 🧠 Key Concepts Demonstrated

This project combines several areas of computer science and AI:

* **Deep Learning**
* **Convolutional Neural Networks**
* **Computer Vision**
* **Image Preprocessing**
* **Model Evaluation**
* **Machine Learning Inference**
* **Algorithm Design**
* **Recursive Backtracking**
* **Web Application Development**
* **AI Model Integration**

## 🔮 Future Improvements

Potential improvements include:

* Improve grid detection for more complex or poorly photographed puzzles
* Add support for handwritten Sudoku puzzles
* Display the recognized digits before solving
* Highlight the solution process step-by-step
* Improve CNN robustness across different fonts and image conditions
* Add confidence scores for individual digit predictions
* Deploy the application for public use

## 👩‍💻 Author

**Swetha Bala**
Purdue University | Artificial Intelligence + Computer Science

Built as a personal project to explore the intersection of **AI, computer vision, and software development**.
