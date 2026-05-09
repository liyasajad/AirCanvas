# AirCanvas 🎨

AirCanvas is an AI-powered virtual painting application that allows you to draw on your screen using nothing but hand gestures. By leveraging computer vision and MediaPipe's hand tracking, it creates a seamless and interactive augmented reality canvas right from your webcam!

## ✨ Features

- **Gesture-Based Drawing**: Simply point your index finger to draw in mid-air.
- **Smart Shape Recognition**: Draw a rough shape, and the AI will automatically snap it into a perfect circle, square, rectangle, triangle, or straight line!
- **Hover Interface**: Raise your index and middle fingers to navigate the UI, and hover over a tool to select it—no physical clicking required.
- **Fist Eraser**: Curl your fingers into a fist to instantly switch to the eraser tool.
- **Stroke Smoothing**: Uses SciPy spline interpolation to naturally smooth out your freehand strokes and eliminate jitter.
- **Undo/Redo System**: Full support for rolling back strokes and clearing the canvas.
- **Save to PNG**: Export your masterpiece directly to your computer.

## 🛠️ Installation

1. Ensure you have Python 3.8+ installed on your system.
2. Clone this repository:
   ```bash
   git clone https://github.com/liyasajad/AirCanvas.git
   cd AirCanvas
   ```
3. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe numpy scipy
   ```
   *(Note: `scipy` is required for advanced stroke smoothing)*

## 🚀 Usage

Run the script from your terminal:

```bash
python AirCanvas.py
```

### ✋ Hand Gestures

| Gesture | Action |
| :--- | :--- |
| **Index Finger Up** | Draw on the canvas |
| **Index + Middle Fingers Up** | Hover mode / Move cursor without drawing |
| **Hover on Toolbar (1 sec)** | Select the hovered color or tool |
| **Fist (All fingers curled)** | Eraser mode |

### ⌨️ Keyboard Shortcuts

If you prefer using your keyboard alongside gestures, the following hotkeys are supported:

| Key | Action |
| :---: | :--- |
| **Z** | Undo last stroke |
| **Y** | Redo |
| **C** | Clear the entire canvas |
| **S** | Save the canvas as a PNG image |
| **[** / **]** | Decrease / Increase brush size |
| **1 - 8** | Quick color selection |
| **Q** or **ESC** | Quit the application |

## 🏗️ Built With
* [OpenCV](https://opencv.org/) - Computer vision framework
* [MediaPipe](https://mediapipe.dev/) - ML pipeline for high-fidelity hand tracking
* [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/) - Data structures and spline interpolation
