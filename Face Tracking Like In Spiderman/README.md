- Spider-Man Inspired Real-Time Face Tracking HUD

A real-time computer vision application built with Python, OpenCV, and MediaPipe that uses a webcam to track facial landmarks and hand gestures. The tracked movements control an interactive heads-up display inspired by Spider-Man-style augmented-reality interfaces.

The project is an interactive computer vision experiment rather than a pre-rendered video effect. Face tracking, hand tracking, gesture recognition, graphical rendering, and visual effects are processed live from the webcam feed.

- Overview

This project combines facial landmark detection, hand tracking, gesture recognition, and real-time graphical rendering to create an interactive augmented-reality-style interface.

MediaPipe detects facial landmarks and hands from the webcam stream. OpenCV processes the camera frames, performs image manipulation, and renders the HUD elements and visual effects over the live video.

- Features

- Real-Time Face Tracking

The application uses MediaPipe Face Mesh to detect facial landmarks from the webcam feed. Facial landmark data is used to locate the user's face, track facial features, position visual elements relative to the face, and trigger interactive facial effects.

- Hand Tracking

MediaPipe Hands detects hands and individual finger landmarks. Hand tracking provides the primary method of interacting with the HUD.

Supported interactions include single-finger movement, two-finger scaling and rotation, and pinch-based interaction.

- Interactive HUD

Graphical interface elements are rendered over the live webcam feed. These elements are dynamically positioned and transformed based on tracked facial and hand coordinates.

- Gesture control

A single index finger can move the HUD element, two index fingers can resize and rotate it, and a pinch gesture can trigger the interactive facial effect.

- Real-time rendering

All visual effects are generated while the application is running. The project does not rely on a pre-recorded video. The webcam stream is processed frame-by-frame so the interface can respond immediately to user movement.

- Video recording

The application includes an optional recording mode for demonstrations, testing, and later viewing.

- Technologies

- Python 3.12
- OpenCV
- MediaPipe
- NumPy

OpenCV handles webcam capture, image processing, frame manipulation, graphical rendering, real-time display, and video recording.

MediaPipe provides facial and hand landmark detection.

NumPy handles numerical operations, coordinate calculations, transformations, and tracking data.

- Requirements

- Python 3.12 or compatible Python 3.x version
- Working webcam
- Git
- Terminal or command prompt

Python 3.12 is recommended for compatibility with the MediaPipe version used during development.

- Installation

Clone the repository:

```bash
git clone https://github.com/808kirai/Face-Tracking-Filter.git
```

Navigate into the project directory:

```bash
cd Face-Tracking-Filter/"Face Tracking Like In Spiderman"
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable:

```bash
pip install mediapipe==0.10.21 opencv-python==4.11.0.86 numpy==1.26.4
```

- Running the Application

Start the application with:

```bash
python main.py
```

The application will open the webcam and begin processing the video stream in real time.

Make sure the webcam is connected and that the operating system has granted the application camera access.

- Controls

- Keyboard controls

`R` to Start or stop video recording 
`Q` or `Esc` to Exit the application (closing the window also works)

- Hand gestures

 One index finger to Move the HUD element 
 Two index fingers to Resize and rotate the HUD element 
 Pinch to  Activate the interactive facial effect 

- Recording

Press `R` while the application is running to start recording the processed webcam feed.

Press `R` again to stop recording.

Videos are saved as `.mp4` files in the `recordings/` folder next to `main.py`. The frame rate matches the speed the app is actually running at, so playback looks like real time.

The resulting video can be used for demonstrations, testing, or sharing.

- Project Structure

```text
Face Tracking Like In Spiderman/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/
```

The `.venv` directory should not be committed to GitHub because it contains locally installed packages and environment-specific files.

The `.gitignore` file should contain:

```text
.venv/
__pycache__/
*.pyc
```

- Performance

Real-time computer vision can require significant processing power. Performance depends on the computer's CPU, webcam resolution, camera frame rate, lighting conditions, and other applications running simultaneously.

For better performance:

- Use a reasonable webcam resolution.
- Close unnecessary applications.
- Use the recommended Python and MediaPipe versions.
- Ensure the face and hands are clearly visible.
- Use adequate lighting.

Good lighting can significantly improve face and hand landmark detection.

- Troubleshooting

- Camera not opening

Check that the webcam is connected and available to other applications.

If multiple cameras are connected, the camera index in `main.py` may need to be changed.

For example:

```python
cv2.VideoCapture(0)
```

can be changed to:

```python
cv2.VideoCapture(1)
```

depending on the camera configuration.

- MediaPipe import error

Check the installed MediaPipe version:

```bash
pip show mediapipe
```

The project was developed and tested using:

```text
mediapipe==0.10.21
```

- Poor face or hand detection

Improve the lighting and make sure the face and hands are clearly visible to the camera.

Avoid extreme camera angles and excessive motion when testing the tracking system.

- Development

The project is structured so additional visual effects, HUD components, and gesture interactions can be added over time.

Potential future improvements include:

- Additional HUD components
- More gesture-controlled interactions
- Improved facial effects
- Custom animations
- Additional interface themes
- Improved tracking stability
- Multiple simultaneously tracked objects
- Additional camera and recording controls

- Purpose

This project was created as an exploration of real-time computer vision, gesture recognition, facial landmark tracking, and interactive graphical interfaces.

It demonstrates how facial landmarks and hand tracking can be combined with traditional computer vision techniques to create an interactive augmented-reality-style experience using commonly available hardware and open-source technologies.

- Disclaimer

This project is an independent computer vision experiment inspired by the visual design language of Spider-Man-related augmented-reality interfaces.

It is not affiliated with, endorsed by, or officially associated with Marvel, Sony, or any Spider-Man production.

- License

This project is provided for educational and experimental purposes. If you choose to distribute or modify the project, consider adding an appropriate open-source license such as the MIT License.