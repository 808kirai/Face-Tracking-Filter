Hand Portal Filter

A real-time webcam effect built with Python, OpenCV and MediaPipe. Frame your face with both hands, like a camera viewfinder, and the area inside your fingers becomes a portal with a filter applied. Everything outside the portal stays as the normal camera image.

Every time you drop your hands and frame again, the portal shows the next filter.

How it works

- MediaPipe Hands tracks both hands. The thumb tip and index fingertip of each hand are the 4 corners of the portal.
- The corners are smoothed so the portal doesn't jitter, then ordered by angle so the shape never crosses itself. The portal can be tilted, skewed, stretched or shrunk just by moving your hands.
- Only the inside of the portal is filtered, and it gets a thin glowing white outline.
- With no hands, or only one, you see the plain camera feed.

Filters

The filters cycle in this order.

- Blue halftone - Light blue and deep blue duotone dots
- Wireframe grid - See-through white grid lines
- Pop art - Orange background and magenta subject. It uses MediaPipe selfie segmentation to find the person.
- Newspaper - Black and white dot halftone
- RGB split - Chromatic aberration glitch
- Thermal - Jet colormap
- Sepia - Warm, faded vintage tone
- Frosted glass - Heavy blur with a white tint

Each filter is a plain function `f(image) -> image` in the `filtros` list in `main.py`, so adding your own is a matter of writing a function and appending it.

Hand controls

- Both hands in a camera frame, thumbs and index fingers out - Opens the portal
- Drop your hands and frame again - Next filter

Keyboard controls

- `N` or `Space` - Next filter
- `R` - Start or stop recording
- `Q`, `Esc` or closing the window - Quit

Requirements

- Python 3.11 or 3.12
- A webcam

Installation

```bash
git clone https://github.com/808kirai/Face-Tracking-Filter.git
cd Face-Tracking-Filter/"Face Tracking Like In Spiderman"
python -m venv .venv
```

Activate the virtual environment.

```powershell
.venv\Scripts\Activate.ps1          # Windows PowerShell
```

```cmd
.venv\Scripts\activate              # Windows Command Prompt
```

```bash
source .venv/bin/activate           # macOS or Linux
```

Install the dependencies and run it.

```bash
pip install -r requirements.txt
python main.py
```

Recording

Press `R` to start recording and `R` again to stop. Videos are saved as `.mp4` files in the `recordings/` folder next to `main.py`.

The frame rate matches the speed the app is actually running at, so playback looks like real time. On-screen labels are left out of the recording.

Performance

- Hand tracking runs on a copy of the frame scaled down to 640px wide, and the camera is read on a background thread.
- Only the portal's bounding box is filtered, not the whole frame.
- Most filters take 2 to 4 ms even on a large portal. Pop art is the heaviest, at around 18 ms, because it runs a segmentation model. That model only loads the first time pop art is shown.

Good, even lighting makes hand tracking much more reliable.

Troubleshooting

Camera not opening or no frames

Check that the webcam is connected and that your OS allows camera access. Close any video call or other app using the camera, because on Windows it can block the app from reading frames and you will see "Camera stopped sending frames." right after startup.

If you have more than one camera, change the index in `main.py`.

```python
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
```

The portal flickers or does not appear

Both hands need to be fully visible, with the thumb and index finger clearly separated. Avoid backlighting and keep your hands within the frame.

MediaPipe import error

The project is tested with `mediapipe==0.10.21`. Check the installed version with `pip show mediapipe`.

Disclaimer

This is an independent computer vision experiment thhat i have made for fun from an Instagram inspo. It is not affiliated with, endorsed by or associated with Marvel, Sony or any Spider-Man production.
