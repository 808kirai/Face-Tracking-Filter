# Hand Portal Filter

A real-time webcam effect built with Python, OpenCV, and MediaPipe. Frame your face with both hands, like a camera viewfinder, and the area inside your fingers becomes a portal with a filter applied. Everything outside the portal stays as the normal camera image.

Every time you drop your hands and frame again, the portal shows the next filter.

## How it works

- MediaPipe Hands tracks both hands. The **thumb tip and index fingertip of each hand** are the 4 corners of the portal.
- The corners are smoothed so the portal doesn't jitter, then ordered by angle so the shape never crosses itself. The portal can be tilted, skewed, stretched, or shrunk just by moving your hands.
- Only the inside of the portal is filtered, and it gets a thin glowing white outline.
- With no hands, or only one, you see the plain camera feed.

## Filters

The filters cycle in this order:

1. **Pink halftone**: pink and magenta duotone dots
2. **Wireframe grid**: see-through white grid lines
3. **Pop art**: orange background, magenta subject (uses MediaPipe selfie segmentation to find the person)
4. **Newspaper**: black and white dot halftone
5. **RGB split**: chromatic aberration glitch
6. **Thermal**: jet colormap
7. **Sepia**: warm, faded vintage tone
8. **Frosted glass**: heavy blur with a white tint

Each filter is a plain function `f(image) -> image` in the `FILTROS` list in `main.py`, so adding your own is a matter of writing a function and appending it.

## Controls

### Hands

| Gesture | Action |
| --- | --- |
| Both hands in a "camera frame" (thumbs and index fingers out) | Open the portal |
| Drop your hands and frame again | Next filter |

### Keyboard

| Key | Action |
| --- | --- |
| `N` or `Space` | Next filter |
| `R` | Start or stop recording |
| `Q`, `Esc`, or closing the window | Quit |

## Requirements

- Python 3.11 or 3.12
- A webcam

## Installation

```bash
git clone https://github.com/808kirai/Face-Tracking-Filter.git
cd Face-Tracking-Filter/"Face Tracking Like In Spiderman"
python -m venv .venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1          # Windows PowerShell
```

```cmd
.venv\Scripts\activate              # Windows Command Prompt
```

```bash
source .venv/bin/activate           # macOS or Linux
```

Install the dependencies and run it:

```bash
pip install -r requirements.txt
python main.py
```

## Recording

Press `R` to start recording and `R` again to stop. Videos are saved as `.mp4` files in the `recordings/` folder next to `main.py`.

The frame rate matches the speed the app is actually running at, so playback looks like real time. On-screen labels are left out of the recording.

## Performance

- Hand tracking runs on a copy of the frame scaled down to 640px wide, and the camera is read on a background thread.
- Only the portal's bounding box is filtered, not the whole frame.
- Most filters take 2 to 4 ms even on a large portal. Pop art is the heaviest, at around 18 ms, because it runs a segmentation model. That model only loads the first time pop art is shown.

Good, even lighting makes hand tracking much more reliable.

## Troubleshooting

### Camera not opening

Check that the webcam is connected and not in use by another app, and that your OS allows camera access. If you have more than one camera, change the index in `main.py`:

```python
cap = cv2.VideoCapture(1)
```

### The portal flickers or does not appear

Both hands need to be fully visible, with the thumb and index finger clearly separated. Avoid backlighting and keep your hands within the frame.

### MediaPipe import error

The project is tested with `mediapipe==0.10.21`. Check the installed version with `pip show mediapipe`.

## Disclaimer

This is an independent computer vision experiment. It is not affiliated with, endorsed by, or associated with Marvel, Sony, or any Spider-Man production.

## License

Provided for educational and experimental use. If you plan to distribute it, consider adding an open-source license such as MIT.
