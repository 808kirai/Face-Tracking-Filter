import os
os.environ["GLOG_minloglevel"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import time

# Live hand portal effect.
# Thumbs and index fingers of both hands frame a portal, and only the
# inside of the portal is filtered. Each new framing shows the next filter.

hands = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)

# Person segmentation for the pop art filter, created when first needed.
segmenter = None

WINDOW_NAME = "Portal Filter"
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")

# Tracking runs on a smaller copy of the frame. Landmarks come back
# normalized, so they still map onto the full-size frame.
MAX_TRACK_WIDTH = 640

# Hand landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8

# A gap longer than this counts as taking the hands away and framing again,
# which moves on to the next filter.
REFRAME_GAP = 0.4

# Keep the last portal on screen through tracking dropouts this short.
HOLD_TIME = 0.15


class CameraStream:
    """Reads the camera on a background thread, so the main loop
    always gets the newest frame instead of waiting for the camera."""

    def __init__(self, cap):
        self.cap = cap
        self.cond = threading.Condition()
        self.ok = True
        self.frame = None
        self.count = 0
        self.running = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()

            with self.cond:
                self.ok = ok
                self.frame = frame
                self.count += 1
                self.cond.notify_all()

            if not ok:
                break

    def read(self, last_count):
        # Wait for a frame newer than the one returned last time.
        with self.cond:
            self.cond.wait_for(lambda: self.count != last_count, timeout=2.0)

            if self.count == last_count:
                return False, None, last_count

            return self.ok, self.frame, self.count

    def release(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()


class OneEuroFilter:
    """Adaptive smoothing for noisy tracking data (Casiez et al., 2012).

    Smooths heavily while the input is still, which removes jitter, and
    lightly while it moves fast, which removes lag. Works on scalars or
    numpy arrays, and uses real time steps so it behaves the same at any
    frame rate.
    """

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)

        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = t - self.t_prev

        if dt <= 0:
            return self.x_prev

        dx = (x - self.x_prev) / dt
        dx_hat = lerp(self.dx_prev, dx, self._alpha(dt, self.d_cutoff))

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        x_hat = lerp(self.x_prev, x, self._alpha(dt, cutoff))

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

camera = CameraStream(cap)
last_frame = 0

# Smooths the 4 fingertip corners (left thumb, left index, right thumb, right index).
corner_filter = OneEuroFilter(min_cutoff=0.5, beta=0.01)

# Last portal corners and when they were seen.
portal = None
portal_seen_at = None

filtro_index = 0

# Set while recording, None otherwise.
writer = None

# Measured loop rate, so recordings play back at real speed.
fps = None
prev_time = None


# Helpers.

def lerp(a, b, amount):
    return a + (b - a) * amount


def glow_polygon(image, pts, thickness=1):
    # Soft glow, no clutter.
    # Only the area around the outline is blurred, in a single pass,
    # so the rest of the frame stays sharp and cheap to draw.
    h, w = image.shape[:2]
    pad = thickness * 3 + 8

    x1 = max(0, int(pts[:, 0].min()) - pad)
    y1 = max(0, int(pts[:, 1].min()) - pad)
    x2 = min(w, int(pts[:, 0].max()) + pad + 1)
    y2 = min(h, int(pts[:, 1].max()) + pad + 1)

    if x1 >= x2 or y1 >= y2:
        return

    roi = image[y1:y2, x1:x2]
    local = (pts - (x1, y1)).astype(np.int32)

    glow = np.zeros(roi.shape[:2], dtype=np.uint8)

    cv2.polylines(
        glow, [local], True,
        (255,),
        thickness * 3,
        cv2.LINE_AA
    )

    glow = cv2.GaussianBlur(glow, (0, 0), 2)

    # Lighten towards white where the glow is, up to 12%.
    lift = cv2.multiply(
        cv2.bitwise_not(roi),
        cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR),
        scale=0.12 / 255.0
    )
    roi[:] = cv2.add(roi, lift)

    cv2.polylines(
        roi, [local], True,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA
    )


def order_corners(pts):
    # Sort by angle around the centroid so the corners always
    # make a polygon that does not cross itself.
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

    return pts[np.argsort(angles)]


def get_portal_corners(hand_results, width, height):
    # The 4 fingertips of a two-hand "camera frame" gesture,
    # or None unless exactly two hands are visible.
    landmarks = hand_results.multi_hand_landmarks

    if not landmarks or len(landmarks) != 2:
        return None

    # Left hand on screen first, so each corner keeps its identity
    # from frame to frame and can be smoothed.
    ordered = sorted(landmarks, key=lambda hand: hand.landmark[WRIST].x)

    return np.array(
        [
            (hand.landmark[tip].x * width, hand.landmark[tip].y * height)
            for hand in ordered
            for tip in (THUMB_TIP, INDEX_TIP)
        ],
        dtype=np.float64
    )


# Filters.
# Each takes a BGR image and returns a filtered BGR image of the same size.

HALFTONE_LEVELS = 24

# Pre-rendered dot tiles, one per dot size, keyed by (cell, paper, ink).
halftone_tiles = {}


def get_halftone_tiles(cell, paper, ink):
    key = (cell, paper, ink)

    if key not in halftone_tiles:
        # Distance from each pixel to the center of the cell.
        d = np.arange(cell) - (cell - 1) / 2.0
        dist = np.sqrt(d[:, None] ** 2 + d[None, :] ** 2)

        radius = np.linspace(0.0, cell * 0.72, HALFTONE_LEVELS)

        # Soft dot edges instead of jagged ones.
        coverage = np.clip(radius[:, None, None] - dist + 0.5, 0.0, 1.0)[..., None]

        paper_c = np.array(paper, dtype=np.float32)
        ink_c = np.array(ink, dtype=np.float32)

        halftone_tiles[key] = (paper_c + (ink_c - paper_c) * coverage).astype(np.uint8)

    return halftone_tiles[key]


def halftone(image, cell, paper, ink):
    h, w = image.shape[:2]

    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # One dot per cell, bigger where the image is darker.
    cells_w = -(-w // cell)
    cells_h = -(-h // cell)

    small = cv2.resize(gray, (cells_w, cells_h), interpolation=cv2.INTER_AREA)
    levels = ((255 - small.astype(np.int32)) * (HALFTONE_LEVELS - 1) + 127) // 255

    # Lay the matching dot tile into every cell, then trim to size.
    tiles = get_halftone_tiles(cell, paper, ink)[levels]
    out = tiles.transpose(0, 2, 1, 3, 4).reshape(cells_h * cell, cells_w * cell, 3)

    return np.ascontiguousarray(out[:h, :w])


def pink_halftone(image):
    return halftone(image, 8, paper=(215, 185, 255), ink=(125, 10, 205))


def wireframe_grid(image):
    out = cv2.convertScaleAbs(image, alpha=0.75, beta=0)

    step = max(12, min(image.shape[:2]) // 14)

    out[::step] = (out[::step] * 0.2 + 255 * 0.8).astype(np.uint8)
    out[:, ::step] = (out[:, ::step] * 0.2 + 255 * 0.8).astype(np.uint8)

    return out


# Colors for the pop art labels, as a lookup table.
POP_ART_COLORS = np.zeros((256, 1, 3), dtype=np.uint8)
POP_ART_COLORS[0] = (0, 140, 255)      # orange background
POP_ART_COLORS[1] = (190, 30, 235)     # magenta subject
POP_ART_COLORS[2] = (80, 0, 90)        # deep purple ink


def pop_art(image):
    global segmenter

    # Created on first use, so the model only loads if this filter is shown.
    if segmenter is None:
        segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    h, w = image.shape[:2]

    # The model works at 256px anyway, so hand it a small copy
    # and scale the mask back up.
    scale = min(1.0, 256 / max(h, w))
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    mask = cv2.resize(segmenter.process(rgb).segmentation_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    subject = mask > 0.5

    # 0 = background, 1 = subject, 2 = ink.
    label = subject.astype(np.uint8)

    if subject.any():
        gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)

        # Shadows from the subject's own brightness, plus local detail
        # (eyes, mouth, outlines) that a single threshold would miss.
        shadow = np.percentile(gray[subject], 25)

        block = max(11, (min(gray.shape) // 16) | 1)
        detail = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block, 6
        ) > 0

        label += subject & ((gray < shadow) | detail)

    return cv2.LUT(cv2.merge((label, label, label)), POP_ART_COLORS)


def newspaper(image):
    return halftone(image, 7, paper=(232, 238, 240), ink=(25, 25, 25))


def rgb_split(image):
    b, g, r = cv2.split(image)

    shift = max(4, image.shape[1] // 40)

    out = cv2.merge((
        np.roll(b, -shift, axis=1),
        g,
        np.roll(r, shift, axis=1)
    ))

    # Faint scan lines.
    out[::3] = (out[::3] * 0.8).astype(np.uint8)

    return out


def thermal(image):
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


# Sepia weights for BGR input. Rows are the output B, G, R channels.
SEPIA = np.array([
    [0.131, 0.534, 0.272],
    [0.168, 0.686, 0.349],
    [0.189, 0.769, 0.393]
], dtype=np.float32)


def sepia(image):
    toned = cv2.transform(image, SEPIA)

    # Slightly faded, like an old print.
    return cv2.convertScaleAbs(toned, alpha=0.88, beta=18)


def frosted_glass(image):
    h, w = image.shape[:2]

    # Blurring a quarter-size copy is much cheaper than a huge kernel.
    small = cv2.resize(image, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # White tint.
    return cv2.convertScaleAbs(blurred, alpha=0.7, beta=255 * 0.3)


FILTROS = [
    pink_halftone,
    wireframe_grid,
    pop_art,
    newspaper,
    rgb_split,
    thermal,
    sepia,
    frosted_glass
]


def filter_name(filtro):
    return filtro.__name__.replace("_", " ").upper()


def render_portal(frame, p1, p2, p3, p4, filtro):
    """Applies filtro inside the quad p1-p4 and draws its outline."""

    h, w = frame.shape[:2]
    pts = np.array([p1, p2, p3, p4], dtype=np.int32)

    # Only filter the portal's bounding box, not the whole frame.
    x, y, bw, bh = cv2.boundingRect(pts)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)

    if x2 - x1 < 2 or y2 - y1 < 2:
        return frame

    roi = frame[y1:y2, x1:x2]
    filtered = filtro(roi.copy())

    # fillPoly, unlike fillConvexPoly, also handles a skewed concave quad.
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts - (x1, y1)], (255,))

    inside = mask > 0
    roi[inside] = filtered[inside]

    glow_polygon(frame, pts, 1)

    return frame


def draw_label(frame, text, origin, scale=0.46):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA
    )


# Start.

print()
print("==========================================")
print("            LIVE PORTAL FILTER")
print("==========================================")
print()
print("Frame with both hands: thumbs and index")
print("fingers make the 4 corners of the portal.")
print("Drop your hands and frame again for the next filter.")
print()
print("N / SPACE = next filter")
print("R = record")
print("Q / ESC = quit")
print()


# Main loop.

try:
    while True:

        ok, frame, last_frame = camera.read(last_frame)

        if not ok or frame is None:
            print("Camera stopped sending frames.")
            break

        now = time.perf_counter()

        if prev_time is not None and now > prev_time:
            instant_fps = 1.0 / (now - prev_time)
            fps = instant_fps if fps is None else lerp(fps, instant_fps, 0.2)

        prev_time = now

        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        track_frame = frame

        if width > MAX_TRACK_WIDTH:
            track_frame = cv2.resize(
                frame,
                (MAX_TRACK_WIDTH, round(height * MAX_TRACK_WIDTH / width)),
                interpolation=cv2.INTER_AREA
            )

        rgb = cv2.cvtColor(
            track_frame,
            cv2.COLOR_BGR2RGB
        )

        # Read-only input lets MediaPipe skip a copy.
        rgb.flags.writeable = False

        hand_results = hands.process(rgb)

        # Portal tracking.

        corners = get_portal_corners(hand_results, width, height)

        if corners is not None:

            if portal_seen_at is None or now - portal_seen_at > REFRAME_GAP:

                # A new framing: start smoothing fresh, and move on
                # to the next filter (except for the very first one).
                corner_filter.reset()

                if portal_seen_at is not None:
                    filtro_index = (filtro_index + 1) % len(FILTROS)

            portal = corner_filter(corners, now)
            portal_seen_at = now

        elif portal_seen_at is None or now - portal_seen_at > HOLD_TIME:
            portal = None

        filtro = FILTROS[filtro_index]

        if portal is not None:
            p1, p2, p3, p4 = order_corners(portal)
            frame = render_portal(frame, p1, p2, p3, p4, filtro)

        # Recording.

        if writer is not None:

            # Write before drawing labels so the video stays clean.
            writer.write(frame)

            cv2.circle(
                frame,
                (width - 25, 25),
                7,
                (255, 255, 255),
                -1
            )

            draw_label(frame, "REC", (width - 65, 30), 0.45)

        # Status.

        status = f"{filtro_index + 1}/{len(FILTROS)}  {filter_name(filtro)}"

        if portal is None:
            status = "FRAME WITH BOTH HANDS  //  NEXT: " + filter_name(filtro)

        draw_label(frame, status, (16, 28))

        # Display.

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Next filter.

        if key in (ord("n"), ord("N"), ord(" ")):
            filtro_index = (filtro_index + 1) % len(FILTROS)

        # Toggle recording.

        if key in (ord("r"), ord("R")):

            if writer is None:

                os.makedirs(RECORDINGS_DIR, exist_ok=True)

                filename = os.path.join(
                    RECORDINGS_DIR,
                    "portal_"
                    + time.strftime("%Y%m%d_%H%M%S")
                    + ".mp4"
                )

                fourcc = cv2.VideoWriter.fourcc(
                    *"mp4v"
                )

                record_fps = float(np.clip(round(fps or 30), 5, 60))

                writer = cv2.VideoWriter(
                    filename,
                    fourcc,
                    record_fps,
                    (width, height)
                )

                if writer.isOpened():
                    print(f"Recording at {record_fps:.0f} fps:", filename)
                else:
                    writer.release()
                    writer = None
                    print("Could not start recording:", filename)

            else:

                writer.release()
                writer = None

                print("Recording stopped.")

        # Quit with Q, ESC, or the window close button.

        if key in (ord("q"), ord("Q"), 27):
            break

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:

    # Cleanup, even after a crash, so the camera is freed
    # and any recording is saved properly.

    if writer is not None:
        writer.release()

    camera.release()

    hands.close()

    if segmenter is not None:
        segmenter.close()

    cv2.destroyAllWindows()

    print("Finished.")
