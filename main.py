import os
os.environ["GLOG_minloglevel"] = "2"

import cv2
import mediapipe as mp
import numpy as np
import math
import time

# live spider-style face effect 

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

smooth_cx = smooth_cy = None
smooth_scale = 1.0
phase = 0.0

recording = False
writer = None


# helpers

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(a, b, amount):
    return a + (b - a) * amount


def rotate_point(cx, cy, x, y, angle):
    a = math.radians(angle)
    dx = x - cx
    dy = y - cy

    return (
        int(cx + dx * math.cos(a) - dy * math.sin(a)),
        int(cy + dx * math.sin(a) + dy * math.cos(a))
    )


def quad_points(cx, cy, w, h, angle):
    corners = [
        (cx - w // 2, cy - h // 2),
        (cx + w // 2, cy - h // 2),
        (cx + w // 2, cy + h // 2),
        (cx - w // 2, cy + h // 2)
    ]

    return np.array(
        [rotate_point(cx, cy, x, y, angle) for x, y in corners],
        dtype=np.int32
    )


def glow_line(image, p1, p2, thickness=1):
    # very subtle glow — no haze.
    glow = image.copy()

    cv2.line(
        glow, p1, p2,
        (255, 255, 255),
        thickness * 3,
        cv2.LINE_AA
    )

    glow = cv2.GaussianBlur(glow, (0, 0), 2)

    cv2.addWeighted(
        glow, 0.12,
        image, 0.88,
        0,
        image
    )

    cv2.line(
        image, p1, p2,
        (245, 245, 245),
        thickness,
        cv2.LINE_AA
    )


def get_face_data(face, width, height):
    points = []

    for p in face.landmark:
        points.append((
            int(np.clip(p.x * width, 0, width - 1)),
            int(np.clip(p.y * height, 0, height - 1))
        ))

    pts = np.array(points, dtype=np.int32)

    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0]))
    y2 = int(np.max(pts[:, 1]))

    return pts, (x1, y1, x2, y2)


def create_face_effect(frame, face_points, box, strong):
    """
    clean reference-style glitch:
    - normal mode = sharp camera image
    - pinch = enlarged face-area scan window
    - horizontal rgb displacement + broken slices
    - keeps the actual face visible underneath
    """

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)

    # a bit wider than the older thin strip, but still centered on the face.
    px = int(fw * 0.28)
    py = int(fh * 0.42)

    rx1 = max(0, x1 - px)
    ry1 = max(0, y1 - py)
    rx2 = min(w, x2 + px)
    ry2 = min(h, y2 + py)

    crop = frame[ry1:ry2, rx1:rx2].copy()

    if crop.size == 0:
        return frame

    ch, cw = crop.shape[:2]

    # face mask.
    local = face_points.copy()
    local[:, 0] -= rx1
    local[:, 1] -= ry1

    face_mask = np.zeros((ch, cw), dtype=np.uint8)
    hull = cv2.convexHull(local)
    cv2.fillConvexPoly(face_mask, hull, 255)

    if not strong:
        return frame

    # --------------------------------------------------------
    # darken + sharpen the face area
    # --------------------------------------------------------

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    base = crop.copy()

    # slightly boost contrast without blurring.
    enhanced = cv2.convertScaleAbs(
        base,
        alpha=1.15,
        beta=-18
    )

    # --------------------------------------------------------
    # rgb channel separation
    # --------------------------------------------------------

    b, g, r = cv2.split(enhanced)

    shift_r = max(5, cw // 22)
    shift_b = max(5, cw // 26)

    r_shift = np.roll(r, shift_r, axis=1)
    b_shift = np.roll(b, -shift_b, axis=1)

    glitched = cv2.merge((b_shift, g, r_shift))

    # --------------------------------------------------------
    # horizontal slice displacement
    # --------------------------------------------------------

    rng = np.random.default_rng()

    # lots of short slices, but not every pixel.
    for _ in range(55):

        yy = int(rng.integers(0, ch))
        thickness = int(rng.integers(1, max(2, ch // 32)))

        y2s = min(ch, yy + thickness)

        # stronger displacement around the face.
        offset = int(
            rng.integers(
                -max(5, cw // 12),
                max(6, cw // 12)
            )
        )

        glitched[yy:y2s] = np.roll(
            glitched[yy:y2s],
            offset,
            axis=1
        )

    # --------------------------------------------------------
    # scanlines
    # --------------------------------------------------------

    line_spacing = max(5, ch // 38)

    for yy in range(0, ch, line_spacing):

        cv2.line(
            glitched,
            (0, yy),
            (cw, yy),
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # digital horizontal bars
    # --------------------------------------------------------

    for _ in range(18):

        yy = int(rng.integers(0, ch))
        bar_h = int(rng.integers(1, 4))
        bar_w = int(rng.integers(max(12, cw // 10), max(13, cw // 2)))

        xx = int(rng.integers(0, max(1, cw - bar_w)))

        cv2.rectangle(
            glitched,
            (xx, yy),
            (min(cw - 1, xx + bar_w), min(ch - 1, yy + bar_h)),
            (235, 235, 235),
            -1
        )

    # --------------------------------------------------------
    # composite
    # --------------------------------------------------------

    # the effect is strongest on the face and weaker around it.
    face_alpha = face_mask.astype(np.float32) / 255.0

    # keep the surrounding rectangle subtle.
    outer_mask = cv2.GaussianBlur(face_mask, (0, 0), 12)
    outer_alpha = outer_mask.astype(np.float32) / 255.0

    alpha = (
        face_alpha * 0.88 +
        outer_alpha * 0.12
    )[:, :, None]

    base_f = base.astype(np.float32)
    glitch_f = glitched.astype(np.float32)

    result = (
        base_f * (1.0 - alpha) +
        glitch_f * alpha
    ).astype(np.uint8)

    frame[ry1:ry2, rx1:rx2] = result

    return frame


def draw_face_hud(frame, box, angle=0):
    x1, y1, x2, y2 = box

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    fw = x2 - x1
    fh = y2 - y1

    # the hud stays close to the face instead of floating around like a huge screen.
    hw = int(fw * 1.65)
    hh = int(fh * 1.75)

    pts = quad_points(
        cx,
        cy,
        hw,
        hh,
        angle
    )

    # thin, clean frame.
    for i in range(4):
        glow_line(
            frame,
            tuple(pts[i]),
            tuple(pts[(i + 1) % 4]),
            1
        )

    # short technical marks at the corners.
    for p in pts:
        px, py = p

        cv2.circle(
            frame,
            (px, py),
            3,
            (245, 245, 245),
            -1,
            cv2.LINE_AA
        )


# start

print()
print("==========================================")
print("       LIVE SPIDER FACE EFFECT")
print("==========================================")
print()
print("1 HAND  -> move face HUD")
print("2 HANDS -> resize + rotate HUD")
print("PINCH   -> digital face mode")
print()
print("R = record")
print("Q = quit")
print()


# main loop

while True:

    ok, frame = cap.read()

    if not ok:
        break

    frame = cv2.flip(frame, 1)

    original = frame.copy()

    height, width = frame.shape[:2]

    phase += 0.075

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    # --------------------------------------------------------
    # face
    # --------------------------------------------------------

    face_points = None
    face_box = None

    if face_results.multi_face_landmarks:

        face_points, face_box = get_face_data(
            face_results.multi_face_landmarks[0],
            width,
            height
        )

    # --------------------------------------------------------
    # hands
    # --------------------------------------------------------

    fingertips = []
    pinch = False

    if hand_results.multi_hand_landmarks:

        for hand in hand_results.multi_hand_landmarks:

            index = hand.landmark[8]
            thumb = hand.landmark[4]

            wrist = hand.landmark[0]
            middle = hand.landmark[9]

            ip = (
                int(index.x * width),
                int(index.y * height)
            )

            tp = (
                int(thumb.x * width),
                int(thumb.y * height)
            )

            wp = (
                int(wrist.x * width),
                int(wrist.y * height)
            )

            mpnt = (
                int(middle.x * width),
                int(middle.y * height)
            )

            fingertips.append(ip)

            hand_size = max(
                distance(wp, mpnt),
                1
            )

            if distance(ip, tp) < hand_size * 0.55:
                pinch = True

    # --------------------------------------------------------
    # hud control
    # --------------------------------------------------------

    if face_box:

        fx1, fy1, fx2, fy2 = face_box

        target_cx = (fx1 + fx2) // 2
        target_cy = (fy1 + fy2) // 2

        target_scale = 1.0

    else:

        target_cx = width // 2
        target_cy = height // 2
        target_scale = 1.0

    angle = 0

    if len(fingertips) >= 2:

        p1, p2 = fingertips[:2]

        target_cx = (p1[0] + p2[0]) // 2
        target_cy = (p1[1] + p2[1]) // 2

        finger_distance = distance(
            p1,
            p2
        )

        target_scale = float(
            np.clip(
                finger_distance / 250.0,
                0.65,
                1.7
            )
        )

        angle = math.degrees(
            math.atan2(
                p2[1] - p1[1],
                p2[0] - p1[0]
            )
        )

    elif len(fingertips) == 1:

        target_cx, target_cy = fingertips[0]

    # --------------------------------------------------------
    # smooth
    # --------------------------------------------------------

    if smooth_cx is None:

        smooth_cx = target_cx
        smooth_cy = target_cy

    smooth_cx = lerp(
        smooth_cx,
        target_cx,
        0.16
    )

    smooth_cy = lerp(
        smooth_cy,
        target_cy,
        0.16
    )

    smooth_scale = lerp(
        smooth_scale,
        target_scale,
        0.12
    )

    # --------------------------------------------------------
    # digital face effect
    # --------------------------------------------------------

    if face_points is not None and face_box is not None:

        if pinch:

            frame = create_face_effect(
                frame,
                face_points,
                face_box,
                True
            )

        else:

            frame = create_face_effect(
                frame,
                face_points,
                face_box,
                False
            )

        # the hud follows the face by default.
        hud_box = face_box

        # if hands are controlling it, move the hud center.
        if fingertips:

            x1, y1, x2, y2 = face_box

            fw = x2 - x1
            fh = y2 - y1

            new_cx = int(smooth_cx)
            new_cy = int(smooth_cy)

            hud_w = int(fw * 1.65 * smooth_scale)
            hud_h = int(fh * 1.75 * smooth_scale)

            hud_box = (
                new_cx - hud_w // 2,
                new_cy - hud_h // 2,
                new_cx + hud_w // 2,
                new_cy + hud_h // 2
            )

        if pinch and face_box:
            x1, y1, x2, y2 = face_box
            cx_face = (x1 + x2) // 2
            cy_face = (y1 + y2) // 2

            glitch_w = int((x2 - x1) * 1.56)
            glitch_h = int((y2 - y1) * 1.82)

            scan_pts = quad_points(
                cx_face,
                cy_face,
                glitch_w,
                glitch_h,
                angle
            )

            for i in range(4):
                glow_line(
                    frame,
                    tuple(scan_pts[i]),
                    tuple(scan_pts[(i + 1) % 4]),
                    1
                )
        else:
            draw_face_hud(
                frame,
                hud_box,
                angle
            )

    # --------------------------------------------------------
    # hand markers
    # --------------------------------------------------------

    for p in fingertips:

        cv2.circle(
            frame,
            p,
            8,
            (240, 240, 240),
            1,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            p,
            2,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # minimal status
    # --------------------------------------------------------

    if pinch:
        status = "GLITCH // ACTIVE"
    elif len(fingertips) >= 2:
        status = "TWO HAND"
    elif len(fingertips) == 1:
        status = "HAND"
    elif face_box:
        status = "FACE LOCK"
    else:
        status = "SEARCH"

    cv2.putText(
        frame,
        status,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (235, 235, 235),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # record
    # --------------------------------------------------------

    if recording:

        cv2.circle(
            frame,
            (width - 25, 25),
            7,
            (255, 255, 255),
            -1
        )

        cv2.putText(
            frame,
            "REC",
            (width - 65, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA
        )

        writer.write(frame)

    # --------------------------------------------------------
    # display
    # --------------------------------------------------------

    cv2.imshow(
        "Spider Live Effect",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # record toggle
    # --------------------------------------------------------

    if key == ord("r"):

        recording = not recording

        if recording:

            filename = (
                "spider_effect_"
                + time.strftime("%Y%m%d_%H%M%S")
                + ".mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                filename,
                fourcc,
                30,
                (width, height)
            )

            print("Recording:", filename)

        else:

            if writer is not None:

                writer.release()
                writer = None

            print("Recording stopped.")

    # --------------------------------------------------------
    # quit
    # --------------------------------------------------------

    if key == ord("q"):
        break


# cleanup

if writer is not None:
    writer.release()

cap.release()

hands.close()
face_mesh.close()

cv2.destroyAllWindows()

print("Finished.")
