"""
Real-time детекція грифа + ладів + порожка (nut) з вебкамери.
- Бере точну лінію кожного ладу і порожка з YOLO-segmentation масок
- Нумерує лади від nut'а: 1, 2, 3, 4, 5, ...
- Будує матрицю (fret x string) — координати кожної клітинки

Залежності:
    pip install ultralytics opencv-python numpy

Запуск:
    python guitar_fret_matrix.py --weights best.pt --source 0
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO


# ========== Налаштування ==========
NUM_STRINGS = 6
NECK_CLASS_NAME = "neck"
FRET_CLASS_NAME = "fret"
NUT_CLASS_NAME = "nut"          # клас порожка
CONF_THRESHOLD = 0.4
STRING_EDGE_MARGIN = 0.08       # відступ крайніх струн від країв ладу
# ===================================


def fit_line_to_points(points: np.ndarray):
    if len(points) < 2:
        return None
    points = points.astype(np.float32)
    return cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()


def line_endpoints_from_polygon(polygon: np.ndarray):
    """З полігона маски — кінці центральної лінії."""
    fit = fit_line_to_points(polygon)
    if fit is None:
        return None
    vx, vy, x0, y0 = fit
    xs, ys = polygon[:, 0], polygon[:, 1]
    t = (xs - x0) * vx + (ys - y0) * vy
    t_min, t_max = float(t.min()), float(t.max())
    p1 = (int(round(x0 + t_min * vx)), int(round(y0 + t_min * vy)))
    p2 = (int(round(x0 + t_max * vx)), int(round(y0 + t_max * vy)))
    return p1, p2


def get_neck_axis(neck_polygon: np.ndarray):
    """PCA маски грифа: центр + вектори вздовж і поперек."""
    points = neck_polygon.astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    center = mean[0]
    direction = eigenvectors[0]      # вздовж грифа
    perpendicular = eigenvectors[1]  # вздовж ладів
    return center, direction, perpendicular


def project_on_axis(point, center, direction):
    return float((point[0] - center[0]) * direction[0] +
                 (point[1] - center[1]) * direction[1])


def normalize_line_orientation(lines):
    """
    Узгоджує орієнтацію ліній: щоб p1 у всіх ліній був з одного боку грифа.
    """
    if not lines:
        return lines
    ref_p1 = np.array(lines[0][0], dtype=np.float32)
    normalized = [lines[0]]
    for p1, p2 in lines[1:]:
        d1 = np.linalg.norm(np.array(p1) - ref_p1)
        d2 = np.linalg.norm(np.array(p2) - ref_p1)
        if d2 < d1:
            p1, p2 = p2, p1
        normalized.append((p1, p2))
    return normalized


def build_fret_string_matrix(all_lines, num_strings, edge_margin=0.08):
    """
    all_lines: [nut, fret1, fret2, ...] вже відсортовані вздовж грифа.
    Повертає:
        matrix: (num_intervals, num_strings, 2) — центри клітинок
        string_lines: лінії струн
    """
    if len(all_lines) < 2:
        return None, None

    lines = normalize_line_orientation(all_lines)
    ts = np.linspace(edge_margin, 1 - edge_margin, num_strings)

    string_points_per_line = []
    for p1, p2 in lines:
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        pts = [(1 - t) * p1 + t * p2 for t in ts]
        string_points_per_line.append(pts)

    string_points_per_line = np.array(string_points_per_line)

    num_intervals = len(lines) - 1
    matrix = np.zeros((num_intervals, num_strings, 2), dtype=np.float32)
    for i in range(num_intervals):
        for j in range(num_strings):
            matrix[i, j] = (string_points_per_line[i, j] +
                            string_points_per_line[i + 1, j]) / 2.0

    string_lines = []
    for j in range(num_strings):
        sp1 = tuple(string_points_per_line[0, j].astype(int))
        sp2 = tuple(string_points_per_line[-1, j].astype(int))
        string_lines.append((sp1, sp2))

    return matrix, string_lines


def process_frame(frame, model):
    results = model(frame, verbose=False)
    r = results[0]

    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        cv2.putText(frame, "No detections", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    names = model.names
    neck_polygons = []
    fret_polygons = []
    nut_polygons = []

    for i in range(len(r.boxes)):
        cls_id = int(r.boxes.cls[i])
        cls_name = names[cls_id]
        conf = float(r.boxes.conf[i])
        if conf < CONF_THRESHOLD:
            continue
        polygon = r.masks.xy[i]
        if len(polygon) < 3:
            continue

        if cls_name == NECK_CLASS_NAME:
            neck_polygons.append((polygon, conf))
        elif cls_name == FRET_CLASS_NAME:
            fret_polygons.append((polygon, conf))
        elif cls_name == NUT_CLASS_NAME:
            nut_polygons.append((polygon, conf))

    if not neck_polygons:
        cv2.putText(frame, "Neck not detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    # Найбільший гриф у кадрі
    neck_polygon, _ = max(neck_polygons,
                          key=lambda x: cv2.contourArea(x[0].astype(np.float32)))
    neck_center, direction, _ = get_neck_axis(neck_polygon)

    if not fret_polygons:
        cv2.putText(frame, "No frets detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return frame

    # Лінії ладів
    fret_lines = []
    for polygon, _ in fret_polygons:
        ep = line_endpoints_from_polygon(polygon)
        if ep is not None:
            fret_lines.append(ep)

    if not fret_lines:
        return frame

    # Лінія nut'а
    nut_line = None
    if nut_polygons:
        nut_polygon, _ = max(nut_polygons, key=lambda x: x[1])
        nut_line = line_endpoints_from_polygon(nut_polygon)

    # Сортуємо всі лінії вздовж осі грифа
    def line_position(line):
        cx = (line[0][0] + line[1][0]) / 2
        cy = (line[0][1] + line[1][1]) / 2
        return project_on_axis((cx, cy), neck_center, direction)

    all_lines = list(fret_lines)
    if nut_line is not None:
        all_lines.append(nut_line)
    all_lines.sort(key=line_position, reverse=True)

    # nut має бути першим у списку — якщо він в кінці, перевертаємо
    if nut_line is not None:
        nut_idx = all_lines.index(nut_line)
        if nut_idx == len(all_lines) - 1:
            all_lines.reverse()
        elif nut_idx != 0:
            cv2.putText(frame, "WARN: nut not at edge", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Малюємо nut
    if nut_line is not None and all_lines and all_lines[0] is nut_line:
        np1, np2 = nut_line
        cv2.line(frame, np1, np2, (0, 255, 255), 4)
        mid = ((np1[0] + np2[0]) // 2, (np1[1] + np2[1]) // 2)
        cv2.putText(frame, "NUT", (mid[0] + 8, mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        fret_lines_sorted = all_lines[1:]
    else:
        fret_lines_sorted = all_lines

    # Лади з нумерацією від nut'а
    for idx, (p1, p2) in enumerate(fret_lines_sorted, start=1):
        cv2.line(frame, p1, p2, (0, 0, 255), 2)
        cv2.putText(frame, str(idx), (p1[0] + 6, p1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Матриця струн × ладів
    matrix, string_lines = build_fret_string_matrix(
        all_lines, NUM_STRINGS, STRING_EDGE_MARGIN)

    if matrix is not None:
        for sp1, sp2 in string_lines:
            cv2.line(frame, sp1, sp2, (200, 200, 200), 1, cv2.LINE_AA)

        h, w = matrix.shape[:2]
        for i in range(h):
            for j in range(w):
                x, y = matrix[i, j]
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        first_label = "nut->F1" if nut_line is not None else "F1->F2"
        info = f"Intervals: {h}  Strings: {w}  First cell: {first_label}"
        cv2.putText(frame, info, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def main():
    global NUM_STRINGS, CONF_THRESHOLD

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--strings", type=int, default=NUM_STRINGS)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    NUM_STRINGS = args.strings
    CONF_THRESHOLD = args.conf

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    print(f"Opening source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source {source}")
        return

    print("Press 'q' to quit, 's' to save current frame.")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = process_frame(frame, model)
        cv2.imshow("Guitar fret matrix", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"frame_{frame_idx:04d}.png"
            cv2.imwrite(fname, out)
            print(f"Saved: {fname}")
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()