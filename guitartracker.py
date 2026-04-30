"""
Real-time детекція грифа гітари з трекінгом ладів через метричну модель.

Ключові ідеї:
- Локальна 1D-система координат уздовж осі грифа (через PCA маски neck)
- Метрична модель ладів за формулою рівномірно темперованого строю:
      s(k) = scale_length * (1 - (1/2)^(k/12))
  де k — номер ладу від nut, scale_length — довжина мензури в "одиницях осі"
- Nut — головний якір (s=0). Якщо nut не детектовано — край маски neck
  з потрібного боку береться як fallback nut
- Зниклі лади переносяться з пам'яті: їхня локальна координата s
  стабільна, в кадр перепроектується через поточне PCA
- False positives відфільтровуються по відхиленню від метричної моделі

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
NUT_CLASS_NAME = "nut"
CONF_THRESHOLD = 0.4
STRING_EDGE_MARGIN = 0.08          # відступ крайніх струн від країв ладу

# Трекінг
FORGET_AFTER_FRAMES = 60           # скільки кадрів тримати лад в пам'яті без детекції
MAX_FRET_NUMBER = 22               # максимальний номер ладу на гітарі
METRIC_MATCH_TOLERANCE = 0.15      # допустиме відхилення детекції від очікуваної
                                   # позиції за метричною моделлю (частка
                                   # відстані до сусіднього ладу)
MIN_FRETS_FOR_CALIB = 3            # мінімум видимих ладів для калібрування scale_length
SCALE_EMA_ALPHA = 0.2              # коеф. експ. згладжування для scale_length

# Візуалізація
COLOR_DETECTED_FRET = (0, 0, 255)        # червоний — детектовані лади
COLOR_PREDICTED_FRET = (0, 165, 255)     # помаранчевий — лади з пам'яті
COLOR_NUT = (0, 255, 255)                # жовтий — nut
COLOR_NUT_FALLBACK = (0, 200, 200)       # темно-жовтий — fallback nut з краю neck
COLOR_STRING = (200, 200, 200)
COLOR_CELL = (0, 255, 0)
# ===================================


# ---------- Геометрія ----------

def fit_line_to_points(points: np.ndarray):
    if len(points) < 2:
        return None
    return cv2.fitLine(points.astype(np.float32),
                       cv2.DIST_L2, 0, 0.01, 0.01).flatten()


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
    """
    PCA маски грифа.
    Повертає center, direction (вздовж грифа), perpendicular (вздовж ладів).
    Знак direction нормалізуємо детерміновано: проекція на (1, 1) має бути
    додатньою. Це означає "напрямок управо-вниз" у пікселях кадру.
    Так напрямок стає стабільним між кадрами незалежно від випадкового
    знаку, що повертає PCA.
    """
    points = neck_polygon.astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None)
    center = mean[0]
    direction = eigenvectors[0]
    perpendicular = eigenvectors[1]

    # Детермінація знаку direction
    if direction[0] + direction[1] < 0:
        direction = -direction
        perpendicular = -perpendicular

    return center, direction, perpendicular


def project_on_axis(point, center, direction):
    return float((point[0] - center[0]) * direction[0] +
                 (point[1] - center[1]) * direction[1])


def project_perpendicular(point, center, perpendicular):
    return float((point[0] - center[0]) * perpendicular[0] +
                 (point[1] - center[1]) * perpendicular[1])


def world_from_local(s, t, center, direction, perpendicular):
    """Перетворити (s, t) у локальній системі грифа в піксельні координати."""
    x = center[0] + s * direction[0] + t * perpendicular[0]
    y = center[1] + s * direction[1] + t * perpendicular[1]
    return (int(round(x)), int(round(y)))


def line_from_local_s(s, t_min, t_max, center, direction, perpendicular):
    """Побудувати лінію (поперек грифа) на координаті s."""
    p1 = world_from_local(s, t_min, center, direction, perpendicular)
    p2 = world_from_local(s, t_max, center, direction, perpendicular)
    return p1, p2


# ---------- Метрична модель ----------

def expected_s(fret_number: int, scale_length: float) -> float:
    """Очікувана координата ладу від nut за рівномірно темперованим строєм."""
    return scale_length * (1.0 - (0.5 ** (fret_number / 12.0)))


def calibrate_scale_length(fret_s_values, fret_numbers):
    """
    Метод найменших квадратів: шукаємо scale_length таке, щоб
    сума (s_observed - expected_s(k, scale))^2 була мінімальною.
    Розв'язок аналітичний (лінійна задача по scale):
        scale = sum(s_i * f_i) / sum(f_i^2)
    де f_i = (1 - (1/2)^(k_i/12))
    """
    if len(fret_s_values) < 2:
        return None
    s_arr = np.array(fret_s_values, dtype=np.float64)
    k_arr = np.array(fret_numbers, dtype=np.float64)
    f_arr = 1.0 - np.power(0.5, k_arr / 12.0)
    denom = float(np.sum(f_arr ** 2))
    if denom < 1e-9:
        return None
    return float(np.sum(s_arr * f_arr) / denom)


def assign_fret_numbers(detected_s, scale_length, max_fret=22, tolerance=0.15):
    """
    Зіставлення кожній виявленій позиції найближчого номера ладу
    за метричною моделлю.

    detected_s: список локальних координат ладів (відносно nut, s=0)
    Повертає dict {fret_number: s_observed} і список s, що відкинуті як шум.
    """
    if scale_length is None or scale_length <= 0:
        return {}, list(detected_s)

    expected = {k: expected_s(k, scale_length) for k in range(1, max_fret + 1)}

    assignments = {}
    rejected = []
    used_numbers = set()

    # Сортуємо детекції за s, щоб робити жадібне присвоєння в порядку
    sorted_dets = sorted(detected_s)
    for s in sorted_dets:
        # Знайти найближчий очікуваний номер
        best_k = None
        best_dist = float('inf')
        for k, s_exp in expected.items():
            if k in used_numbers:
                continue
            d = abs(s - s_exp)
            if d < best_dist:
                best_dist = d
                best_k = k

        if best_k is None:
            rejected.append(s)
            continue

        # Перевірка толерантності: відстань має бути менша за частку
        # типової відстані між сусідніми ладами в цій зоні.
        if best_k < max_fret:
            local_spacing = expected[best_k + 1] - expected[best_k]
        else:
            local_spacing = expected[best_k] - expected[best_k - 1]
        if best_dist > tolerance * local_spacing:
            rejected.append(s)
            continue

        assignments[best_k] = s
        used_numbers.add(best_k)

    return assignments, rejected


# ---------- Стан трекера ----------

class FretTracker:
    def __init__(self):
        self.scale_length = None         # довжина мензури в "одиницях осі"
        self.fret_memory = {}            # {fret_number: {'s': float, 'last_seen': int}}
        self.frame_idx = 0
        self.t_min = -50.0               # межі грифа в перпендикулярному напрямку
        self.t_max = 50.0                # (для малювання ліній)

    def update(self, detected_s_list, t_range, nut_s, frame_idx):
        """
        detected_s_list: список локальних s видимих ладів (без врахування nut)
                         де s рахується ВІД nut (тобто nut_s вже віднято)
        t_range: (t_min, t_max) — межі грифа в перпендикулярному напрямку
        nut_s: координата nut в локальній системі (вже взята як 0 в detected_s_list)
        """
        self.frame_idx = frame_idx
        self.t_min, self.t_max = t_range

        # Калібрування / уточнення scale_length: пробуємо з поточних детекцій
        # Спочатку грубе припущення: пронумеровуємо детекції за порядком і
        # робимо первинне калібрування. Потім беремо найкраще присвоєння.
        if len(detected_s_list) >= MIN_FRETS_FOR_CALIB:
            sorted_s = sorted(detected_s_list)
            # Гіпотеза: ці відсортовані s — це лади 1, 2, 3, ..., N
            # (валідно якщо nut є якорем; для перших ладів ця гіпотеза
            # точна, для далеких — приблизна, але все одно дає розумне
            # початкове scale_length, яке далі уточнюється).
            naive_numbers = list(range(1, len(sorted_s) + 1))
            scale_candidate = calibrate_scale_length(sorted_s, naive_numbers)
            if scale_candidate is not None and scale_candidate > 0:
                if self.scale_length is None:
                    self.scale_length = scale_candidate
                else:
                    # експ. згладжування — повільне адаптування
                    self.scale_length = ((1 - SCALE_EMA_ALPHA) * self.scale_length
                                         + SCALE_EMA_ALPHA * scale_candidate)

        # Якщо scale_length все ще немає (мало детекцій на старті) — нічого не вдієш
        if self.scale_length is None or self.scale_length <= 0:
            return {}

        # Призначення номерів детекціям через метричну модель
        assignments, rejected = assign_fret_numbers(
            detected_s_list, self.scale_length,
            max_fret=MAX_FRET_NUMBER,
            tolerance=METRIC_MATCH_TOLERANCE,
        )

        # Уточнюємо scale_length на основі правильно присвоєних номерів
        if len(assignments) >= MIN_FRETS_FOR_CALIB:
            refined = calibrate_scale_length(
                list(assignments.values()),
                list(assignments.keys()),
            )
            if refined is not None and refined > 0:
                self.scale_length = ((1 - SCALE_EMA_ALPHA) * self.scale_length
                                     + SCALE_EMA_ALPHA * refined)

        # Оновлюємо пам'ять
        for k, s_obs in assignments.items():
            self.fret_memory[k] = {'s': s_obs, 'last_seen': frame_idx}

        # Видаляємо старі
        to_delete = [k for k, info in self.fret_memory.items()
                     if frame_idx - info['last_seen'] > FORGET_AFTER_FRAMES]
        for k in to_delete:
            del self.fret_memory[k]

        return assignments

    def get_all_frets(self):
        """
        Повертає {fret_number: (s_local, is_predicted)} для всіх ладів,
        яких ми знаємо: ті що є в пам'яті + ті, що між ними можна заповнити
        за метричною моделлю.
        """
        result = {}
        if self.scale_length is None:
            return result

        # 1. Те, що в пам'яті
        for k, info in self.fret_memory.items():
            is_predicted = (info['last_seen'] != self.frame_idx)
            result[k] = (info['s'], is_predicted)

        # 2. Заповнюємо дірки: лади між min і max з пам'яті за метрикою
        if self.fret_memory:
            min_k = min(self.fret_memory.keys())
            max_k = max(self.fret_memory.keys())
            for k in range(min_k, max_k + 1):
                if k not in result:
                    result[k] = (expected_s(k, self.scale_length), True)

        return result


# ---------- Обробка кадру ----------

def process_frame(frame, model, tracker: FretTracker):
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

    # Найбільша маска грифа
    neck_polygon, _ = max(neck_polygons,
                          key=lambda x: cv2.contourArea(x[0].astype(np.float32)))
    neck_center, direction, perpendicular = get_neck_axis(neck_polygon)

    # Межі грифа в перпендикулярному напрямку (для малювання ліній на повну ширину)
    neck_t = [project_perpendicular(p, neck_center, perpendicular)
              for p in neck_polygon]
    t_min, t_max = float(np.min(neck_t)), float(np.max(neck_t))

    # ---------- Знаходимо nut ----------
    nut_s_in_neck = None     # s nut у системі координат з центром у neck_center
    nut_is_fallback = False

    if nut_polygons:
        # Беремо найвпевненіший nut
        nut_polygon, _ = max(nut_polygons, key=lambda x: x[1])
        nut_endpoints = line_endpoints_from_polygon(nut_polygon)
        if nut_endpoints is not None:
            np1, np2 = nut_endpoints
            nut_center_pt = ((np1[0] + np2[0]) / 2.0, (np1[1] + np2[1]) / 2.0)
            nut_s_in_neck = project_on_axis(nut_center_pt, neck_center, direction)

    if nut_s_in_neck is None:
        # Fallback: край маски neck з боку, де "має бути nut".
        # За домовленістю — це край з МЕНШИМ значенням s
        # (бо direction обрано так, що s зростає від nut'а до корпусу).
        neck_s = [project_on_axis(p, neck_center, direction)
                  for p in neck_polygon]
        nut_s_in_neck = float(np.min(neck_s))
        nut_is_fallback = True

    # ---------- Локальні координати ладів (відносно nut) ----------
    fret_s_local = []  # уже відносно nut: s=0 це nut
    fret_pixel_centers = []
    for polygon, _ in fret_polygons:
        ep = line_endpoints_from_polygon(polygon)
        if ep is None:
            continue
        p1, p2 = ep
        cx = (p1[0] + p2[0]) / 2.0
        cy = (p1[1] + p2[1]) / 2.0
        s_in_neck = project_on_axis((cx, cy), neck_center, direction)
        s_from_nut = s_in_neck - nut_s_in_neck
        if s_from_nut <= 0:
            continue  # не може бути ладу з боку nut'а
        fret_s_local.append(s_from_nut)
        fret_pixel_centers.append((cx, cy))

    # ---------- Оновлюємо трекер ----------
    tracker.update(fret_s_local, (t_min, t_max), 0.0, tracker.frame_idx + 1)

    # ---------- Малюємо ----------

    # Контур грифа
    cv2.polylines(frame, [neck_polygon.astype(np.int32)],
                  True, (255, 200, 0), 1)

    # nut
    nut_color = COLOR_NUT_FALLBACK if nut_is_fallback else COLOR_NUT
    nut_label = "NUT (edge)" if nut_is_fallback else "NUT"
    nut_p1, nut_p2 = line_from_local_s(
        nut_s_in_neck, t_min, t_max,
        neck_center, direction, perpendicular,
    )
    cv2.line(frame, nut_p1, nut_p2, nut_color, 4)
    nut_mid = ((nut_p1[0] + nut_p2[0]) // 2, (nut_p1[1] + nut_p2[1]) // 2)
    cv2.putText(frame, nut_label, (nut_mid[0] + 8, nut_mid[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, nut_color, 2)

    # Усі лади з трекера
    all_frets = tracker.get_all_frets()  # {k: (s_local, is_predicted)}
    drawn_lines_for_matrix = [(0, (nut_p1, nut_p2))]  # (k, (p1, p2)) — k=0 це nut

    for k in sorted(all_frets.keys()):
        s_from_nut, is_predicted = all_frets[k]
        s_in_neck = nut_s_in_neck + s_from_nut
        p1, p2 = line_from_local_s(s_in_neck, t_min, t_max,
                                   neck_center, direction, perpendicular)
        color = COLOR_PREDICTED_FRET if is_predicted else COLOR_DETECTED_FRET
        thickness = 2
        cv2.line(frame, p1, p2, color, thickness)
        cv2.putText(frame, str(k), (p1[0] + 6, p1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        drawn_lines_for_matrix.append((k, (p1, p2)))

    # ---------- Матриця струн × ладів ----------

    if len(drawn_lines_for_matrix) >= 2:
        # Узгоджуємо орієнтацію кінців ліній (щоб p1 у всіх з одного боку)
        ref_p1 = np.array(drawn_lines_for_matrix[0][1][0], dtype=np.float32)
        normalized = []
        for k, (p1, p2) in drawn_lines_for_matrix:
            d1 = np.linalg.norm(np.array(p1) - ref_p1)
            d2 = np.linalg.norm(np.array(p2) - ref_p1)
            if d2 < d1:
                p1, p2 = p2, p1
            normalized.append((k, (p1, p2)))

        ts = np.linspace(STRING_EDGE_MARGIN, 1 - STRING_EDGE_MARGIN, NUM_STRINGS)
        string_points = []
        for _, (p1, p2) in normalized:
            p1n = np.array(p1, dtype=np.float32)
            p2n = np.array(p2, dtype=np.float32)
            pts = [(1 - t) * p1n + t * p2n for t in ts]
            string_points.append(pts)
        string_points = np.array(string_points)  # (num_lines, num_strings, 2)

        # Лінії струн (від першої лінії до останньої)
        for j in range(NUM_STRINGS):
            sp1 = tuple(string_points[0, j].astype(int))
            sp2 = tuple(string_points[-1, j].astype(int))
            cv2.line(frame, sp1, sp2, COLOR_STRING, 1, cv2.LINE_AA)

        # Центри клітинок
        for i in range(len(normalized) - 1):
            for j in range(NUM_STRINGS):
                pt = (string_points[i, j] + string_points[i + 1, j]) / 2.0
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, COLOR_CELL, -1)

    # ---------- Інфо-панель ----------

    detected_count = sum(1 for k, (_, pred) in all_frets.items() if not pred)
    predicted_count = sum(1 for k, (_, pred) in all_frets.items() if pred)
    info = (f"Frets: detected={detected_count} predicted={predicted_count} "
            f"scale={tracker.scale_length:.1f}"
            if tracker.scale_length else
            f"Calibrating... visible={len(fret_s_local)}")
    cv2.putText(frame, info, (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if nut_is_fallback:
        cv2.putText(frame, "Using neck edge as nut fallback",
                    (20, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_NUT_FALLBACK, 1)

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

    tracker = FretTracker()

    print("Press 'q' to quit, 's' to save current frame, 'r' to reset tracker.")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = process_frame(frame, model, tracker)
        cv2.imshow("Guitar fret matrix", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"frame_{frame_idx:04d}.png"
            cv2.imwrite(fname, out)
            print(f"Saved: {fname}")
        elif key == ord('r'):
            tracker = FretTracker()
            print("Tracker reset")
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()