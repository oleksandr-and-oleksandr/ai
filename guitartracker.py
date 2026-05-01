"""
Real-time детекція грифа + ладів + порожка (nut) з вебкамери.
- Бере точну лінію кожного ладу і порожка з YOLO-segmentation масок
- Нумерує лади від nut'а: 1, 2, 3, 4, 5, ...
- Будує матрицю (fret x string) — координати кожної клітинки
- ДОДАНО: математичний скелет ладів за рівномірним темпераментом.
  Поки nut в кадрі — скелет калібрується по nut + видимих ладах.
  Коли nut зникає — скелет переноситься за рухом видимих ладів.

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
STRING_EDGE_MARGIN = 0.08

# Скелет
MAX_FRET_NUMBER = 22                # скільки ладів моделюємо в скелеті
SKELETON_FIT_TOLERANCE = 0.20       # макс. відхилення детекції від скелета
                                    # (частка типової відстані між сусідами)
MIN_FRETS_FOR_SCALE_FIT = 2         # мінімум видимих ладів щоб зафітити scale
# ===================================


# ---------- Геометрія ----------

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
    direction = eigenvectors[0]
    perpendicular = eigenvectors[1]
    return center, direction, perpendicular


def project_on_axis(point, center, direction):
    return float((point[0] - center[0]) * direction[0] +
                 (point[1] - center[1]) * direction[1])


def project_perpendicular(point, center, perpendicular):
    return float((point[0] - center[0]) * perpendicular[0] +
                 (point[1] - center[1]) * perpendicular[1])


def world_from_local(s, t, center, direction, perpendicular):
    x = center[0] + s * direction[0] + t * perpendicular[0]
    y = center[1] + s * direction[1] + t * perpendicular[1]
    return (int(round(x)), int(round(y)))


def normalize_line_orientation(lines):
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


# ---------- Скелет за рівномірним темпераментом ----------

def expected_offset(fret_number, scale_length):
    """Відстань ладу від nut за рівномірним темпераментом."""
    return scale_length * (1.0 - (0.5 ** (fret_number / 12.0)))


def fit_scale_length(fret_offsets, fret_numbers):
    """
    Зіставляє масштаб: мінімізує sum((s_i - scale * f(k_i))^2)
    де f(k) = 1 - (1/2)^(k/12).
    Аналітичний розв'язок: scale = sum(s*f) / sum(f^2)
    """
    if len(fret_offsets) < 1:
        return None
    s = np.array(fret_offsets, dtype=np.float64)
    k = np.array(fret_numbers, dtype=np.float64)
    f = 1.0 - np.power(0.5, k / 12.0)
    denom = float(np.sum(f * f))
    if denom < 1e-9:
        return None
    return float(np.sum(s * f) / denom)


def fit_skeleton_to_detections(detected_offsets, max_fret=22, tolerance=0.20):
    """
    Підбирає таке scale_length, при якому МАКСИМАЛЬНА КІЛЬКІСТЬ
    detected_offsets (відстаней ладів від nut) лягає на математичний скелет.

    Алгоритм: припускаємо, що мінімальний detected_offset відповідає якомусь
    ладу k_min ∈ [1, max_fret]. Для кожного припущення:
      - припускаємо що інші detected_offsets — це послідовні номери далі
        (k_min, k_min+1, ...) АБО з пропусками (зіставляємо кожен з найкращим
        номером)
      - фітимо scale методом найменших квадратів
      - рахуємо скільки точок лягло в межах tolerance
    Беремо гіпотезу з максимальною кількістю inliers; при рівності — мінімальна
    помилка.

    Повертає (scale_length, list[(observed, fret_number)]) для inliers.
    """
    if len(detected_offsets) < 1:
        return None, []

    sorted_offsets = sorted(detected_offsets)
    best_scale = None
    best_inliers = []
    best_err = float('inf')

    # Гіпотеза: лад #1 знаходиться на найменшому offset.
    # Це стандартний випадок коли nut в кадрі і всі видимі лади починаються
    # від 1-го. Якщо в кадрі видно лади з 5-го, наприклад, ця гіпотеза дасть
    # завищений scale, який не пройде перевірку tolerance — і ми спробуємо
    # k_min=2, 3 і т.д.
    for k_min_first in range(1, max_fret):
        # Жадібне присвоєння: для кожної детекції шукаємо найближчий
        # очікуваний номер з гіпотетичного scale.
        # Спочатку грубий scale: припускаємо що sorted_offsets — це лади
        # k_min_first, k_min_first+1, ...
        N = len(sorted_offsets)
        if k_min_first + N - 1 > max_fret:
            break
        naive_numbers = list(range(k_min_first, k_min_first + N))
        scale_guess = fit_scale_length(sorted_offsets, naive_numbers)
        if scale_guess is None or scale_guess <= 0:
            continue

        # Тепер для кожної детекції шукаємо найближчий очікуваний лад
        # (з можливими пропусками — деякі номери можуть не співпасти).
        expected = {k: expected_offset(k, scale_guess) for k in range(1, max_fret + 1)}
        assignments = []
        used_k = set()
        for s in sorted_offsets:
            best_k = None
            best_d = float('inf')
            for k, e in expected.items():
                if k in used_k:
                    continue
                d = abs(s - e)
                if d < best_d:
                    best_d = d
                    best_k = k
            if best_k is None:
                continue
            # tolerance — частка локальної відстані між сусідніми ладами
            if best_k < max_fret:
                local_spacing = expected[best_k + 1] - expected[best_k]
            else:
                local_spacing = expected[best_k] - expected[best_k - 1]
            if best_d <= tolerance * local_spacing:
                assignments.append((s, best_k))
                used_k.add(best_k)

        if not assignments:
            continue

        # Фітимо scale ще раз, на присвоєних точках
        refined_scale = fit_scale_length(
            [a[0] for a in assignments],
            [a[1] for a in assignments],
        )
        if refined_scale is None or refined_scale <= 0:
            continue

        # Сумарна квадратична похибка
        err = sum((a[0] - expected_offset(a[1], refined_scale)) ** 2
                  for a in assignments)
        # Метрика якості: спочатку кількість inliers, потім помилка
        if (len(assignments) > len(best_inliers) or
            (len(assignments) == len(best_inliers) and err < best_err)):
            best_scale = refined_scale
            best_inliers = assignments
            best_err = err

    return best_scale, best_inliers


def fit_offset_with_fixed_scale(detected_offsets, scale_length,
                                 max_fret=22, tolerance=0.20):
    """
    Скелет вже відомий (scale_length зафіксований). Знаходимо ОФСЕТ
    у локальній системі грифа: координату nut в поточному кадрі, при
    якому скелет ладів максимально лягає на видимі лади.

    detected_offsets тут — це абсолютні s в локальній PCA-системі
    (НЕ відносно nut'а, бо ми його не бачимо).
    Повертає (offset_nut_in_current_axis, list[(observed_s, fret_number)]).
    """
    if len(detected_offsets) < 1 or scale_length is None:
        return None, []

    sorted_obs = sorted(detected_offsets)
    skeleton = {k: expected_offset(k, scale_length) for k in range(1, max_fret + 1)}

    best_offset = None
    best_inliers = []
    best_err = float('inf')

    # Перебираємо: припускаємо що мінімальний спостережуваний лад — це k_min.
    # Тоді offset_nut = obs_min - skeleton[k_min].
    obs_min = sorted_obs[0]
    for k_min in range(1, max_fret + 1):
        offset_nut = obs_min - skeleton[k_min]

        assignments = []
        used_k = set()
        for s in sorted_obs:
            # Шукаємо найближчий лад у скелеті при цьому offset
            best_k = None
            best_d = float('inf')
            for k, sk in skeleton.items():
                if k in used_k:
                    continue
                expected_s = sk + offset_nut
                d = abs(s - expected_s)
                if d < best_d:
                    best_d = d
                    best_k = k
            if best_k is None:
                continue
            if best_k < max_fret:
                local_spacing = skeleton[best_k + 1] - skeleton[best_k]
            else:
                local_spacing = skeleton[best_k] - skeleton[best_k - 1]
            if best_d <= tolerance * local_spacing:
                assignments.append((s, best_k))
                used_k.add(best_k)

        if not assignments:
            continue

        err = sum((s - (skeleton[k] + offset_nut)) ** 2 for s, k in assignments)
        if (len(assignments) > len(best_inliers) or
            (len(assignments) == len(best_inliers) and err < best_err)):
            best_offset = offset_nut
            best_inliers = assignments
            best_err = err

    return best_offset, best_inliers


# ---------- Стан скелета ----------

class SkeletonState:
    """Збережений скелет після калібрування."""
    def __init__(self):
        self.scale_length = None     # довжина мензури в одиницях осі PCA
        self.calibrated = False      # чи маємо валідний скелет
        self.reference_direction = None  # вектор direction в момент останнього
                                         # калібрування — для узгодження знаку
                                         # PCA на наступних кадрах

    def update_with_nut(self, nut_offset_in_axis, fret_offsets_in_axis,
                         current_direction):
        """
        Викликається коли nut детектовано.
        nut_offset_in_axis — s_nut в локальній PCA-системі поточного кадру
        fret_offsets_in_axis — s_fret_i в тій самій системі
        current_direction — вектор direction PCA в цьому кадрі (вже узгоджений
                            за nut'ом, тобто такий, що s зростає від nut вглиб)
        """
        relative_offsets = [s - nut_offset_in_axis for s in fret_offsets_in_axis
                            if s - nut_offset_in_axis > 0]
        _dbg(f"  update_with_nut: relative_count={len(relative_offsets)}, "
             f"min_required={MIN_FRETS_FOR_SCALE_FIT}")
        if len(relative_offsets) < MIN_FRETS_FOR_SCALE_FIT:
            _dbg(f"    FAIL: not enough positive relative offsets")
            return False

        scale, inliers = fit_skeleton_to_detections(
            relative_offsets,
            max_fret=MAX_FRET_NUMBER,
            tolerance=SKELETON_FIT_TOLERANCE,
        )
        _dbg(f"    fit result: scale={scale}, inliers={len(inliers)}")
        if scale is None or len(inliers) < MIN_FRETS_FOR_SCALE_FIT:
            _dbg(f"    FAIL: scale={scale} or inliers too few")
            return False

        self.scale_length = scale
        self.calibrated = True
        # Запам'ятовуємо узгоджений напрямок осі
        self.reference_direction = np.array(current_direction, dtype=np.float64)
        return True


# ---------- Обробка кадру ----------

_DEBUG_FRAME_COUNTER = 0
_DEBUG_EVERY_N_FRAMES = 30


def _dbg(msg):
    """Друкувати раз на N кадрів."""
    if _DEBUG_FRAME_COUNTER % _DEBUG_EVERY_N_FRAMES == 0:
        print(f"[frame {_DEBUG_FRAME_COUNTER}] {msg}")


def process_frame(frame, model, skeleton: SkeletonState):
    global _DEBUG_FRAME_COUNTER
    _DEBUG_FRAME_COUNTER += 1

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
    neck_center, direction, perpendicular = get_neck_axis(neck_polygon)

    # Узгодження знаку direction зі збереженим (якщо скелет уже калібрований).
    # Це найголовніший фікс: PCA повертає вісь, але знак випадковий між кадрами.
    # Якщо ми вже калібрувались — є збережений напрямок, до якого підлаштовуємо.
    if skeleton.calibrated and skeleton.reference_direction is not None:
        if float(np.dot(direction, skeleton.reference_direction)) < 0:
            direction = -direction
            perpendicular = -perpendicular
            _dbg(f"  axis sign aligned to skeleton reference")
        # Оновлюємо reference_direction поступово — щоб поворот гітари не
        # призводив до раптової інверсії. Просте середнє з малою вагою.
        ref = skeleton.reference_direction
        new_ref = 0.9 * ref + 0.1 * np.array(direction, dtype=np.float64)
        norm = float(np.linalg.norm(new_ref))
        if norm > 1e-9:
            skeleton.reference_direction = new_ref / norm

    # Межі грифа в перпендикулярному напрямку (для малювання ліній скелета)
    neck_t = [project_perpendicular(p, neck_center, perpendicular)
              for p in neck_polygon]
    t_min, t_max = float(np.min(neck_t)), float(np.max(neck_t))

    # Лади (як було)
    fret_lines = []
    fret_offsets_in_axis = []
    for polygon, _ in fret_polygons:
        ep = line_endpoints_from_polygon(polygon)
        if ep is None:
            continue
        fret_lines.append(ep)
        cx = (ep[0][0] + ep[1][0]) / 2.0
        cy = (ep[0][1] + ep[1][1]) / 2.0
        fret_offsets_in_axis.append(project_on_axis((cx, cy), neck_center, direction))

    # Лінія nut'а (як було)
    nut_line = None
    nut_offset_in_axis = None
    if nut_polygons:
        nut_polygon, _ = max(nut_polygons, key=lambda x: x[1])
        nut_line = line_endpoints_from_polygon(nut_polygon)
        if nut_line is not None:
            cx = (nut_line[0][0] + nut_line[1][0]) / 2.0
            cy = (nut_line[0][1] + nut_line[1][1]) / 2.0
            nut_offset_in_axis = project_on_axis((cx, cy), neck_center, direction)

    # Якщо ми ще не калібрувались і nut детектовано — використовуємо nut щоб
    # визначити правильний знак вектора direction (первинна орієнтація осі).
    # Після калібрування цей блок не потрібен — знак узгоджується через
    # reference_direction вище.
    if not skeleton.calibrated and nut_offset_in_axis is not None and fret_offsets_in_axis:
        below = sum(1 for s in fret_offsets_in_axis if s < nut_offset_in_axis)
        above = len(fret_offsets_in_axis) - below
        if below > above:
            direction = -direction
            perpendicular = -perpendicular
            nut_offset_in_axis = -nut_offset_in_axis
            fret_offsets_in_axis = [-s for s in fret_offsets_in_axis]
            _dbg(f"  initial axis sign flip via nut: now nut={nut_offset_in_axis:.1f}, "
                 f"frets first 3={[round(s,1) for s in sorted(fret_offsets_in_axis)[:3]]}")

    # ---------- Калібрування / трекінг скелета ----------
    skeleton_offset_nut = None  # s nut в системі поточного кадру (для малювання)

    _dbg(f"nut_detected={nut_offset_in_axis is not None}  "
         f"frets_count={len(fret_offsets_in_axis)}  "
         f"calibrated={skeleton.calibrated}")

    if nut_offset_in_axis is not None and len(fret_offsets_in_axis) >= MIN_FRETS_FOR_SCALE_FIT:
        # nut є — калібруємо/уточнюємо скелет
        relative = sorted([s - nut_offset_in_axis for s in fret_offsets_in_axis])
        _dbg(f"  trying calibration. nut_axis={nut_offset_in_axis:.1f}, "
             f"fret_axis_first5={[round(s,1) for s in sorted(fret_offsets_in_axis)[:5]]}")
        _dbg(f"  relative offsets (fret - nut): {[round(r,1) for r in relative[:8]]}")
        positive = [r for r in relative if r > 0]
        _dbg(f"  positive count: {len(positive)} of {len(relative)}")

        ok = skeleton.update_with_nut(nut_offset_in_axis, fret_offsets_in_axis, direction)
        _dbg(f"  update_with_nut returned: {ok}, "
             f"scale_length={skeleton.scale_length}")

        skeleton_offset_nut = nut_offset_in_axis
    elif skeleton.calibrated and len(fret_offsets_in_axis) >= MIN_FRETS_FOR_SCALE_FIT:
        # nut'а немає, але скелет вже відомий — переносимо
        offset_nut, inliers = fit_offset_with_fixed_scale(
            fret_offsets_in_axis,
            skeleton.scale_length,
            max_fret=MAX_FRET_NUMBER,
            tolerance=SKELETON_FIT_TOLERANCE,
        )
        _dbg(f"  tracking mode: offset={offset_nut}, inliers={len(inliers)}")
        if offset_nut is not None:
            skeleton_offset_nut = offset_nut

    # ---------- Існуюча візуалізація (як було) ----------

    if not fret_lines:
        cv2.putText(frame, "No frets detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return frame

    def line_position(line):
        cx = (line[0][0] + line[1][0]) / 2
        cy = (line[0][1] + line[1][1]) / 2
        return project_on_axis((cx, cy), neck_center, direction)

    all_lines = list(fret_lines)
    if nut_line is not None:
        all_lines.append(nut_line)
    all_lines.sort(key=line_position, reverse=True)

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

    for idx, (p1, p2) in enumerate(fret_lines_sorted, start=1):
        cv2.line(frame, p1, p2, (0, 0, 255), 2)
        cv2.putText(frame, str(idx), (p1[0] + 6, p1[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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

    # ---------- НОВЕ: малювання скелета ----------
    if skeleton.calibrated and skeleton_offset_nut is not None:
        # Малюємо очікувані позиції всіх ладів за скелетом
        # Колір — пурпуровий, тонкі лінії, щоб відрізнялись від детекцій
        skeleton_color = (255, 0, 255)
        for k in range(1, MAX_FRET_NUMBER + 1):
            s_local = skeleton_offset_nut + expected_offset(k, skeleton.scale_length)
            # Перетворюємо локальну s в координати кадру:
            # точка з координатою s_local на осі грифа і t_min/t_max на перпендикулярі.
            # Локальна s рахується від neck_center, тому беремо різницю.
            s_rel = s_local - project_on_axis(neck_center, neck_center, direction)
            # project_on_axis для neck_center дасть 0 (бо ми його віднімаємо самого від себе),
            # тому s_rel == s_local. Залишаю явно для читабельності.
            p1 = world_from_local(s_local, t_min, neck_center, direction, perpendicular)
            p2 = world_from_local(s_local, t_max, neck_center, direction, perpendicular)
            cv2.line(frame, p1, p2, skeleton_color, 1, cv2.LINE_AA)
            # Підпис номера скелета — біля верхньої точки, трохи зміщений
            label_pos = (p1[0] - 18, p1[1] - 18)
            cv2.putText(frame, f"s{k}", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, skeleton_color, 1)

        # Малюємо лінію nut за скелетом — позицію де він "має бути"
        nut_p1 = world_from_local(skeleton_offset_nut, t_min,
                                   neck_center, direction, perpendicular)
        nut_p2 = world_from_local(skeleton_offset_nut, t_max,
                                   neck_center, direction, perpendicular)
        cv2.line(frame, nut_p1, nut_p2, skeleton_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "s0(nut)", (nut_p1[0] - 30, nut_p1[1] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, skeleton_color, 1)

        # Інфо про скелет
        scale_info = f"Skeleton scale: {skeleton.scale_length:.1f}"
        cv2.putText(frame, scale_info, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, skeleton_color, 1)
    else:
        if skeleton.calibrated:
            cv2.putText(frame, "Skeleton: cannot fit to current frame",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        else:
            cv2.putText(frame, "Skeleton: waiting for nut + frets to calibrate",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

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

    skeleton = SkeletonState()

    print("Press 'q' to quit, 's' to save current frame, 'r' to reset skeleton.")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = process_frame(frame, model, skeleton)
        cv2.imshow("Guitar fret matrix", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"frame_{frame_idx:04d}.png"
            cv2.imwrite(fname, out)
            print(f"Saved: {fname}")
        elif key == ord('r'):
            skeleton = SkeletonState()
            print("Skeleton reset")
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()