"""
Puzzle piece shape classifier using contour shape features and a Random Forest model.

Trained on synthetic puzzle piece shapes to recognize puzzle-piece-like contours.
This classifier is used to bias contour selection in ImageAligner toward
shapes that resemble actual puzzle pieces (roughly rectangular with
convex tabs and concave blanks on edges).
"""

import numpy as np
import cv2
import os
import pickle
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'puzzle_piece_model.pkl')


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_shape_features(contour):
    """Extract shape features from a contour for puzzle-piece classification.

    Features captured:
    - Aspect ratio and its inverse-normalised form
    - Extent  (area / bounding-rect area)
    - Solidity (area / convex-hull area)
    - Circularity (4π·area / perimeter²)
    - Convexity-defect statistics (count, significant count, mean/max depth)
    - 7 log-scaled Hu moments

    Returns a 1-D float32 numpy array, or None if the contour is degenerate.
    """
    area = cv2.contourArea(contour)
    if area < 1:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1:
        return None

    # Bounding rectangle
    _, _, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 1.0
    aspect_normalised = min(aspect_ratio, 1.0 / aspect_ratio) if aspect_ratio > 0 else 0.0
    extent = area / (w * h) if w * h > 0 else 0.0

    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Equivalent diameter (for normalising defect depths)
    equiv_diameter = np.sqrt(4 * area / np.pi)

    # Convexity defects
    num_defects = 0
    significant_defects = 0
    mean_defect_depth_norm = 0.0
    max_defect_depth_norm = 0.0
    try:
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        if hull_idx is not None and len(hull_idx) >= 3 and len(contour) >= 4:
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is not None:
                depths = defects[:, 0, 3] / 256.0  # fixed-point → pixels
                num_defects = len(depths)
                threshold = 0.05 * equiv_diameter
                significant_defects = int(np.sum(depths > threshold))
                mean_defect_depth_norm = float(np.mean(depths)) / equiv_diameter if equiv_diameter > 0 else 0.0
                max_defect_depth_norm = float(np.max(depths)) / equiv_diameter if equiv_diameter > 0 else 0.0
    except cv2.error:
        pass

    # Hu moments (log-scaled for better numerical range)
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    features = np.array([
        aspect_ratio,
        aspect_normalised,
        extent,
        solidity,
        circularity,
        float(num_defects),
        float(significant_defects),
        mean_defect_depth_norm,
        max_defect_depth_norm,
    ] + list(hu_log), dtype=np.float32)

    return features


# ---------------------------------------------------------------------------
# Synthetic shape generators
# ---------------------------------------------------------------------------

def _generate_puzzle_piece_contour(size=200, rng=None):
    """Generate a single synthetic puzzle-piece contour.

    A puzzle piece is a rectangle with circular tabs (protrusions) and blanks
    (indentations) on each of its four edges.  Each edge is independently
    assigned flat (0), tab-out (+1), or blank-in (-1) at random.

    Returns an OpenCV contour array (N, 1, 2) or None on failure.
    """
    if rng is None:
        rng = np.random.RandomState()

    canvas = np.zeros((size * 2, size * 2), dtype=np.uint8)
    cx, cy = size, size

    # Random base rectangle (roughly square)
    w = int(size * rng.uniform(0.55, 0.85))
    h = int(size * rng.uniform(0.55, 0.85))
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2
    cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, -1)

    # Tab/blank radius ≈ 12–20 % of the perpendicular edge dimension
    r_h = int(h * rng.uniform(0.10, 0.20))  # radius for top/bottom tabs
    r_w = int(w * rng.uniform(0.10, 0.20))  # radius for left/right tabs

    edge_types = rng.choice([-1, 0, 1], size=4)

    # Top edge
    tx = cx + rng.randint(-w // 8, w // 8 + 1)
    if edge_types[0] == 1:
        cv2.circle(canvas, (tx, y1 - r_h), r_h, 255, -1)
    elif edge_types[0] == -1:
        cv2.circle(canvas, (tx, y1), r_h, 0, -1)

    # Bottom edge
    tx = cx + rng.randint(-w // 8, w // 8 + 1)
    if edge_types[1] == 1:
        cv2.circle(canvas, (tx, y2 + r_h), r_h, 255, -1)
    elif edge_types[1] == -1:
        cv2.circle(canvas, (tx, y2), r_h, 0, -1)

    # Left edge
    ty = cy + rng.randint(-h // 8, h // 8 + 1)
    if edge_types[2] == 1:
        cv2.circle(canvas, (x1 - r_w, ty), r_w, 255, -1)
    elif edge_types[2] == -1:
        cv2.circle(canvas, (x1, ty), r_w, 0, -1)

    # Right edge
    ty = cy + rng.randint(-h // 8, h // 8 + 1)
    if edge_types[3] == 1:
        cv2.circle(canvas, (x2 + r_w, ty), r_w, 255, -1)
    elif edge_types[3] == -1:
        cv2.circle(canvas, (x2, ty), r_w, 0, -1)

    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _generate_non_puzzle_contour(size=200, rng=None):
    """Generate a contour of a shape that is NOT a puzzle piece.

    Shapes: pure rectangle, ellipse/circle, triangle, L-shape, irregular polygon.

    Returns an OpenCV contour array (N, 1, 2) or None on failure.
    """
    if rng is None:
        rng = np.random.RandomState()

    canvas = np.zeros((size * 2, size * 2), dtype=np.uint8)
    cx, cy = size, size
    shape_type = rng.randint(0, 5)

    if shape_type == 0:
        # Pure rectangle
        w = int(size * rng.uniform(0.45, 0.90))
        h = int(size * rng.uniform(0.45, 0.90))
        cv2.rectangle(canvas, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), 255, -1)

    elif shape_type == 1:
        # Ellipse / circle
        a = int(size * rng.uniform(0.25, 0.50))
        b = int(size * rng.uniform(0.25, 0.50))
        angle = rng.randint(0, 360)
        cv2.ellipse(canvas, (cx, cy), (a, b), angle, 0, 360, 255, -1)

    elif shape_type == 2:
        # Triangle
        pts = []
        for base_angle in [90, 210, 330]:
            a = base_angle + rng.randint(-25, 25)
            r = int(size * rng.uniform(0.28, 0.50))
            pts.append([
                int(cx + r * np.cos(np.radians(a))),
                int(cy + r * np.sin(np.radians(a)))
            ])
        cv2.fillPoly(canvas, [np.array(pts, dtype=np.int32)], 255)

    elif shape_type == 3:
        # L-shape (two overlapping rectangles)
        w = int(size * rng.uniform(0.50, 0.80))
        h = int(size * rng.uniform(0.50, 0.80))
        x1, y1 = cx - w // 2, cy - h // 2
        cv2.rectangle(canvas, (x1, y1), (x1 + w, y1 + h), 255, -1)
        cw, ch = w // 2, h // 2
        cv2.rectangle(canvas, (x1 + cw, y1 + ch), (x1 + w, y1 + h), 0, -1)

    else:
        # Irregular convex polygon
        n_pts = rng.randint(5, 9)
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_pts))
        radii = rng.uniform(0.28, 0.50, n_pts) * size
        pts = np.array([
            [int(cx + r * np.cos(a)), int(cy + r * np.sin(a))]
            for a, r in zip(angles, radii)
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], 255)

    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_training_data(n_samples=2000, seed=42):
    """Generate balanced synthetic training data.

    Returns:
        X (ndarray, shape [n, n_features]): feature matrix.
        y (ndarray, shape [n]):             labels (1=puzzle, 0=non-puzzle).
    """
    rng = np.random.RandomState(seed)
    X, y = [], []

    n_puzzle = n_samples // 2
    n_non_puzzle = n_samples - n_puzzle

    for _ in range(n_puzzle):
        c = _generate_puzzle_piece_contour(size=200, rng=rng)
        if c is not None:
            f = extract_shape_features(c)
            if f is not None:
                X.append(f)
                y.append(1)

    for _ in range(n_non_puzzle):
        c = _generate_non_puzzle_contour(size=200, rng=rng)
        if c is not None:
            f = extract_shape_features(c)
            if f is not None:
                X.append(f)
                y.append(0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(n_samples=2000, seed=42, model_path=MODEL_PATH, verbose=True):
    """Train a Random Forest classifier on synthetic puzzle-piece data.

    Performs a 70 / 15 / 15 train / validation / test split and prints
    accuracy and a full classification report for both val and test sets.

    Args:
        n_samples:  Total number of synthetic samples (balanced classes).
        seed:       Random seed for reproducibility.
        model_path: Where to save the trained model (pickle).
        verbose:    Print progress and metrics.

    Returns:
        Trained ``RandomForestClassifier``.
    """
    if verbose:
        print(f"Generating {n_samples} synthetic training samples …")

    X, y = generate_training_data(n_samples=n_samples, seed=seed)

    if verbose:
        print(f"Generated {len(X)} valid samples "
              f"({int(np.sum(y == 1))} puzzle, {int(np.sum(y == 0))} non-puzzle)")

    # 70 / 15 / 15 split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)

    if verbose:
        print(f"Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Validation metrics
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    if verbose:
        print(f"\nValidation accuracy: {val_acc:.4f}")
        print(classification_report(y_val, y_val_pred,
                                    target_names=['non-puzzle', 'puzzle']))

    # Test metrics
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    if verbose:
        print(f"Test accuracy: {test_acc:.4f}")
        print(classification_report(y_test, y_test_pred,
                                    target_names=['non-puzzle', 'puzzle']))

    # Persist
    with open(model_path, 'wb') as fh:
        pickle.dump(clf, fh)
    if verbose:
        print(f"\nModel saved to {model_path}")

    return clf


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_classifier(model_path=MODEL_PATH):
    """Load a previously saved classifier from *model_path*.

    Returns ``None`` if the file does not exist.
    """
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as fh:
        return pickle.load(fh)


def score_contour(contour, classifier):
    """Return the puzzle-piece probability of *contour* in [0, 1].

    Returns 0.5 (neutral) when *classifier* is ``None`` or features cannot
    be extracted, so the caller's area-based selection is unaffected.
    """
    if classifier is None:
        return 0.5
    features = extract_shape_features(contour)
    if features is None:
        return 0.5
    try:
        prob = classifier.predict_proba(features.reshape(1, -1))[0, 1]
        return float(prob)
    except Exception:
        return 0.5
