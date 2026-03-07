"""
Test suite for the B/W/Green region detection feature.

Runs the missing-puzzle-piece pipeline for each of the three region_color
modes (white, black, green) using the testcases/ directory, then compares
each output against testcases/correct.jpg by aligning the images and
computing a structural similarity score.

Usage:
    python test_bwgreen.py
"""

import os
import sys
import tempfile

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from summon_missing_puzzle_piece import ImageAligner

TESTCASES_DIR = os.path.join(os.path.dirname(__file__), "testcases")
COMPLETE_IMAGE = os.path.join(TESTCASES_DIR, "complete.jpg")
CORRECT_IMAGE  = os.path.join(TESTCASES_DIR, "correct.jpg")

MISSING_IMAGES = {
    "white": os.path.join(TESTCASES_DIR, "missing_white.jpg"),
    "black": os.path.join(TESTCASES_DIR, "missing_black.jpg"),
    "green": os.path.join(TESTCASES_DIR, "missing_green.jpg"),
}


# ---------------------------------------------------------------------------
# Alignment-based similarity
# ---------------------------------------------------------------------------

def _load_as_bgr_white_bg(path: str) -> np.ndarray:
    """Load an image and composite any transparent areas onto a white background."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img is not None, f"Could not load image: {path}"
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(float) / 255.0
        bgr   = img[:, :, :3].astype(float)
        white = np.full_like(bgr, 255.0)
        return (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
    elif img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def align_and_similarity(output_path: str, correct_bgr: np.ndarray) -> float:
    """
    Align *output_path* (PNG, possibly with transparency) to *correct_bgr* using
    SIFT + affine transform and return a normalised cross-correlation score in
    [0, 1].  Transparent pixels in the output are composited onto a white
    background before comparison.

    Falls back to direct resize-comparison when feature matching fails (e.g.
    for very small images).
    """
    output_bgr = _load_as_bgr_white_bg(output_path)
    sift = cv2.SIFT_create()
    bfm  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    gray_out  = cv2.cvtColor(output_bgr,  cv2.COLOR_BGR2GRAY)
    gray_corr = cv2.cvtColor(correct_bgr, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(gray_out,  None)
    kp2, des2 = sift.detectAndCompute(gray_corr, None)

    aligned = None
    if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
        matches = bfm.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
            if M is not None:
                h, w = correct_bgr.shape[:2]
                aligned = cv2.warpAffine(output_bgr, M, (w, h))

    # Fall back: just resize output to match correct dimensions
    if aligned is None:
        h, w = correct_bgr.shape[:2]
        aligned = cv2.resize(output_bgr, (w, h), interpolation=cv2.INTER_AREA)

    # Normalised cross-correlation as similarity score
    a = aligned.astype(np.float32).ravel()
    b = correct_bgr.astype(np.float32).ravel()
    norm_a = np.linalg.norm(a - a.mean())
    norm_b = np.linalg.norm(b - b.mean())
    if norm_a == 0 or norm_b == 0:
        return 0.0
    score = float(np.dot(a - a.mean(), b - b.mean()) / (norm_a * norm_b))
    return max(0.0, score)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def run_testcase(region_color: str, tmpdir: str) -> str:
    """Run the aligner for *region_color* and return the output image path."""
    out_path = os.path.join(tmpdir, f"replacement_{region_color}.png")
    aligner = ImageAligner(debug=False)
    aligner.process(
        COMPLETE_IMAGE,
        MISSING_IMAGES[region_color],
        out_path,
        puzzle_width_inches=24,
        region_color=region_color,
    )
    assert os.path.exists(out_path), f"Output image not created at {out_path}"
    return out_path


def test_region_color_white(tmpdir: str):
    out_path = run_testcase("white", tmpdir)
    correct  = cv2.imread(CORRECT_IMAGE)
    score    = align_and_similarity(out_path, correct)
    print(f"  [white]  similarity score: {score:.4f}")
    assert score > 0.3, f"White-region output too dissimilar to correct.jpg (score={score:.4f})"


def test_region_color_black(tmpdir: str):
    out_path = run_testcase("black", tmpdir)
    correct  = cv2.imread(CORRECT_IMAGE)
    score    = align_and_similarity(out_path, correct)
    print(f"  [black]  similarity score: {score:.4f}")
    assert score > 0.3, f"Black-region output too dissimilar to correct.jpg (score={score:.4f})"


def test_region_color_green(tmpdir: str):
    out_path = run_testcase("green", tmpdir)
    correct  = cv2.imread(CORRECT_IMAGE)
    score    = align_and_similarity(out_path, correct)
    print(f"  [green]  similarity score: {score:.4f}")
    assert score > 0.3, f"Green-region output too dissimilar to correct.jpg (score={score:.4f})"


def test_invalid_region_color():
    """_find_region should raise ValueError for unknown color names."""
    aligner = ImageAligner()
    dummy   = np.zeros((100, 100, 4), dtype=np.uint8)
    try:
        aligner._find_region(dummy, region_color="purple")
        raise AssertionError("Expected ValueError was not raised")
    except ValueError:
        pass
    print("  [invalid] ValueError raised correctly for unknown region_color")


def test_find_region_white_detects_white():
    """_find_region('white') should detect white pixels and return a non-empty mask."""
    aligner = ImageAligner()
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    # Draw a white square
    img[20:80, 20:80] = [255, 255, 255, 255]
    mask = aligner._find_region(img, region_color='white')
    assert mask.max() == 255, "White region mask is empty"
    assert mask[50, 50] == 255, "Center of white square not in mask"
    print("  [unit]   _find_region('white') detects white region")


def test_find_region_black_detects_black():
    """_find_region('black') should detect black pixels and return a non-empty mask."""
    aligner = ImageAligner()
    # White background with a black square
    img = np.full((100, 100, 4), 255, dtype=np.uint8)
    img[20:80, 20:80] = [0, 0, 0, 255]
    mask = aligner._find_region(img, region_color='black')
    assert mask.max() == 255, "Black region mask is empty"
    assert mask[50, 50] == 255, "Center of black square not in mask"
    print("  [unit]   _find_region('black') detects black region")


def test_find_region_green_detects_green():
    """_find_region('green') should detect #67c885 pixels."""
    aligner = ImageAligner()
    # White background with a #67c885 green square (BGR: 133, 200, 103)
    img = np.full((100, 100, 4), 255, dtype=np.uint8)
    img[20:80, 20:80] = [133, 200, 103, 255]
    mask = aligner._find_region(img, region_color='green')
    assert mask.max() == 255, "Green region mask is empty"
    assert mask[50, 50] == 255, "Center of green square not in mask"
    print("  [unit]   _find_region('green') detects #67c885 green region")


def test_find_region_green_shadow_range():
    """_find_region('green') should detect shadow variants of the green (#51613d and #66cc83)."""
    aligner = ImageAligner()
    # BGR values derived from hex: #RRGGBB → (B, G, R)
    # #51613d → R=0x51=81, G=0x61=97, B=0x3d=61  → BGR=(61, 97, 81)
    # #66cc83 → R=0x66=102, G=0xcc=204, B=0x83=131 → BGR=(131, 204, 102)
    for hex_color, bgr in [("#51613d", (61, 97, 81)), ("#66cc83", (131, 204, 102))]:
        img = np.full((100, 100, 4), 255, dtype=np.uint8)
        img[20:80, 20:80] = (*bgr, 255)
        mask = aligner._find_region(img, region_color='green')
        assert mask.max() == 255, f"Shadow green {hex_color} not detected"
        assert mask[50, 50] == 255, f"Center of {hex_color} square not in mask"
    print("  [unit]   _find_region('green') detects shadow variants #51613d and #66cc83")


def test_backward_compat_find_white_region():
    """_find_white_region should still work (backward compatibility)."""
    aligner = ImageAligner()
    img = np.zeros((100, 100, 4), dtype=np.uint8)
    img[20:80, 20:80] = [255, 255, 255, 255]
    mask = aligner._find_white_region(img)
    assert mask.max() == 255, "_find_white_region backward-compat method returns empty mask"
    print("  [compat] _find_white_region still works via backward-compat wrapper")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    # Check prerequisites
    for path in [COMPLETE_IMAGE, CORRECT_IMAGE, *MISSING_IMAGES.values()]:
        if not os.path.exists(path):
            print(f"ERROR: Required testcase file missing: {path}")
            sys.exit(1)

    failures = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tests = [
            ("white region (integration)",       lambda: test_region_color_white(tmpdir)),
            ("black region (integration)",       lambda: test_region_color_black(tmpdir)),
            ("green region (integration)",       lambda: test_region_color_green(tmpdir)),
            ("invalid region_color",             test_invalid_region_color),
            ("unit: white detection",            test_find_region_white_detects_white),
            ("unit: black detection",            test_find_region_black_detects_black),
            ("unit: green detection",            test_find_region_green_detects_green),
            ("unit: green shadow range",         test_find_region_green_shadow_range),
            ("backward compat _find_white_region", test_backward_compat_find_white_region),
        ]

        print("Running tests...\n")
        for name, fn in tests:
            print(f"Test: {name}")
            try:
                fn()
                print(f"  PASSED\n")
            except Exception as exc:
                print(f"  FAILED: {exc}\n")
                failures.append((name, exc))

    if failures:
        print(f"{len(failures)} test(s) FAILED:")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
