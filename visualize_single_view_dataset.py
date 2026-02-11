import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import os

def load_yolo_labels(
    label_file: Path, img_width: int, img_height: int
) -> List[Tuple[int, float, float, float, float]]:
    """
    Load YOLO-format labels from a .txt file.

    Each line: <class_id> <cx> <cy> <w> <h> (all normalized to [0, 1]).
    Returns list of (class_id, x_min, y_min, x_max, y_max) in pixel coords.
    """
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not label_file.is_file():
        return boxes

    with label_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Support standard YOLO format and lines with extra columns
            # (e.g. predictions that include confidence, etc.).
            if len(parts) < 5:
                # Ignore malformed/too-short lines
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])

            # Convert from normalized center format to pixel corner format
            cx_px = cx * img_width
            cy_px = cy * img_height
            w_px = w * img_width
            h_px = h * img_height

            x_min = cx_px - w_px / 2
            y_min = cy_px - h_px / 2
            x_max = cx_px + w_px / 2
            y_max = cy_px + h_px / 2

            boxes.append((cls_id, x_min, y_min, x_max, y_max))

    return boxes


def find_images(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    """
    Recursively find all images under root with given extensions.
    """
    images: List[Path] = []
    for ext in exts:
        images.extend(root.rglob(f"*{ext}"))
    return sorted(images)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO labels on images from the Single-View-Detection dataset."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="MVBroTrack",
        help="Path to the MVBroTrack root folder (relative to the current working directory or absolute).",
    )

    args = parser.parse_args()

    # Convert provided path to an absolute path relative to the current working directory
    dataset_root_path = Path(args.dataset_root).resolve()
    if not dataset_root_path.is_dir():
        # raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
        print("Dataset directory does not exist.")
        print(f"Checked absolute path: {dataset_root_path}")
        raise SystemExit(1)

    # Build the Single-View-Detection directory as an absolute Path
    dataset_dir = dataset_root_path / "Single-View-Detection"

    # Fixed list of image extensions
    exts = (".jpg", ".png", ".jpeg")
    images = find_images(dataset_dir, exts)

    if not images:
        raise SystemExit(f"No images found in {dataset_dir} with extensions {exts}")

    print(f"Found {len(images)} images in {dataset_dir}")
    print("Press any key (except 'q'/ESC) to go to the next image, or 'q' / ESC to quit.")

    total = len(images)
    idx = 0
    while True:
        img_path = images[idx]
        # label_path = img_path.with_suffix(".txt")
        # Assume labels are stored in a sibling "labels" folder next to "images"
        labels_root = img_path.parent.parent / "labels"
        label_path = labels_root / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read image: {img_path}, skipping.")
            idx = (idx + 1) % total
            continue

        h, w = img.shape[:2]
        boxes = load_yolo_labels(label_path, w, h)

        # Draw bounding boxes on a copy of the image
        vis_img = img.copy()
        if not boxes:
            print(f"No labels found or parsed for: {label_path}")
        for cls_id, x_min, y_min, x_max, y_max in boxes:
            pt1 = (int(x_min), int(y_min))
            pt2 = (int(x_max), int(y_max))
            cv2.rectangle(vis_img, pt1, pt2, (0, 255, 0), 2)

        # Resize after drawing labels so boxes and text scale with the image
        vis_img = cv2.resize(vis_img, (1920, 1080))

        # Overlay info text: filename, index/total, number of boxes
        info_text = f"{img_path.name} | {idx+1}/{total} | boxes: {len(boxes)}"
        cv2.putText(
            vis_img,
            info_text,
            (10, vis_img.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


        cv2.imshow("YOLO Single-View Detection", vis_img)
        # print(f"[{idx+1}/{total}] {img_path.name} - labels from {label_path.name} - boxes: {len(boxes)}")

        key = cv2.waitKey(0) & 0xFF
        # 'q' or ESC: quit, otherwise advance and wrap around
        if key in (ord("q"), 27):
            break
        idx = (idx + 1) % total

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()