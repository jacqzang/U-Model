"""
Validates the structure of an uploaded dataset zip against U-Model's rules.
Handles both zip layouts:
  cats/dog1.jpg              (class folders at zip root)
  test_data/cats/dog1.jpg    (class folders nested one level in, e.g. from
                               zipping a folder that itself contains class folders)
"""
import zipfile
from collections import defaultdict
from pathlib import Path

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_IMAGES_PER_CLASS = 50
MIN_CLASSES = 2
MAX_CLASSES = 10
IMBALANCE_RATIO = 3


def _real_image_paths(zf: zipfile.ZipFile) -> list[Path]:
    """Filters out directory entries and macOS junk files."""
    paths = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        path = Path(name)
        if path.name.startswith("._") or path.name == ".DS_Store":
            continue
        paths.append(path)
    return paths


def _strip_common_wrapper(paths: list[Path]) -> list[Path]:
    """
    If every file shares the same single top-level folder (e.g. everything
    starts with 'test_data/'), strip that wrapper so class folders are at
    the front. Only strips one level, and only if ALL files share it.
    """
    if not paths:
        return paths

    first_parts = {p.parts[0] for p in paths if len(p.parts) > 1}
    if len(first_parts) == 1:
        wrapper = next(iter(first_parts))
        stripped = [Path(*p.parts[1:]) for p in paths if p.parts[0] == wrapper]
        return stripped
    return paths


def validate_dataset_zip(zip_path: str) -> dict:
    errors = []
    warnings = []
    class_counts: dict[str, int] = defaultdict(int)
    invalid_files = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        paths = _real_image_paths(zf)
        paths = _strip_common_wrapper(paths)

        for path in paths:
            if len(path.parts) < 2:
                continue  # file sitting at root, not inside a class folder

            class_name = path.parts[0]
            extension = path.suffix.lower()

            if extension not in ALLOWED_EXTENSIONS:
                invalid_files.append(str(path))
                continue

            class_counts[class_name] += 1

    if invalid_files:
        errors.append(
            f"Unsupported file type found: {invalid_files[0]}"
            + (f" (and {len(invalid_files) - 1} more)" if len(invalid_files) > 1 else "")
            + " — please use JPG or PNG only."
        )

    num_classes = len(class_counts)
    if num_classes < MIN_CLASSES:
        errors.append(f"Found only {num_classes} categor{'y' if num_classes == 1 else 'ies'} — you need at least 2.")
    elif num_classes > MAX_CLASSES:
        errors.append(f"Found {num_classes} categories — the maximum is 10.")

    for class_name, count in class_counts.items():
        if count < MIN_IMAGES_PER_CLASS:
            warnings.append(
                f"Only {count} images in \"{class_name}\" — we recommend at least "
                f"{MIN_IMAGES_PER_CLASS} per category for reliable results."
            )

    if class_counts:
        largest = max(class_counts.values())
        smallest = min(class_counts.values())
        if smallest > 0 and largest > smallest * IMBALANCE_RATIO:
            warnings.append(
                "Classes are imbalanced — model may be biased toward the larger class."
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "classes": dict(class_counts),
        "image_paths": [p for p in paths if p.suffix.lower() in ALLOWED_EXTENSIONS and len(p.parts) >= 2],
    }