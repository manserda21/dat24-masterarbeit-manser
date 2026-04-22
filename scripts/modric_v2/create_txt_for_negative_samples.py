from pathlib import Path

NEGATIVE_DIR = Path("/data/manser/datasets/modric_v2/intermediate/negative_samples")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

def main():
    n_created = 0
    n_skipped = 0

    for image_path in NEGATIVE_DIR.iterdir():
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        label_path = NEGATIVE_DIR / f"{image_path.stem}.txt"

        if label_path.exists():
            n_skipped += 1
            continue

        label_path.write_text("")
        n_created += 1

    print(f"Leere Labels erstellt: {n_created}")
    print(f"Bereits vorhanden: {n_skipped}")


if __name__ == "__main__":
    main()