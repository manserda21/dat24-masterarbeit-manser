from pathlib import Path
import shutil

# --------------------------------------------------
# DATENSATZ
# --------------------------------------------------

DATASET_DIR = Path("/data/manser/datasets/calibration/malaga_home_stadium_v1")

IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

EXCLUDED_DIR = DATASET_DIR / "excluded"

# --------------------------------------------------
# BILDER
# --------------------------------------------------

EXCLUDED_IMAGES = [

    # --------------------------------------------------
    # Hinter dem Tor
    # --------------------------------------------------
    "malaga_barca_2017_03056",
    "malaga_barca_2017_03064",
    "malaga_barca_2017_03071",
    "malaga_barca_2017_03080",
    "malaga_barca_2017_03084",
    "malaga_barca_2017_03094",
    "malaga_barca_2017_03102",
    "malaga_barca_2017_03110",
    "malaga_barca_2017_03137",
    "malaga_real_2016_02663",
    "malaga_real_2016_02666",
    "malaga_real_2016_02687",
    "malaga_real_2016_02690",
    "malaga_real_2016_02695",
    "malaga_real_2016_02716",
    "malaga_real_2016_02722",
    "malaga_real_2016_02727",
    "malaga_real_2016_02729",
    "malaga_real_2017_16409",
    "malaga_real_2017_16421",
    "malaga_real_2017_16431",
    "malaga_real_2017_16432",
    "malaga_real_2017_16434",
    "malaga_real_2017_16439",
    "malaga_real_2017_16446",
    "malaga_real_2017_16456",

    # --------------------------------------------------
    # Zoom
    # --------------------------------------------------
    "malaga_barca_2017_03057",
    "malaga_barca_2017_03072",
    "malaga_barca_2017_03074",
    "malaga_barca_2017_03075",
    "malaga_barca_2017_03082",
    "malaga_barca_2017_03090",
    "malaga_barca_2017_03099",
    "malaga_barca_2017_03103",
    "malaga_barca_2017_03107",
    "malaga_barca_2017_03113",
    "malaga_barca_2017_03124",
    "malaga_barca_2017_03126",
    "malaga_barca_2017_03128",
    "malaga_barca_2017_03134",
    "malaga_real_2016_02674",
    "malaga_real_2016_02680",
    "malaga_real_2016_02684",
    "malaga_real_2016_02700",
    "malaga_real_2016_02702",
    "malaga_real_2016_02703",
    "malaga_real_2016_02728",
    "malaga_real_2017_16444",

    # --------------------------------------------------
    # Andere Perspektive
    # --------------------------------------------------
    "malaga_barca_2017_03059",
    "malaga_barca_2017_03070",
    "malaga_barca_2017_03073",
    "malaga_barca_2017_03077",
    "malaga_barca_2017_03096",
    "malaga_barca_2017_03123",
    "malaga_real_2016_02731",
    "malaga_real_2017_16410",
    "malaga_real_2017_16411",
    "malaga_real_2017_16442",
    "malaga_real_2017_16452",
    "malaga_real_2017_16454",

    # --------------------------------------------------
    # Zu wenige Linien
    # --------------------------------------------------
    "malaga_real_2016_02705",
    "malaga_real_2016_02733",
    "malaga_real_2017_16423",
]


def move_file(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"Moved: {src.name}")


def main():

    excluded_images_dir = EXCLUDED_DIR / "images"
    excluded_annotations_dir = EXCLUDED_DIR / "annotations"

    moved = 0

    for stem in EXCLUDED_IMAGES:

        image = IMAGES_DIR / f"{stem}.jpg"
        annotation = ANNOTATIONS_DIR / f"{stem}.json"

        if image.exists():
            move_file(
                image,
                excluded_images_dir / image.name,
            )

        if annotation.exists():
            move_file(
                annotation,
                excluded_annotations_dir / annotation.name,
            )

        moved += 1

    print()
    print(f"Finished.")
    print(f"Excluded frames: {moved}")


if __name__ == "__main__":
    main()