import argparse

from .duplicate_analyzer import DuplicateAnalyzer


CLASS_NAMES = {
    0: "opponent",
    1: "danilo",
    2: "varane",
    3: "ramos",
    4: "marcelo",
    5: "modric",
    6: "casemiro",
    7: "kroos",
    8: "isco",
    9: "benzema",
    10: "ronaldo",
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels",
        required=True,
        type=str,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    analyzer = DuplicateAnalyzer(
        labels_dir=args.labels,
        class_names=CLASS_NAMES,
    )

    results = analyzer.analyze()

    print("\n=== Duplicate Analysis ===\n")

    for result in results:

        duplicate_text = ", ".join(
            f"{name} x{count}"
            for name, count in result["duplicates"].items()
        )

        print(
            f"{result['frame']}: {duplicate_text}"
        )


if __name__ == "__main__":
    main()