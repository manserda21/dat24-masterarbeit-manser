from collections import Counter
from pathlib import Path


class DuplicateAnalyzer:

    def __init__(
        self,
        labels_dir: str,
        class_names: dict[int, str],
    ):
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names

    def analyze(self) -> list[dict]:

        results = []

        for label_file in sorted(
            self.labels_dir.glob("*.txt")
        ):

            counts = self._count_classes(label_file)

            duplicates = {
                self.class_names.get(cls_id, str(cls_id)): count
                for cls_id, count in counts.items()
                if cls_id != 0 and count > 1
            }

            if duplicates:
                results.append(
                    {
                        "frame": label_file.name,
                        "duplicates": duplicates,
                    }
                )

        return results

    def _count_classes(
        self,
        label_file: Path,
    ) -> Counter:

        class_ids = []

        with open(label_file, "r") as f:

            for line in f:

                parts = line.strip().split()

                if not parts:
                    continue

                class_ids.append(
                    int(float(parts[0]))
                )

        return Counter(class_ids)