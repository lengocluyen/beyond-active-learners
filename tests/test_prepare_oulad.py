from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_oulad import RAW_NAMES, WEEKS, initialization_commands, prepare


class OuladInitializationTests(unittest.TestCase):
    def test_fresh_preparation_covers_metadata_labels_static_and_all_landmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            commands = initialization_commands(Path(temporary), "python3")
        self.assertEqual([command[-1] for command in commands[:3]],
                         ["src.build_competency_graph", "src.make_labels", "src.features_static"])
        self.assertEqual(commands[-1][commands[-1].index("--weeks") + 1:], list(map(str, WEEKS)))

    def test_existing_fixed_inputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename in ("competencies.csv", "labels.csv", "static_features.csv",
                             *(f"traversal_week{week}.csv" for week in WEEKS if week != 8)):
                (root / filename).write_text("retained\n", encoding="utf-8")
            commands = initialization_commands(root, "python3")
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][commands[0].index("--weeks") + 1:], ["8"])

    def test_missing_raw_inputs_stop_before_any_subprocess(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("src.paths.get_data_path", side_effect=lambda path: Path(temporary) / path), \
                 patch("scripts.prepare_oulad.subprocess.run") as run:
                with self.assertRaisesRegex(FileNotFoundError, "no builders were run"):
                    prepare()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
