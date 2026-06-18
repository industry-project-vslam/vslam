from __future__ import annotations

import ast
import inspect
from pathlib import Path
import unittest

from gui.main_window import MainWindow


ROOT = Path(__file__).resolve().parents[2]


def _button_commands() -> set[str]:
    panel_path = ROOT / "gui" / "widgets" / "mission_control_panel.py"
    tree = ast.parse(panel_path.read_text(encoding="utf-8"))
    commands: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "labels" for target in node.targets):
            for item in node.value.elts:
                if isinstance(item, ast.Tuple) and len(item.elts) == 2:
                    commands.add(ast.literal_eval(item.elts[0]))
    return commands


def _worker_handlers() -> set[str]:
    worker_path = ROOT / "gui" / "worker.py"
    tree = ast.parse(worker_path.read_text(encoding="utf-8"))
    handlers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "name":
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    handlers.add(comparator.value)
    return handlers


class ButtonWiringTest(unittest.TestCase):
    def test_all_buttons_have_worker_handlers(self) -> None:
        self.assertEqual(set(), _button_commands() - _worker_handlers())

    def test_emergency_buttons_are_direct_not_queued(self) -> None:
        source = inspect.getsource(MainWindow.send_command)
        self.assertIn('command == "emergency_stop"', source)
        self.assertIn("emergency_stop_direct", source)
        self.assertIn('command == "safe_hover_land"', source)
        self.assertIn("safe_hover_land_direct", source)
        self.assertIn('command == "hard_motor_kill"', source)
        self.assertIn("hard_motor_kill_direct", source)


if __name__ == "__main__":
    unittest.main()
