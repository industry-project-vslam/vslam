from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_ROOT = Path("stream_out/fixed_formation_missions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a fixed-formation swarm mission log folder.")
    parser.add_argument("run_dir", nargs="?", help="Mission log folder. Omit with --latest.")
    parser.add_argument("--latest", action="store_true", help="Analyze the newest mission folder under stream_out/fixed_formation_missions.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Root folder that contains mission log folders.")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.latest, Path(args.root))
    report = analyze_run(run_dir)
    print(report)


def analyze_run(run_dir: Path) -> str:
    lines: list[str] = []
    lines.append(f"Log folder: {run_dir}")
    summary = _read_json(run_dir / "run_summary.json")
    if summary:
        lines.append(
            "Final: "
            f"mode={summary.get('mode', 'unknown')} "
            f"state={summary.get('state', 'unknown')} "
            f"emergency={summary.get('emergency', 'unknown')} "
            f"battery={summary.get('battery_summary', 'unknown')} "
            f"heading={summary.get('heading_deg', 'unknown')}"
        )

    issues: list[str] = []
    issues.extend(_emergency_issues(run_dir))
    issues.extend(_decision_issues(run_dir))
    issues.extend(_command_issues(run_dir))
    issues.extend(_reslot_issues(run_dir))
    issues.extend(_ranger_issues(run_dir))

    if issues:
        lines.append("")
        lines.append("Likely issues:")
        lines.extend(f"- {issue}" for issue in _dedupe(issues))
    else:
        lines.append("")
        lines.append("Likely issues: none obvious in logs.")

    lines.append("")
    lines.append("Last events:")
    for row in _tail_csv(run_dir / "event_log.csv", 12):
        lines.append(f"- [{row.get('elapsed_s', '?')}s] {row.get('state', '?')}: {row.get('event', '')}")

    lines.append("")
    lines.append("Last decisions:")
    for row in _tail_csv(run_dir / "decision_log.csv", 12):
        lines.append(
            "- "
            f"[{row.get('elapsed_s', '?')}s] "
            f"{row.get('decision', '')} -> {row.get('action', '')}: {row.get('reason', '')}"
        )
    return "\n".join(lines)


def _resolve_run_dir(raw: str | None, latest: bool, root: Path) -> Path:
    if raw:
        run_dir = Path(raw)
    elif latest:
        candidates = [path for path in root.glob("*") if path.is_dir()]
        if not candidates:
            raise SystemExit(f"No mission folders found in {root}")
        run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    else:
        raise SystemExit("Pass a run folder or use --latest")
    if not run_dir.exists():
        raise SystemExit(f"Log folder does not exist: {run_dir}")
    return run_dir


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _tail_csv(path: Path, count: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-count:]


def _all_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _emergency_issues(run_dir: Path) -> list[str]:
    issues: list[str] = []
    for row in _tail_csv(run_dir / "emergency_log.csv", 5):
        issues.append(
            f"emergency action={row.get('action', '')} targets={row.get('target_count', '')} "
            f"detail={row.get('detail', '')}"
        )
    return issues


def _decision_issues(run_dir: Path) -> list[str]:
    issues: list[str] = []
    watched_actions = {"NO_MOTION", "SAFE_HOVER", "SAFE_HOVER_LAND", "BLOCK_TURN", "EMERGENCY_HOVER", "KILL_ALL"}
    for row in _all_csv(run_dir / "decision_log.csv"):
        action = row.get("action", "")
        decision = row.get("decision", "")
        if action in watched_actions or "BLOCKED" in decision or "FAILED" in decision or "CRITICAL" in decision:
            issues.append(f"decision {decision} -> {action}: {row.get('reason', '')}")
    return issues[-20:]


def _command_issues(run_dir: Path) -> list[str]:
    issues: list[str] = []
    for row in _all_csv(run_dir / "command_log.csv"):
        result = row.get("result", "")
        ack = row.get("ack_done", "")
        if result not in {"DONE", "FREE"} and ack != "ACK_DONE":
            issues.append(
                f"command {row.get('command', '')} on {row.get('target_drones', '')} "
                f"state={row.get('state', '')} result={result} ack={ack}"
            )
    return issues[-20:]


def _reslot_issues(run_dir: Path) -> list[str]:
    issues: list[str] = []
    for row in _all_csv(run_dir / "reslot_path_log.csv"):
        result = row.get("result", "")
        if result != "FREE":
            issues.append(
                f"re-slot {row.get('turn', '')} check={row.get('check', '')} "
                f"measured={row.get('measured', '')} required={row.get('required', '')} "
                f"result={result} reason={row.get('reason', '')}"
            )
    return issues[-20:]


def _ranger_issues(run_dir: Path) -> list[str]:
    issues: list[str] = []
    for row in _tail_csv(run_dir / "ranger_log.csv", 40):
        valid = row.get("valid", "")
        front = _float(row.get("front", "inf"))
        left = _float(row.get("left", "inf"))
        right = _float(row.get("right", "inf"))
        up = _float(row.get("up", "inf"))
        if "false" in valid.lower():
            issues.append(f"Ranger invalid on {row.get('source', '')}: valid={valid}")
        if min(front, left, right, up) < 0.40:
            issues.append(
                f"Ranger critical on {row.get('source', '')}: "
                f"f/l/r/u={front:.2f}/{left:.2f}/{right:.2f}/{up:.2f}"
            )
    return issues[-12:]


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("inf")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


if __name__ == "__main__":
    main()
