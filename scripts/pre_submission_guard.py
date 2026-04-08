import ast
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "inference.py"
OPENENV_YAML = ROOT / "openenv.yaml"
SCENARIOS = ROOT / "configs" / "scenarios.yaml"


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def run_openenv_validate() -> None:
    cmd = [str(ROOT / ".venv" / "Scripts" / "openenv.exe"), "validate"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        fail("openenv validate failed")
    ok("openenv validate passed")


def check_inference_contract() -> None:
    text = INFERENCE.read_text(encoding="utf-8")
    tree = ast.parse(text)

    if 'os.environ["API_KEY"]' not in text:
        fail("inference.py must read API_KEY from os.environ")
    if 'os.environ["API_BASE_URL"]' not in text:
        fail("inference.py must read API_BASE_URL from os.environ")
    ok("inference.py uses injected proxy credentials")

    normalized = text.replace(" ", "").replace("\n", "")
    if "openai.OpenAI(base_url=API_BASE_URL,api_key=API_KEY)" not in normalized:
        fail("OpenAI client must be initialized with base_url=API_BASE_URL and api_key=API_KEY")
    ok("OpenAI client initialization matches validator requirement")

    # Static TASKS / score bound checks
    tasks_len = None
    min_score = None
    max_score = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TASKS":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        tasks_len = len(node.value.elts)
                if isinstance(target, ast.Name) and target.id == "MIN_SCORE":
                    if isinstance(node.value, ast.Constant):
                        min_score = float(node.value.value)
                if isinstance(target, ast.Name) and target.id == "MAX_SCORE":
                    if isinstance(node.value, ast.Constant):
                        max_score = float(node.value.value)

    if tasks_len is None or tasks_len < 3:
        fail("TASKS must contain at least 3 entries")
    if min_score is None or max_score is None:
        fail("MIN_SCORE and MAX_SCORE must be defined")
    if not (0.0 < min_score < max_score < 1.0):
        fail("MIN_SCORE and MAX_SCORE must satisfy 0 < MIN_SCORE < MAX_SCORE < 1")
    ok(f"tasks={tasks_len}, MIN_SCORE={min_score}, MAX_SCORE={max_score} satisfy Phase-2 bounds")


def check_manifest_consistency() -> None:
    oenv = yaml.safe_load(OPENENV_YAML.read_text(encoding="utf-8"))
    scen = yaml.safe_load(SCENARIOS.read_text(encoding="utf-8"))

    tasks = oenv.get("tasks", {})
    if not isinstance(tasks, dict) or len(tasks) < 3:
        fail("openenv.yaml must define at least 3 tasks")

    missing_scenarios = [k for k in tasks.keys() if k not in scen]
    if missing_scenarios:
        fail(f"scenarios.yaml missing scenario keys: {missing_scenarios}")

    missing_cfg_files = []
    for key, task_cfg in tasks.items():
        path = task_cfg.get("config_path")
        if not path:
            fail(f"Task '{key}' missing config_path")
        cfg = ROOT / path
        if not cfg.exists():
            missing_cfg_files.append(path)
    if missing_cfg_files:
        fail(f"Missing task config files: {missing_cfg_files}")

    ok("openenv.yaml task registry is consistent with configs/")


def main() -> None:
    print("Running pre-submission guard...")
    check_inference_contract()
    check_manifest_consistency()
    run_openenv_validate()
    ok("All guard checks passed. Repository is submission-safe.")


if __name__ == "__main__":
    main()
