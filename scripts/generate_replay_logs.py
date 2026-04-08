from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment import WarehouseEnvironment
from models import RobotStatus


SCENARIOS = ("easy", "medium", "hard")
MAX_STEPS = 50


def heuristic_action(env: WarehouseEnvironment, scenario: str) -> dict:
    state = env.current_state
    if state is None:
        return {"command_type": "WAIT", "target_id": None}

    if scenario == "easy":
        if state.active_exceptions:
            return {"command_type": "REROUTE_ORDER", "target_id": state.active_exceptions[0].id}
        return {"command_type": "WAIT", "target_id": None}

    if scenario == "medium":
        for comp_name, qty in state.inventory_status.items():
            if qty <= 50:
                return {
                    "command_type": "REQUEST_RESTOCK",
                    "target_id": None,
                    "parameters": {"component_name": comp_name},
                }
        if state.active_exceptions:
            return {"command_type": "REROUTE_ORDER", "target_id": state.active_exceptions[0].id}
        return {"command_type": "WAIT", "target_id": None}

    if scenario == "hard":
        for comp_name, qty in state.inventory_status.items():
            if qty <= 50:
                return {
                    "command_type": "REQUEST_RESTOCK",
                    "target_id": None,
                    "parameters": {"component_name": comp_name},
                }
        for robot in state.robots:
            if robot.status == RobotStatus.SENSOR_FAILURE:
                return {"command_type": "RE_POLL_SENSOR", "target_id": robot.id}
        for robot in state.robots:
            if robot.status != RobotStatus.ACTIVE:
                return {"command_type": "DISPATCH_MAINTENANCE", "target_id": robot.id}
        if state.active_exceptions:
            return {"command_type": "REROUTE_ORDER", "target_id": state.active_exceptions[0].id}
        return {"command_type": "WAIT", "target_id": None}

    return {"command_type": "WAIT", "target_id": None}


def run_scenario(scenario: str) -> str:
    env = WarehouseEnvironment("configs/warehouse_map.json", "configs/scenarios.yaml", scenario)
    env.reset()

    initial_exceptions = len(env.current_state.active_exceptions)
    total_reward = 0.0
    lines = [f"scenario={scenario}", f"initial_exceptions={initial_exceptions}"]

    for step in range(1, MAX_STEPS + 1):
        before = len(env.current_state.active_exceptions)
        action = heuristic_action(env, scenario)
        _, reward, done, info = env.step(action)
        after = len(env.current_state.active_exceptions)
        total_reward += reward
        lines.append(
            "step="
            + str(step)
            + " action="
            + str(action)
            + f" reward={reward:.4f} exceptions={before}->{after} done={str(done).lower()}"
        )
        if info.get("event_log"):
            lines.append("events=" + " | ".join(info["event_log"]))
        if done:
            break

    remaining = len(env.current_state.active_exceptions)
    resolved = initial_exceptions - remaining
    success = remaining == 0
    lines.append(f"final_steps={env.current_state.time_step}")
    lines.append(f"final_reward={total_reward:.4f}")
    lines.append(f"resolved={resolved}")
    lines.append(f"remaining={remaining}")
    lines.append(f"success={str(success).lower()}")
    return "\n".join(lines) + "\n"


def main() -> None:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIOS:
        data = run_scenario(scenario)
        (log_dir / f"{scenario}_run.txt").write_text(data, encoding="utf-8")

    print("Generated logs:", ", ".join(f"{s}_run.txt" for s in SCENARIOS))


if __name__ == "__main__":
    main()
