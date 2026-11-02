import json
import time
from pathlib import Path
import typer
import yaml
from stable_baselines3 import PPO
from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.utils import env_factory

app = typer.Typer()


def _load_base_config() -> dict:
    """Lädt die Basis-Umgebungskonfiguration aus config-defaults.yaml."""
    defaults_path = config.EXP_CONFIG_PATH
    if defaults_path.exists():
        with open(defaults_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            return cfg.get("env", {})
    return {}


def _load_stage_config(stage_idx: int) -> dict:
    """Lädt die Konfiguration für eine bestimmte Stage aus der config-defaults.yaml."""
    defaults_path = config.EXP_CONFIG_PATH
    if defaults_path.exists():
        with open(defaults_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            curriculum_stages = cfg.get("curriculum", {}).get("stages", [])
            if 0 <= stage_idx < len(curriculum_stages):
                return curriculum_stages[stage_idx]
    return {}


@app.command()
def main(
    model_path: str = "reports/45.0deg_72x9/dummy-ebf3bloq/run_20261027_123707/model_sb3.zip",
    stage_idx: int = 1,
    episodes: int = 5,
    show_lidar: bool = typer.Option(False, "--show-lidar/--no-show-lidar", help="LiDAR-Strahlen in PyBullet rendern"),
):
    """
    Lädt ein trainiertes Modell und rendert die Drohne in PyBullet 3D.
    Liest automatisch die Konfiguration aus der config.json im Modellordner oder verwendet Defaults.
    """
    model_file = Path(model_path)
    config_file = model_file.parent / "config.json"

    # 1. Basis-Defaults aus config-defaults.yaml laden
    env_kwargs = _load_base_config()

    # 2. Falls eine gespeicherte config.json existiert, diese drüberlegen
    if config_file.exists():
        print(f"📄 Config gefunden: {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        if "env_config" in saved_data:
            env_kwargs.update(saved_data["env_config"])
        elif "lidar_structure" in saved_data:
            env_kwargs["lidar"] = saved_data["lidar_structure"]

    # Render-Modus erzwingen
    env_kwargs["render_mode"] = "human"
    env_kwargs["visual_obstacles"] = True

    # LiDAR-Sichtbarkeit setzen (falls Lidar aktiv ist)
    if "lidar" in env_kwargs and isinstance(env_kwargs["lidar"], dict):
        env_kwargs["lidar"]["show_rays"] = show_lidar

    # 3. Environment mit Basis-Kwargs erstellen
    env = env_factory.make_flat_voyager(**env_kwargs)

    # 4. Stage-Parameter sauber NACH der Erstellung über set_stage übergeben
    stage_cfg = _load_stage_config(stage_idx)
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env

    if stage_cfg and hasattr(raw_env, "set_stage"):
        print(f"🎯 Wende Stage {stage_idx} via set_stage() an: {stage_cfg.get('desc', '')}")
        raw_env.set_stage(stage_cfg)
    elif hasattr(raw_env, "set_stage"):
        raw_env.set_stage(stage_idx)

    # 5. Modell laden
    print(f"🤖 Lade Modell aus: {model_path}")
    model = PPO.load(model_path, env=env)

    # 6. Simulation ausführen
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        print(f"\n--- Starte Episode {ep + 1}/{episodes} in Stage {stage_idx} ---")

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(1 / 30.0)

        print(f"Episode {ep + 1} beendet! Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    app()
