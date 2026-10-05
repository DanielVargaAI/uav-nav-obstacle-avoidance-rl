from pathlib import Path
import typer
import time
from stable_baselines3 import PPO
from uav_nav_obstacle_avoidance_rl.utils import env_factory

app = typer.Typer()

@app.command()
def main(
    model_path: str = "reports/models/best_model.zip",
    stage_idx: int = 1,
    episodes: int = 5,
):
    """
    Lädt ein trainiertes Modell und rendert die Drohne in PyBullet 3D.
    """
    # 1. Basis-Environment erstellen (mit PyBullet GUI!)
    # Hinweis: In deiner env_factory make_flat_voyager mit render_mode="human" rufen
    env = env_factory.make_flat_voyager(
        render_mode="human",
        visual_obstacles=True,
    )

    # 2. Modell laden
    print(f"Lade Modell aus: {model_path}")
    model = PPO.load(model_path, env=env)

    # 3. Stage manuell setzen
    # Lade die Stage-Config aus deiner defaults.yaml falls nötig
    # env.set_stage(...)

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
            time.sleep(1 / 30.0)  # Agent-Hz (30 FPS Simulation bremsen)

        print(f"Episode {ep + 1} beendet! Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    app()

# Starten über:
# uv run python enjoy.py --model-path "path/to/best_model.zip" --stage-idx 1
