import csv
from pathlib import Path
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LocalMetricsCallback(BaseCallback):
    """
    Kompakter Callback zum lokalen Speichern von Trainings- und Evaluationsmetriken
    ohne Abhängigkeit von WandB.
    """

    def __init__(self, save_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.save_dir / "training_metrics.csv"

        self.headers = [
            "episode",
            "timestep",
            "reward",
            "length",
            "success",
            "collision",
            "timeout",
            "stage",
            "final_distance_to_target",
        ]

        # CSV initialisieren
        with open(self.csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

        self.episode_count = 0

    def _on_step(self) -> bool:
        # Prüfen, ob eine Episode in einem der Vector-Envs beendet wurde
        for env_idx, done in enumerate(self.locals["dones"]):
            if done:
                self.episode_count += 1
                info = self.locals["infos"][env_idx]

                # Metriken aus der Environment-Info auslesen
                ep_reward = info.get("episode", {}).get("r", 0.0)
                ep_length = info.get("episode", {}).get("l", 0)

                success = 1 if info.get("env_complete", False) or info.get("is_success", False) else 0
                collision = 1 if info.get("collision", False) else 0
                timeout = 1 if ep_length >= 2400 and not success and not collision else 0

                stage = info.get("stage", 0)
                dist = info.get("distance_to_target", np.nan)

                row = [
                    self.episode_count,
                    self.num_timesteps,
                    ep_reward,
                    ep_length,
                    success,
                    collision,
                    timeout,
                    stage,
                    dist,
                ]

                with open(self.csv_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

        return True
