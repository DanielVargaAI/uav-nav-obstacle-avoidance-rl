import itertools
from pathlib import Path
import typer
import yaml

from uav_nav_obstacle_avoidance_rl.modeling.train import run_train

app = typer.Typer()


def _update_yaml_lidar_config(config_path: Path, lidar_updates: dict):
    """Lädt die YAML-Defaults, überschreibt die LiDAR-Werte und speichert sie temporär."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # LiDAR-Parameter in der Config aktualisieren
    if "env" not in cfg:
        cfg["env"] = {}
    if "lidar" not in cfg["env"]:
        cfg["env"]["lidar"] = {}

    cfg["env"]["lidar"].update(lidar_updates)

    # In temporäre Run-Config zurückschreiben
    temp_config_path = config_path.parent / "temp_exp_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    return temp_config_path


@app.command()
def main(
    config_file: str = "uav_nav_obstacle_avoidance_rl/modeling/config-cnn2d.yaml",
    use_wandb: bool = False,
    timesteps: int = 2_000_000,
    eval_freq: int = 200_000,
    n_envs: int = 16,
):
    """
    Generiert automatisch Permutationen aus der Test-Matrix und startet die Experimente nacheinander.
    """
    # --- TEST-MATRIX DEFINITION ---
    experiment_matrix = {
        "30deg_vert": {
            "fov_vertical": 30.0,
            "fov_horizontal": 360.0,
            "num_rays_horizontal": [36, 72, 128],
            "num_rays_vertical": [3, 12, 32],
        },
        # "45deg_vert": {
        #     "fov_vertical": 45.0,
        #     "fov_horizontal": 360.0,
        #     "num_rays_horizontal": [72, 128, 256],
        #     "num_rays_vertical": [12, 32, 64],
        # },
    }
    # ------

    total_runs = 0
    # Erstes Durchzählen der geplanten Läufe
    for group, params in experiment_matrix.items():
        h_rays = params["num_rays_horizontal"]
        v_rays = params["num_rays_vertical"]
        total_runs += len(h_rays) * len(v_rays)

    print(f"🚀 Starte Experiment-Grid-Search! Geplante Läufe gesamt: {total_runs}\n")

    current_run = 1

    for group_name, group_params in experiment_matrix.items():
        fov_v = group_params["fov_vertical"]
        fov_h = group_params["fov_horizontal"]
        h_list = group_params["num_rays_horizontal"]
        v_list = group_params["num_rays_vertical"]

        # Erstelle alle Permutationen von Horizontal x Vertikal
        for h_rays, v_rays in itertools.product(h_list, v_list):
            exp_name = f"{group_name}_H{h_rays}_V{v_rays}"

            print(f"==================================================")
            print(f"▶️ [{current_run}/{total_runs}] Starte Experiment: {exp_name}")
            print(f"   FOV: {fov_h}° x {fov_v}° | Strahlen: {h_rays}h x {v_rays}v")
            print(f"==================================================")

            # 1. Parameter für diesen spezifischen Run vorbereiten
            lidar_override = {
                "fov_horizontal": fov_h,
                "fov_vertical": fov_v,
                "num_rays_horizontal": h_rays,
                "num_rays_vertical": v_rays,
            }

            # 2. Temporäre Config erzeugen
            temp_cfg_path = _update_yaml_lidar_config(
                Path("uav_nav_obstacle_avoidance_rl/modeling/config-defaults.yaml"),
                lidar_override,
            )

            try:
                # 3. Direkt deine bestehende run_train Funktion aufrufen
                run_train(
                    exp_name=exp_name,
                    config_file=config_file,
                    use_wandb=use_wandb,
                    timesteps=timesteps,
                    eval_freq=eval_freq,
                    n_envs=n_envs,
                    wandb_tags=["grid-search", group_name, f"{h_rays}x{v_rays}"],
                )
            except Exception as e:
                print(f"❌ Fehler bei Experiment {exp_name}: {e}")
                print("Fahre mit dem nächsten Experiment fort...\n")

            current_run += 1

    print("\n🎉 Alle Experiment-Permutationen erfolgreich abgeschlossen!")


if __name__ == "__main__":
    app()
