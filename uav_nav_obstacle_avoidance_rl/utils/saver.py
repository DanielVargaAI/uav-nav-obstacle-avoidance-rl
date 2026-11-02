import os
import json
import datetime
import torch


class ModelSaver:
    def __init__(self, base_dir="reports"):
        self.base_dir = base_dir

    def get_experiment_dir(self, lidar_config_name, model_name, custom_tag=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_lidar = str(lidar_config_name).replace(" ", "_").replace("/", "_")
        clean_model = str(model_name).replace(" ", "_").replace("/", "_")

        run_folder = f"run_{timestamp}"
        if custom_tag:
            run_folder += f"_{custom_tag}"

        save_path = os.path.join(self.base_dir, clean_lidar, clean_model, run_folder)
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def save_checkpoint(
            self,
            model,
            lidar_config,
            model_config,
            lidar_name,
            model_name,
            epoch=None,
            metrics=None,
            env_config=None,  # <-- HIER HINZUGEFÜGT
    ):
        save_dir = self.get_experiment_dir(lidar_name, model_name)

        # 1. Modellgewichte speichern
        model_file = os.path.join(save_dir, "model.pth")
        if hasattr(model, "save"):
            # Speichert Stable-Baselines3 zip-Modell
            model.save(os.path.join(save_dir, "model_sb3"))
        elif isinstance(model, torch.nn.Module):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }, model_file)
        else:
            torch.save(model, model_file)

        # 2. Konfigurationen als JSON speichern
        config_data = {
            "lidar_structure": lidar_config,
            "model_config": model_config,
            "env_config": env_config,  # <-- MIT SPEICHERN
            "save_timestamp": str(datetime.datetime.now()),
            "metrics": metrics
        }

        config_file = os.path.join(save_dir, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, default=str)

        print(f"[ModelSaver] Modell und Konfiguration gespeichert unter:\n  -> {save_dir}")
        return save_dir
