import json
from uav_nav_obstacle_avoidance_rl import config
from pathlib import Path

logger = config.logger


def save_model_metadata(save_dir: Path, env_config: dict, policy_kwargs: dict):
    """Speichert die exakten Sensor- und CNN-Parameter als JSON im Modell-Ordner."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sensor-Spezifikationen extrahieren
    lidar_cfg = env_config.get("lidar", {})

    metadata = {
        "sensor": {
            "num_rays_vertical": lidar_cfg.get("num_rays_vertical"),
            "num_rays_horizontal": lidar_cfg.get("num_rays_horizontal"),
            "fov_vertical": lidar_cfg.get("fov_vertical"),
            "fov_horizontal": lidar_cfg.get("fov_horizontal"),
            "max_range": lidar_cfg.get("max_range"),
        },
        "architecture": {
            "is_cnn": "features_extractor_class" in policy_kwargs,
            "extractor_class": str(policy_kwargs.get("features_extractor_class", "MLP")),
            "extractor_kwargs": policy_kwargs.get("features_extractor_kwargs", {}),
        }
    }

    metadata_path = save_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Modell-Metadaten erfolgreich gespeichert unter: {metadata_path}")
