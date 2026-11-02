from dataclasses import dataclass
from pathlib import Path

import typer
import wandb
import yaml
import importlib
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.spaces import Dict as DictSpace
from wandb.integration.sb3 import WandbCallback

from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.test import base_env_test
from uav_nav_obstacle_avoidance_rl.utils import env_factory
from uav_nav_obstacle_avoidance_rl.utils.curriculum_callback import CurriculumCallback
from uav_nav_obstacle_avoidance_rl.utils.eval_metrics_callback import CustomEvalCallback
from uav_nav_obstacle_avoidance_rl.utils.train_metadata_logger import save_model_metadata
from uav_nav_obstacle_avoidance_rl.utils.train_metrics_callback import TrainMetricsCallback
from uav_nav_obstacle_avoidance_rl.utils.cnn_extractor import Lidar2DCombinedExtractor
from uav_nav_obstacle_avoidance_rl.utils.saver import ModelSaver
from uav_nav_obstacle_avoidance_rl.utils.local_metrics_logger import LocalMetricsCallback

app = typer.Typer()
logger = config.logger

MONITOR_INFO_KEYWORDS = ("collision", "env_complete", "num_targets_reached", "num_obstacles")


@dataclass
class TrainParams:
    timesteps: int = 2_000_000
    eval_freq: int = 200_000
    n_envs: int = 2
    n_eval_episodes: int = 30
    log_interval: int = 10
    seed: int = 9
    verbose: int = 0


def _load_config(path: Path = config.EXP_CONFIG_PATH) -> dict:
    """Load experiment config from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def _merge_configs(base: dict, custom_path: Path) -> dict:
    """Merges custom config into base config."""
    with open(custom_path) as f:
        custom_cfg = yaml.safe_load(f) or {}

    # Ergänze/Überstrich die policy_kwargs
    if "policy_kwargs" in custom_cfg:
        base["policy_kwargs"] = custom_cfg["policy_kwargs"]
    return base


def _train(
        run: wandb.sdk.wandb_run.Run,
        params: TrainParams,
        exp_analysis: bool,
):
    """training logic used by both run_train and sweep agents"""
    env_config = dict(run.config["env"])
    ppo_config = dict(run.config["ppo"])
    curriculum_config = dict(run.config["curriculum"])

    # 1. policy_kwargs aus der W&B run.config laden (falls vorhanden)
    policy_kwargs = dict(run.config.get("policy_kwargs", {}))

    # 2. Automatische Extraktion der LiDAR-Dimensionen für CNN
    if "features_extractor_class" in policy_kwargs:
        # Falls die Klasse als String übergeben wird (z.B. aus YAML), dynamisch importieren
        if isinstance(policy_kwargs["features_extractor_class"], str):
            module_name, class_name = policy_kwargs["features_extractor_class"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            policy_kwargs["features_extractor_class"] = getattr(module, class_name)

        # Injected automatisch num_rays_v und num_rays_h direkt aus der env-Config
        extractor_kwargs = policy_kwargs.get("features_extractor_kwargs", {}).copy()
        extractor_kwargs["num_rays_v"] = env_config["lidar"]["num_rays_vertical"]
        extractor_kwargs["num_rays_h"] = env_config["lidar"]["num_rays_horizontal"]

        policy_kwargs["features_extractor_kwargs"] = extractor_kwargs

    monitor_info = {"info_keywords": MONITOR_INFO_KEYWORDS}

    # ----- create environments --------
    vec_env = make_vec_env(
        env_factory.make_flat_voyager,
        n_envs=params.n_envs,
        env_kwargs={**env_config, "visual_obstacles": False},
        monitor_kwargs=monitor_info,
    )

    base_env_test.analyse_env(vec_env)

    # create separate evaluation env
    vec_env_eval = make_vec_env(
        env_factory.make_flat_voyager,
        n_envs=params.n_envs,
        env_kwargs=env_config,
        monitor_kwargs=monitor_info,
    )

    # 1. ZENTRALEN EXPERIMENT-ORDNER IM PROJEKT ERSTELLEN
    lidar_name = f"{env_config['lidar']['fov_vertical']}deg_{env_config['lidar']['num_rays_horizontal']}x{env_config['lidar']['num_rays_vertical']}"
    run_name = run.name if hasattr(run, "name") and run.name else "PPO_Run"

    # Erzeugt z. B.: reports/45.0deg_72x9/PPO_Run/
    exp_dir = Path(config.REPORTS_DIR) / lidar_name / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Ordner für Modelle und Eval-Plots vorbereiten
    model_save_dir = exp_dir / "models"
    eval_plots_dir = exp_dir / "eval_logs"
    tb_log_dir = exp_dir / "tensorboard"

    # 2. METADATEN SPEICHERN
    save_model_metadata(
        save_dir=model_save_dir,
        env_config=env_config,
        policy_kwargs=policy_kwargs,
    )

    # 3. CALLBACKS MIT DEM PROJEKT-PFAD FÜTTERN
    callbacks = []
    wandb_callback = WandbCallback(verbose=params.verbose, model_save_path=None)

    train_callback = LocalMetricsCallback(
        save_dir=str(exp_dir),
        verbose=params.verbose,
    )

    adj_eval_freq = params.eval_freq // params.n_envs
    adj_n_eval_episodes = max(1, 32 // params.n_envs)

    eval_callback = CustomEvalCallback(
        vec_env_eval,
        best_model_save_path=str(model_save_dir),
        log_path=str(eval_plots_dir),  # <-- Hier landen die HTMLs jetzt direkt im Projekt!
        eval_freq=adj_eval_freq,
        n_eval_episodes=adj_n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=params.verbose,
        exp_analysis=exp_analysis,
        seed=params.seed,
    )

    callbacks.extend([wandb_callback, train_callback, eval_callback])

    if curriculum_config.pop('enabled', False):
        curriculum_callback = CurriculumCallback(**curriculum_config, verbose=params.verbose, eval_env=vec_env_eval)
        eval_callback.curriculum_callback = curriculum_callback
        callbacks.append(curriculum_callback)

    # 3. Model Init mit übergabe der policy_kwargs
    if isinstance(vec_env.observation_space, DictSpace):
        policy_type = "MultiInputPolicy"
    else:
        policy_type = "MlpPolicy"

    model = PPO(
        policy_type,
        vec_env,
        **ppo_config,
        policy_kwargs=policy_kwargs,
        verbose=params.verbose,
        tensorboard_log=str(tb_log_dir),
        seed=params.seed,
        device="cpu",  # TODO: test "cuda" ("cpu" is old value)
    )

    model.learn(
        total_timesteps=params.timesteps,
        callback=callbacks,
        progress_bar=True,
        log_interval=params.log_interval,
    )

    # 4. Letztes Modell und Konfiguration lokal speichern
    saver = ModelSaver(base_dir=exp_dir.as_posix())
    saver.save_checkpoint(
        model=model,
        lidar_config=env_config.get("lidar", {}),
        model_config=ppo_config,
        lidar_name="",  # Leer lassen, da exp_dir bereits den lidar_name enthält
        model_name="",  # Leer lassen
        metrics={"timesteps": params.timesteps},
        env_config=env_config,
    )

    # Optional: Auch direkt als W&B Modell-Artefakt als 'latest_model' sichern
    model.save(f"{model_save_dir}/latest_model.zip")


"""
uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train run-train --exp-name "name" --wandb-tags exp- --timesteps 2_000_000 --eval-freq 200_000                      
"""
@app.command()
def run_train(
    exp_name: str = "exp",
    config_file: str = "uav_nav_obstacle_avoidance_rl/modeling/config-mlp.yaml",  # Default MLP
    use_wandb: bool = True,
    timesteps: int = TrainParams.timesteps,
    eval_freq: int = TrainParams.eval_freq,
    n_envs: int = TrainParams.n_envs,
    n_eval_episodes: int = TrainParams.n_eval_episodes,
    log_interval: int = TrainParams.log_interval,
    seed: int = TrainParams.seed,
    verbose: int = TrainParams.verbose,
    exp_analysis: bool = True,
    wandb_project: str = "uav-nav-obstacle-avoidance-rl",
    wandb_tags: list[str] | None = None,
):
    """
    training script with W&B integration for experiment tracking
    """
    exp_config = _load_config()
    exp_config = _merge_configs(exp_config, Path(config_file))  # merges cnn2d or mlp

    # Mode festlegen: 'online' (hochladen) oder 'disabled' (stumm)
    wandb_mode = "online" if use_wandb else "disabled"

    with wandb.init(
        project=wandb_project,
        name=exp_name,
        tags=wandb_tags,
        config=exp_config,
        dir=config.REPORTS_DIR.as_posix(),
        monitor_gym=False,
        save_code=False,
        mode=wandb_mode,
        settings=wandb.Settings(x_disable_stats=True),
    ) as run:
        _train(
            run=run,
            params=TrainParams(
                timesteps=timesteps,
                eval_freq=eval_freq,
                n_envs=n_envs,
                n_eval_episodes=n_eval_episodes,
                log_interval=log_interval,
                seed=seed,
                verbose=verbose,
            ),
            exp_analysis=exp_analysis,
        )

"""
# terminal 1: create new sweep 
uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train sweep --exp-name "name" --wandb-tags exp- --count 25
# with custom sweep config:
uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train sweep --exp-name "name" --wandb-tags exp- --sweep-config uav_nav_obstacle_avoidance_rl/modeling/exp-6-sweep.yaml --count 25

# terminal 2: join existing sweep 
uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train sweep --exp-name "name" --wandb-tags exp- --sweep-id abc123 --count 25
"""
@app.command()
def sweep(
    exp_name: str = "exp-",
    wandb_project: str = "uav-nav-obstacle-avoidance-rl",
    count: int = 20,
    timesteps: int = TrainParams.timesteps,
    eval_freq: int = TrainParams.eval_freq,
    n_envs: int = TrainParams.n_envs,
    n_eval_episodes: int = TrainParams.n_eval_episodes,
    log_interval: int = TrainParams.log_interval,
    seed: int = TrainParams.seed,
    verbose: int = TrainParams.verbose,
    wandb_tags: list[str] | None = None,
    sweep_config: str = "uav_nav_obstacle_avoidance_rl/modeling/sweep.yaml",
    sweep_id: str | None = None,
):
    """
    W&B sweep using Bayesian optimization.

    Creates a new sweep from --sweep-config and launches an agent that runs
    `count` training runs with hyperparameters sampled by the sweep controller.

    To add more agents to an existing sweep, pass --sweep-id instead.
    """
    exp_config = _load_config()

    def _sweep_train():
        with wandb.init(
            config=exp_config,
            group=exp_name,
            dir=config.REPORTS_DIR.as_posix(),
            save_code=True,
            settings=wandb.Settings(x_disable_stats=True),
            tags=wandb_tags,
        ) as run:
            run.name = f"{exp_name}-{run.id}"
            _train(
                run=run,
                params=TrainParams(
                    timesteps=timesteps,
                    eval_freq=eval_freq,
                    n_envs=n_envs,
                    n_eval_episodes=n_eval_episodes,
                    log_interval=log_interval,
                    seed=seed,
                    verbose=verbose,
                ),
                exp_analysis=False,
            )

    if sweep_id:
        active_sweep_id = sweep_id
    else:
        loaded_sweep_config = _load_config(Path(sweep_config))
        loaded_sweep_config["name"] = f"{exp_name}-sweep-{loaded_sweep_config['name']}"
        active_sweep_id = wandb.sweep(sweep=loaded_sweep_config, project=wandb_project)
        logger.info(f"Created sweep with ID: {active_sweep_id}")

    logger.info(f"Starting sweep agent (sweep_id={active_sweep_id}, count={count})")
    wandb.agent(active_sweep_id, function=_sweep_train, count=count, project=wandb_project)

"""
uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train seed-sweep --exp-name "name" --wandb-tags seed --wandb-tags exp- --timesteps 2_000_000 --eval-freq 200_000
uv run python -m uav_nav_obstacle_avoidance_rl.modeling.train seed-sweep --exp-name "name" --wandb-tags seed --wandb-tags exp- --timesteps 2_000_000 --eval-freq 200_000 --sweep-id abc123
"""
@app.command()
def seed_sweep(
    exp_name: str = "exp-",
    wandb_project: str = "uav-nav-obstacle-avoidance-rl",
    timesteps: int = TrainParams.timesteps,
    eval_freq: int = TrainParams.eval_freq,
    n_envs: int = TrainParams.n_envs,
    n_eval_episodes: int = TrainParams.n_eval_episodes,
    log_interval: int = TrainParams.log_interval,
    verbose: int = TrainParams.verbose,
    wandb_tags: list[str] | None = None,
    sweep_config: str = "uav_nav_obstacle_avoidance_rl/modeling/seed-three.yaml",
    sweep_id: str | None = None,
):
    """
    Run the same experiment with multiple seeds using a W&B grid sweep.

    Seeds are defined in the sweep config YAML. Each seed becomes a separate
    W&B run, grouped together for easy comparison and aggregation.

    To add more agents to an existing seed sweep, pass --sweep-id instead.
    """
    exp_config = _load_config()

    def _seed_sweep_train():
        with wandb.init(
            config=exp_config,
            group=exp_name,
            dir=config.REPORTS_DIR.as_posix(),
            save_code=True,
            settings=wandb.Settings(x_disable_stats=True),
            tags=wandb_tags,
        ) as run:
            run.name = f"{exp_name}-{run.config['seed']}"
            _train(
                run=run,
                params=TrainParams(
                    timesteps=timesteps,
                    eval_freq=eval_freq,
                    n_envs=n_envs,
                    n_eval_episodes=n_eval_episodes,
                    log_interval=log_interval,
                    seed=run.config["seed"],
                    verbose=verbose,
                ),
                exp_analysis=False,
            )

    if sweep_id:
        active_sweep_id = sweep_id
    else:
        loaded_sweep_config = _load_config(Path(sweep_config))
        loaded_sweep_config["name"] = f"{exp_name}-seed-{loaded_sweep_config['name']}"
        active_sweep_id = wandb.sweep(sweep=loaded_sweep_config, project=wandb_project)
        logger.info(f"Created seed sweep with ID: {active_sweep_id}")

    logger.info(f"Starting seed sweep agent (sweep_id={active_sweep_id})")
    wandb.agent(active_sweep_id, function=_seed_sweep_train, project=wandb_project)


if __name__ == "__main__":
    app()
