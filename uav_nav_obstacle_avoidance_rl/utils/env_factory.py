import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

from uav_nav_obstacle_avoidance_rl.environment.cam_wrapper import ThirdPersonCamWrapper
from uav_nav_obstacle_avoidance_rl.environment.flaten_wrapper import (
    FlattenVectorVoyagerEnv,
)
from uav_nav_obstacle_avoidance_rl.environment.lidar_wrapper import (
    LidarFlattenWrapper,
    LidarObservationWrapper,
)
from uav_nav_obstacle_avoidance_rl.environment.normalize_obs_wrapper import (
    NormalizeObservationWrapper,
)
from uav_nav_obstacle_avoidance_rl.environment.reward_wrapper import (
    CustomRewardWrapper,
    PyFlytRewardWrapper,
)
from uav_nav_obstacle_avoidance_rl.environment.vector_voyager_env import (
    VectorVoyagerEnv,
)
from uav_nav_obstacle_avoidance_rl.utils.ego_grid_wrapper import EgoGridObservationWrapper


class DictEgoGridWrapper(gym.ObservationWrapper):
    """
    Formatiert Dict-Observations für Stable Baselines3 im Ego-Grid-Modus:
    Wandelt 'target_deltas' von Sequence/Liste in eine feste Box um.
    """

    def __init__(self, env: gym.Env, context_length: int = 1):
        super().__init__(env)
        self.context_length = context_length if context_length is not None else 1

        # Original Spaces kopieren
        obs_spaces = dict(self.env.observation_space.spaces)

        # target_deltas von Sequence zu fixer 1D Box konvertieren
        self.target_len = self.context_length * 3
        obs_spaces["target_deltas"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.target_len,), dtype=np.float32
        )

        self.observation_space = spaces.Dict(obs_spaces)

    def observation(self, obs: dict) -> dict:
        new_obs = dict(obs)

        # target_deltas flachklopfen und auf feste Länge bringen
        deltas = np.array(obs["target_deltas"], dtype=np.float32).flatten()

        if len(deltas) < self.target_len:
            deltas = np.pad(deltas, (0, self.target_len - len(deltas)))
        else:
            deltas = deltas[: self.target_len]

        new_obs["target_deltas"] = deltas
        return new_obs


def make_flat_voyager(**env_kwargs):
    """
    Create a flattened Vector Voyager environment
    """
    # 1. Extract wrapper-specific parameters (not accepted directly by VectorVoyagerEnv)
    perception_mode = env_kwargs.pop("perception_mode", "ego_grid")
    lidar_kwargs = env_kwargs.pop("lidar", {})
    ego_grid_kwargs = env_kwargs.pop("ego_grid", {})

    reward_cfg = env_kwargs.pop("reward", {})
    reward_type = reward_cfg.get("type", "custom")
    reward_kwargs = reward_cfg.get(reward_type, {}).copy()

    # w_smoothness aus reward_kwargs entfernen und für VectorVoyagerEnv freigeben
    w_smoothness = reward_kwargs.pop("w_smoothness", 0.1)
    if "w_smoothness" not in env_kwargs:
        env_kwargs["w_smoothness"] = w_smoothness

    # 2. Base environment
    env = VectorVoyagerEnv(**env_kwargs)

    # 3. Perception wrapper
    if perception_mode == "lidar":
        env = LidarObservationWrapper(env, **lidar_kwargs)
    elif perception_mode == "ego_grid":
        grid_size = ego_grid_kwargs.get("grid_size_meters", 10.0)
        grid_height = ego_grid_kwargs.get("grid_height_meters", 4.0)
        res = ego_grid_kwargs.get("resolution", 0.2)
        env = EgoGridObservationWrapper(
            env,
            grid_size_meters=grid_size,
            grid_height_meters=grid_height,
            resolution=res,
        )

    # 4. Reward wrappers
    if reward_type == "pyflyt":
        env = PyFlytRewardWrapper(env, **reward_kwargs)
    elif reward_type == "custom":
        env = CustomRewardWrapper(env, **reward_kwargs)

    # 5. Normalization wrappers
    env = NormalizeObservationWrapper(env)

    # 6. Flatten / Dict env
    if perception_mode == "lidar":
        env = LidarFlattenWrapper(env, context_length=env_kwargs.get("num_targets"))
    elif perception_mode == "ego_grid":
        # Nutzt den DictEgoGridWrapper anstelle des 1D-Flatten-Wrappers
        env = DictEgoGridWrapper(env, context_length=env_kwargs.get("num_targets"))
    else:
        env = FlattenVectorVoyagerEnv(env, context_length=env_kwargs.get("num_targets"))

    return env


def make_voyager_for_recording(video_folder, video_length, video_name, **env_kwargs):
    """
    Create environment for video recording with third-person camera and optional metrics.
    """
    env = make_vec_env(
        env_id=lambda **env_kwargs: ThirdPersonCamWrapper(make_flat_voyager(**env_kwargs)),
        n_envs=1,
        env_kwargs=env_kwargs,
    )

    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda step: step % video_length == 0,
        video_length=video_length,
        name_prefix=video_name,
    )

    return env
