import os
from collections.abc import Sequence
from typing import Any, List, Literal, Optional

import gymnasium as gym
import numpy as np
from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler

from uav_nav_obstacle_avoidance_rl import config
from uav_nav_obstacle_avoidance_rl.environment.occupancy_grid import OccupancyGrid

logger = config.logger


# my waypoint environment based on PyFlyt/QuadX-Waypoints-v3
class VectorVoyagerEnv(QuadXBaseEnv):
    def __init__(
            self,
            start_pos: list,
            # boundary parameters
            grid_sizes: Sequence[float],
            cell_size: float,
            min_height: float,
            # waypoint parameters
            num_targets: int,
            target_distance_range: List[float] | None,
            use_yaw_targets: bool,
            goal_reach_distance: float,
            goal_reach_angle: float,
            # obstacle parameters
            num_obstacles_range: List[int] | None,
            obstacle_shapes: List[str] | None,
            obstacle_size_range: List[float] | None,
            visual_obstacles: bool | None,
            # simulation parameters
            max_duration_seconds: float,
            flight_mode: int,
            angle_representation: Literal["euler", "quaternion"],
            agent_hz: int,
            render_mode: Literal["human", "rgb_array"] | None,
            render_resolution: Sequence[int],
            # NEW: Flight control mode & smoothness reward parameters
            control_mode: Literal["velocity", "waypoint"] = "waypoint",
            max_waypoint_distance: float = 1.5,
            w_smoothness: float = 0.1,
    ):
        # Bei control_mode == "waypoint" erzwingen wir flight_mode = 7 (Position control x, y, z, yaw in PyFlyt)
        self.control_mode = control_mode
        self.max_waypoint_distance = max_waypoint_distance
        self.w_smoothness = w_smoothness
        self.prev_action: Optional[np.ndarray] = None

        effective_flight_mode = 7 if control_mode == "waypoint" else flight_mode

        super().__init__(
            start_pos=np.array([start_pos]),  # uav starting at pos [0, 0, 1]. Is randomized and overwritten in reset()
            flight_mode=effective_flight_mode,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,  # required in base class
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=(render_resolution[0], render_resolution[1]),
        )
        # boundary parameters
        self.grid_sizes = grid_sizes
        self.z_size = grid_sizes[2]  # z dimension stored separately
        self.cell_size = cell_size
        self.min_height = min_height
        # waypoint parameters
        self.num_targets = num_targets
        self.target_distance_range = target_distance_range
        # obstacle parameters
        self.num_obstacles_range = num_obstacles_range
        self.obstacle_shapes = obstacle_shapes
        self.obstacle_size_range = obstacle_size_range
        self.visual_obstacles = visual_obstacles
        self.obstacle_placement: Literal["random", "around_target", "along_path"] = "random"
        self.obstacle_distance_range: List[float] | None = None
        self.num_obs = 0
        self.obstacles = []  # store PyBullet body IDs
        self.boundary_wall_ids = []  # store PyBullet body IDs for walls
        self.walls_enabled = False  # toggled by curriculum set_stage()

        # init 3D occupancy grid (alle 3 Dimensionen uebergeben)
        self.occupancy_grid = OccupancyGrid(grid_sizes, cell_size, rng=self.np_random)

        # init waypoint handler
        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=5,  # dummy value
            min_height=min_height,
            np_random=self.np_random,
        )

        # define action space according to control mode
        if self.control_mode == "waypoint":
            # Relative offset (dx, dy, dz) normalized in [-1, 1]
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )
        else:
            # Velocity control (u, v, vr, vz)
            self.action_space = gym.spaces.Box(
                low=np.array([-26.0, -26.0, -2.62, -8.0], np.float32),
                high=np.array([26.0, 26.0, 2.62, 8.0], np.float32),
                dtype=np.float32,
            )

        # define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "attitude": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(11,),
                    dtype=np.float32,
                ),
                "target_deltas": gym.spaces.Sequence(
                    space=gym.spaces.Box(
                        low=-2 * max(self.grid_sizes),
                        high=2 * max(self.grid_sizes),
                        shape=(4,) if use_yaw_targets else (3,),
                        dtype=np.float32,
                    ),
                    stack=True,
                ),
            }
        )

    def reset(
            self,
            *,
            seed: None | int = None,
            options: None | dict[str, Any] = None,
            drone_options: None | dict[str, Any] = None,
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
        """
        reset the env: resets simulation state, waypoint handler, and counters
        """
        if options is None:
            options = {}

        # reset occupancy grid
        self.occupancy_grid.reset_occupancy()

        # reset action smoothness memory
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # --- set random UAV start position
        free_cell = self.occupancy_grid.get_random_free_cell()
        x, y = self.occupancy_grid.cell_to_world(free_cell[:2])
        z = self.np_random.uniform(self.min_height + 0.5, self.z_size)
        self.start_pos = np.array([[x, y, z]])
        self.occupancy_grid.mark_cells([free_cell], occupied=True)

        # --- start reset procedure
        with open(os.devnull, "w") as devnull:
            old_fd = os.dup(1)
            os.dup2(devnull.fileno(), 1)
            try:
                super().begin_reset(seed, drone_options=drone_options)
            finally:
                os.dup2(old_fd, 1)
                os.close(old_fd)

        # --- create new boundary walls
        self.boundary_wall_ids.clear()
        if self.walls_enabled:
            self._create_boundary_walls()

        # --- reset waypoint handler
        self.waypoints.reset(self.env, self.np_random)
        self.waypoints.targets = self._create_targets()

        # --- create new obstacles
        self.obstacles.clear()
        self._create_obstacles()

        # init tracked metrics
        self.info["num_targets_reached"] = 0
        self.info["num_obstacles"] = self.num_obs

        # finish reset
        super().end_reset()

        return self.state, self.info

    def compute_state(self) -> None:
        """Compute current state representation of the UAV."""
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()

        attitude = np.empty(11, dtype=np.float32)
        attitude[0] = ang_vel[2]
        attitude[1] = np.cos(ang_pos[2])  # yaw cos
        attitude[2] = np.sin(ang_pos[2])  # yaw sin
        attitude[3:6] = lin_vel
        attitude[6] = lin_pos[2]

        # Pad or slice action to fit attitude vector space size of 4
        if self.action.shape[0] == 3:
            attitude[7:10] = self.action
            attitude[10] = 0.0
        else:
            attitude[7:11] = self.action

        target_deltas = self.waypoints.distance_to_targets(
            ang_pos, lin_pos, quaternion
        ).astype(np.float32)

        self.state = {"attitude": attitude, "target_deltas": target_deltas}

    def step(self, action: np.ndarray):
        """
        Step environment forward using either relative waypoint targets or velocity commands.
        """
        self.info["sub_progress"] = []
        self.info["sub_distance"] = []

        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)

        if self.control_mode == "waypoint":
            # Current drone position
            uav_pos = self.env.state(0)[-1]  # Position [x, y, z]
            uav_yaw = self.env.state(0)[0][2]  # Aktueller Yaw-Winkel der Drohne

            delta_xyz = action_clipped * self.max_waypoint_distance
            target_pos = uav_pos + delta_xyz

            # Aktuellen Yaw beibehalten (statt starr 0.0)
            env_action = np.array([target_pos[0], target_pos[1], target_pos[2], uav_yaw], dtype=np.float32)
        else:
            env_action = action_clipped

        # Execute base PyFlyt step loop
        obs, reward, terminated, truncated, info = super().step(env_action)

        # Apply action smoothness penalty
        if self.prev_action is not None and self.w_smoothness > 0:
            action_diff = np.sum(np.square(action_clipped - self.prev_action))
            smoothness_penalty = float(self.w_smoothness * action_diff)
            reward -= smoothness_penalty
            info["smoothness_penalty"] = smoothness_penalty

        self.prev_action = action_clipped.copy()

        return obs, reward, terminated, truncated, info

    def compute_term_trunc_reward(self) -> None:
        """Compute termination/truncation and collect per-sub-step data."""
        self.reward = 0.0  # neutral; wrapper overwrites

        # ── truncation: time limit ──
        if self.step_count > self.max_steps:
            self.truncation |= True

        # ── termination: collision ──
        if np.any(self.env.contact_array[self.env.drones[0].Id]):
            self.info["collision"] = True
            self.termination |= True

        # ── per-sub-step dense shaping data ──
        self.info["sub_progress"].append(float(self.waypoints.progress_to_next_target))
        self.info["sub_distance"].append(float(self.waypoints.distance_to_next_target))

        # ── scalar waypoint info ──
        self.info["target_reached"] = bool(self.waypoints.target_reached)

        # ── advance waypoints on reach ──
        if self.waypoints.target_reached:
            self.waypoints.advance_targets()
            self.truncation |= self.waypoints.all_targets_reached
            self.info["env_complete"] = self.waypoints.all_targets_reached
            self.info["num_targets_reached"] = self.waypoints.num_targets_reached

    # ──────────────────────────────────────────────────────────────
    #  Environment Builders
    # ──────────────────────────────────────────────────────────────

    def set_stage(self, stage):
        """Adjust environment difficulty level according to curriculum stage."""
        self.num_obstacles_range = stage['num_obstacles_range']
        self.obstacle_shapes = stage["obstacle_shapes"]
        self.obstacle_size_range = stage["obstacle_size_range"]
        self.num_targets = stage['num_targets']
        self.target_distance_range = stage['target_distance_range']
        self.obstacle_placement = stage['obstacle_placement']
        self.obstacle_distance_range = stage['obstacle_distance_range']
        self.walls_enabled = stage.get('walls_enabled', False)

    def _create_targets(self) -> np.ndarray:
        """Create waypoint targets within configured distance range."""
        og = self.occupancy_grid
        start_xy = (self.start_pos[0, 0], self.start_pos[0, 1])

        targets = []
        for _ in range(self.num_targets):
            if self.target_distance_range is None:
                free_cell = og.get_random_free_cell()
            else:
                free_cell = self._get_free_cell_in_range(start_xy, self.target_distance_range)

            x, y = og.cell_to_world(free_cell[:2])
            z = self.np_random.uniform(self.min_height + 0.5, self.z_size)
            targets.append([x, y, z])
            og.mark_cells([free_cell], occupied=True)

        return np.array(targets, dtype=np.float32)

    def _create_boundary_walls(self):
        """Create collision walls at environment boundaries."""
        og = self.occupancy_grid
        half_thickness = 0.01
        z_half = self.z_size / 2

        walls = [
            # -x wall
            ([og.x_min - half_thickness, 0.0, z_half], [half_thickness, og.y_size / 2, z_half]),
            # +x wall
            ([og.x_max + half_thickness, 0.0, z_half], [half_thickness, og.y_size / 2, z_half]),
            # -y wall
            ([0.0, og.y_min - half_thickness, z_half], [og.x_size / 2, half_thickness, z_half]),
            # +y wall
            ([0.0, og.y_max + half_thickness, z_half], [og.x_size / 2, half_thickness, z_half]),
            # ceiling
            ([0.0, 0.0, self.z_size + half_thickness], [og.x_size / 2, og.y_size / 2, half_thickness]),
        ]

        for pos, half_extents in walls:
            col_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_BOX,
                halfExtents=half_extents,
            )
            body_id = self.env.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=int(col_id),
                basePosition=pos,
                baseOrientation=[0, 0, 0, 1],
            )

            self.boundary_wall_ids.append(body_id)

    def _create_obstacles(self):
        """Generate obstacles and position them in environment."""
        if self.num_obstacles_range is None:
            self.num_obs = 0
        else:
            self.num_obs = self.np_random.integers(
                low=self.num_obstacles_range[0],
                high=self.num_obstacles_range[1] + 1,
            )

            sub_set = max(2, self.num_obs // 3)
            obj_shapes = []
            for _ in range(sub_set):
                shape_type = self.np_random.choice(self.obstacle_shapes)
                size_fraction_np = self.np_random.uniform(
                    low=self.obstacle_size_range[0],
                    high=self.obstacle_size_range[1],
                    size=(3,),
                )
                size_fractions = size_fraction_np.tolist()

                obj_shape = self._create_object_shape(
                    object_type=shape_type,
                    scaled_radius=size_fractions[0],
                    height=1.0,
                    half_extents=size_fractions,
                )
                obj_shapes.append(obj_shape)

            for _ in range(self.num_obs):
                free_cell = self._sample_obstacle_cell()
                x, y = self.occupancy_grid.cell_to_world(free_cell[:2])
                z = self.z_size / 2
                free_position = [x, y, z]

                shape = self.np_random.choice(obj_shapes)
                collision_id = shape[0]
                visual_id = shape[1]

                self._spawn_object(collision_id=collision_id, base_position=free_position, visual_id=visual_id)

                # Markiere das gesamte 3D-Volumen des Hindernisses im Grid
                i, j = free_cell[:2]
                self.occupancy_grid.mark_cells([(i, j, k) for k in range(self.occupancy_grid.nz)], occupied=True)

    def _create_object_shape(
            self,
            object_type: Literal['cylinder', 'sphere', 'box'],
            scaled_radius: float = 1.0,
            height: float = 1.0,
            half_extents: Optional[List[float]] = None,
    ):
        """Create collision and visual shapes for PyBullet."""
        visual_id = -1
        if object_type == 'cylinder':
            r = self.cell_size * scaled_radius
            h = self.grid_sizes[-1] * height

            collision_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_CYLINDER, radius=r, height=h,
            )
            if self.visual_obstacles:
                visual_id = self.env.createVisualShape(
                    shapeType=self.env.GEOM_CYLINDER, radius=r, length=h,
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                )

        elif object_type == 'sphere':
            r = self.cell_size * scaled_radius
            collision_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_SPHERE, radius=r,
            )

            if self.visual_obstacles:
                visual_id = self.env.createVisualShape(
                    shapeType=self.env.GEOM_SPHERE, radius=r,
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                )

        elif object_type == 'box':
            if half_extents is None:
                half_extents = [0.5, 0.5, 0.5]
            he = [
                self.cell_size * half_extents[0] / 2.0,
                self.cell_size * half_extents[1] / 2.0,
                self.grid_sizes[-1] * half_extents[2] / 2.0,
            ]
            collision_id = self.env.createCollisionShape(
                shapeType=self.env.GEOM_BOX, halfExtents=he,
            )
            if self.visual_obstacles:
                visual_id = self.env.createVisualShape(
                    shapeType=self.env.GEOM_BOX, halfExtents=he,
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                )

        return int(collision_id), int(visual_id)

    def _spawn_object(
            self,
            collision_id,
            base_position: List[float],
            base_orientation: List[int] | None = None,
            visual_id=-1,
    ):
        """Spawn multibody object in PyBullet world."""
        object_id = self.env.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(visual_id),
            baseCollisionShapeIndex=int(collision_id),
            basePosition=base_position,
            baseOrientation=base_orientation if base_orientation is not None else [0, 0, 0, 1],
        )
        self.obstacles.append(object_id)

    def _sample_obstacle_cell(self) -> tuple[int, ...]:
        """Pick a free cell using configured placement strategy."""
        if self.obstacle_placement == "around_target":
            target_xy = tuple(self.waypoints.targets[0][:2])
            if self.obstacle_distance_range is not None:
                return self._get_free_cell_in_range(target_xy, self.obstacle_distance_range)
            return self.occupancy_grid.get_random_free_cell()

        if self.obstacle_placement == "along_path":
            return self._get_free_cell_along_path()

        return self.occupancy_grid.get_random_free_cell()

    def _get_free_cell_in_range(
            self,
            center_xy: tuple[float, float],
            distance_range: Sequence[float],
    ) -> tuple[int, ...]:
        """Sample free cell within distance range of center coordinate."""
        og = self.occupancy_grid
        ci, cj = og.world_to_cell(*center_xy)
        r_min, r_max = distance_range

        cells_inner = int(np.floor(r_min / og.cell_size))
        cells_outer = int(np.ceil(r_max / og.cell_size))

        i_lo = max(ci - cells_outer, 0)
        i_hi = min(ci + cells_outer, og.nx - 1)
        j_lo = max(cj - cells_outer, 0)
        j_hi = min(cj + cells_outer, og.ny - 1)

        ii, jj = np.mgrid[i_lo:i_hi + 1, j_lo:j_hi + 1]

        outside_inner = (np.abs(ii - ci) >= cells_inner) | (np.abs(jj - cj) >= cells_inner)

        # 2D Projektion der Belegung prüfen (frei über alle Z)
        free_2d = ~np.any(og.grid[i_lo:i_hi + 1, j_lo:j_hi + 1, :], axis=-1)

        valid_cells = np.argwhere(outside_inner & free_2d)
        if len(valid_cells) == 0:
            raise ValueError("No free cells in the requested distance range")

        pick = self.np_random.integers(len(valid_cells))
        return (int(valid_cells[pick, 0] + i_lo), int(valid_cells[pick, 1] + j_lo), 0)

    def _get_free_cell_along_path(self, spread: int = 2):
        """Sample free cell near start->target path."""
        og = self.occupancy_grid

        start_ij = np.array(og.world_to_cell(*self.start_pos[0, :2]))
        target_ij = np.array(og.world_to_cell(*self.waypoints.targets[0][:2]))

        n_points = int(np.linalg.norm(target_ij - start_ij)) + 1
        line_cells = np.round(
            np.linspace(start_ij, target_ij, max(n_points, 2))
        ).astype(int)

        offsets = self.np_random.integers(-spread, spread + 1, size=line_cells.shape)
        candidates = np.clip(
            line_cells + offsets,
            [0, 0],
            [og.nx - 1, og.ny - 1],
        )

        mask = np.array([
            not np.any(og.grid[c[0], c[1], :])
            and not np.array_equal(c, start_ij)
            and not np.array_equal(c, target_ij)
            for c in candidates
        ])

        valid = candidates[mask]
        if len(valid) == 0:
            return og.get_random_free_cell()

        idx = self.np_random.integers(len(valid))
        return (int(valid[idx][0]), int(valid[idx][1]), 0)
