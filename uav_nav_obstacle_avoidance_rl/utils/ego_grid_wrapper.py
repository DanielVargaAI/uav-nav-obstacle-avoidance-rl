import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.ndimage import rotate


class EgoGridObservationWrapper(gym.ObservationWrapper):
    """
    Erstellt eine lokale Ego-Centric Occupancy Grid Map um die Drohne herum.
    Schneidet eine Karte der Größe `grid_size_meters` aus und richtet sie
    an der Orientierung (Yaw) der Drohne aus.
    """

    def __init__(
        self,
        env: gym.Env,
        grid_size_meters: float = 10.0,
        grid_height_meters: float = 4.0,
        resolution: float = 0.2,
        add_to_obs: str = "separate",
    ):
        super().__init__(env)
        self._base_env = self.unwrapped

        self.grid_size_meters = grid_size_meters
        self.grid_height_meters = grid_height_meters
        self.resolution = resolution

        self.grid_dim = int(np.round(grid_size_meters / resolution))        # 10 / 0.2 = 50
        self.grid_z_dim = int(np.round(grid_height_meters / resolution))    # 4.0 / 0.2 = 20

        self.add_to_obs = add_to_obs

        # Shape ist 20 x 50 x 50 (Höhe x Breite x Tiefe)
        self.ego_grid_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.grid_z_dim, self.grid_dim, self.grid_dim),
            dtype=np.float32,
        )

        if self.add_to_obs == "separate":
            new_spaces = dict(self.env.observation_space.spaces)
            new_spaces["ego_grid"] = self.ego_grid_space
            self.observation_space = spaces.Dict(new_spaces)

    def _extract_ego_grid(self) -> np.ndarray:
        """
        Schneidet ein lokales 3D-Voxel-Grid (z.B. 20 x 50 x 50) um die Drohne herum aus
        und richtet es an deren Yaw-Winkel aus.
        """
        og = self._base_env.occupancy_grid  # Erwartet 3D-Occupancy-Grid: Shape (Nx, Ny, Nz)

        # 1. Drohnenzustand abfragen
        uav_pos = self._base_env.env.state(0)[-1]  # [x, y, z]
        ang_pos = self._base_env.env.state(0)[0]  # [roll, pitch, yaw]
        yaw = ang_pos[2]

        x_uav, y_uav, z_uav = uav_pos[0], uav_pos[1], uav_pos[2]

        # 2. Bounding-Ausschnitt für X/Y (mit Rotations-Puffer)
        half_dim = self.grid_dim // 2
        pad_size = int(half_dim * 1.5)

        # 3. Z-Ausschnitt (Höhe: z_uav - 2m bis z_uav + 2m)
        half_z_dim = self.grid_z_dim // 2  # z.B. 20 // 2 = 10 Zellen (2.0 Meter)

        # Weltkoordinaten in Grid-Zellen-Indizes umrechnen
        ci, cj, ck = og.world_to_cell_3d(x_uav, y_uav, z_uav)

        # Padding auf dem 3D-Grid für Randbereiche
        padded_grid = np.pad(
            og.grid.astype(np.float32),
            pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode="constant",
            constant_values=1.0,  # Wände/Außerhalb als blockiert (1.0)
        )

        # X/Y-Ausschnitt
        start_i = ci + pad_size - pad_size
        end_i = ci + pad_size + pad_size
        start_j = cj + pad_size - pad_size
        end_j = cj + pad_size + pad_size

        # Z-Ausschnitt (Höhe) direkt mit Grenzen schneiden
        start_k = max(0, ck - half_z_dim)
        end_k = min(og.grid.shape[2], ck + half_z_dim)

        # Crop in 3D: [X, Y, Z]
        crop_3d = padded_grid[start_i:end_i, start_j:end_j, start_k:end_k]

        # 4. Um Yaw-Winkel rotieren (Rotation in der X-Y Ebene, also achsenweise über Z)
        yaw_deg = np.degrees(yaw)
        rotated_3d = rotate(crop_3d, angle=-yaw_deg, axes=(0, 1), reshape=False, order=0, cval=1.0)

        # 5. Auf finale Dim-Größe (50x50) zentrieren
        center_x = rotated_3d.shape[0] // 2
        center_y = rotated_3d.shape[1] // 2

        final_crop = rotated_3d[
                     center_x - half_dim: center_x + half_dim,
                     center_y - half_dim: center_y + half_dim,
                     :
                     ]

        # Transponieren auf PyTorch Standard: (Channels/Höhe, Height/X, Width/Y) -> (20, 50, 50)
        # Falls `final_crop` die Form (X, Y, Z) hat:
        final_ego_grid = np.transpose(final_crop, (2, 0, 1))

        # Sicherheitsprüfung bezüglich der Shape
        expected_shape = (self.grid_z_dim, self.grid_dim, self.grid_dim)
        if final_ego_grid.shape != expected_shape:
            final_ego_grid = np.ones(expected_shape, dtype=np.float32)

        return final_ego_grid.astype(np.float32)

    def observation(self, observation: dict) -> dict:
        ego_grid_data = self._extract_ego_grid()

        if self.add_to_obs == "separate":
            return {**observation, "ego_grid": ego_grid_data}

        return observation
