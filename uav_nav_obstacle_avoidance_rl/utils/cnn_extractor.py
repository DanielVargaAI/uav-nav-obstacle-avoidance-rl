import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Lidar2DCombinedExtractor(BaseFeaturesExtractor):
    """
    Dynamischer Feature Extractor für LiDAR 2D Grids beliebiger Auflösung.
    Voraussetzung: LiDAR-Daten stehen an Position 0 im Observation-Vektor!
    """

    def __init__(
            self,
            observation_space: spaces.Box,
            num_rays_v: int,
            num_rays_h: int,
            features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        self.num_rays_v = num_rays_v
        self.num_rays_h = num_rays_h

        # DYNAMISCHE Berechnungen (Keine hartcodierten Zahlen!)
        self.lidar_dim = self.num_rays_v * self.num_rays_h

        total_obs_dim = observation_space.shape[0]
        self.scalar_dim = total_obs_dim - self.lidar_dim

        assert self.scalar_dim >= 0, (
            f"Observation space dimension ({total_obs_dim}) ist kleiner als "
            f"die erforderlichen LiDAR-Strahlen ({self.lidar_dim})!"
        )

        # 1. 2D-CNN mit 3 Schichten (reduziert 72x72 -> 36x36 -> 18x18 -> 9x9)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Flatten(),
        )

        # CNN Output Dimension dynamisch ermitteln
        with th.no_grad():
            dummy_lidar = th.zeros(1, 1, self.num_rays_v, self.num_rays_h)
            cnn_out_dim = self.cnn(dummy_lidar).shape[1]

        # 2. Skalare Features (Attitude + Targets)
        self.scalar_head = nn.Sequential(
            nn.Linear(self.scalar_dim, 64),
            nn.ReLU(),
        )

        # 3. Fusion aus CNN-Features + Skalaren
        combined_dim = cnn_out_dim + 64
        self.linear = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # 1. LiDAR-Teil dynamisch gemäß lidar_dim herausschneiden
        lidar_flat = observations[:, : self.lidar_dim]
        scalars = observations[:, self.lidar_dim:]  # Die restlichen Skalare

        # 2. Reshape zu 2D-Grid (Batch, Channel=1, Vertikal, Horizontal)
        lidar_2d = lidar_flat.view(-1, 1, self.num_rays_v, self.num_rays_h)

        # 3. Forward-Pass
        cnn_features = self.cnn(lidar_2d)
        scalar_features = self.scalar_head(scalars)

        combined = th.cat([cnn_features, scalar_features], dim=1)
        return self.linear(combined)