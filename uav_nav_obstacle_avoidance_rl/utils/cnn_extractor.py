import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Lidar2DCombinedExtractor(BaseFeaturesExtractor):
    """
    Dynamischer Feature Extractor für LiDAR 2D-Grids beliebiger Auflösung.
    Nutzt Adaptive Pooling, um bei jeder Eingabe-Rastergröße automatisch
    einen optimal dimensionierten Feature-Vektor für die Fusion zu erzeugen.
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
        self.lidar_dim = self.num_rays_v * self.num_rays_h

        total_obs_dim = observation_space.shape[0]
        self.scalar_dim = total_obs_dim - self.lidar_dim

        assert self.scalar_dim >= 0, (
            f"Observation space ({total_obs_dim}) ist kleiner als "
            f"die LiDAR-Dimension ({self.lidar_dim})!"
        )

        # --- DYNAMISCHE STRATEGIE ---
        # Je nach vertikaler Auflösung wählen wir ein geeignetes Adaptive Pooling Target:
        # Bei sehr flachen Grids (V <= 4) reichen 2 vertikale Bin-Zellen.
        # Bei höheren Grids (V > 4) nutzen wir 4 vertikale Bin-Zellen.
        target_v = 2 if self.num_rays_v <= 4 else 4
        target_h = 8 if self.num_rays_h <= 36 else 16

        # Target Feature-Map Dimension: (32 Channels x target_v x target_h)
        # Bsp 1: V=3, H=36 -> 32 * 2 * 8 = 512 CNN Features
        # Bsp 2: V=72, H=72 -> 32 * 4 * 16 = 2048 -> Heruntergebrochen auf 512 via Conv

        self.cnn = nn.Sequential(
            # Schicht 1: Lokale Features
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, padding_mode="circular"),
            nn.ReLU(),
            # Schicht 2: Räumliche Muster
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, padding_mode="circular"),
            nn.ReLU(),
            # Dynamisches Pooling: Erzeugt IMMER ein (target_v x target_h) Grid
            nn.AdaptiveAvgPool2d((target_v, target_h)),
            nn.Flatten(),
        )

        # CNN-Output-Dimension exakt ermitteln
        with th.no_grad():
            dummy_lidar = th.zeros(1, 1, self.num_rays_v, self.num_rays_h)
            cnn_out_dim = self.cnn(dummy_lidar).shape[1]

        # 2. Skalare Features (Attitude + Target Deltas)
        self.scalar_head = nn.Sequential(
            nn.Linear(self.scalar_dim, 64),
            nn.ReLU(),
        )

        # 3. Fusion (CNN-Features + Skalare -> Latent Features Dim)
        combined_dim = cnn_out_dim + 64
        self.linear = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # 1. Splitten: LiDAR am Anfang, Skalare dahinter
        lidar_flat = observations[:, : self.lidar_dim]
        scalars = observations[:, self.lidar_dim:]

        # 2. Reshape zu 2D-Grid (Batch, Channel=1, Vertikal, Horizontal)
        lidar_2d = lidar_flat.view(-1, 1, self.num_rays_v, self.num_rays_h)

        # 3. Processing & Concatenation
        cnn_features = self.cnn(lidar_2d)
        scalar_features = self.scalar_head(scalars)

        combined = th.cat([cnn_features, scalar_features], dim=1)
        return self.linear(combined)
