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


class EgoGridCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor für 3D Ego-Grid (Z, H, W) + Vektor-Observations.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256, **kwargs):
        super().__init__(observation_space, features_dim=features_dim)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "ego_grid":
                # subspace.shape ist (20, 50, 50) -> 20 Eingabekanäle (Z-Ebenen)
                n_input_channels = subspace.shape[0] if len(subspace.shape) == 3 else 1

                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),  # -> 32x25x25
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),                 # -> 64x13x13
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),                # -> 128x7x7
                    nn.ReLU(),
                    nn.Flatten(),
                )

                with th.no_grad():
                    dummy_input = th.zeros(1, *subspace.shape)  # Shape: (1, 20, 50, 50)
                    cnn_out_dim = cnn(dummy_input).shape[1]

                extractors[key] = cnn
                total_concat_size += cnn_out_dim

            else:
                vector_dim = int(th.prod(th.tensor(subspace.shape)))
                extractors[key] = nn.Sequential(
                    nn.Linear(vector_dim, 64),
                    nn.ReLU(),
                )
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)

        self.policy_net = nn.Sequential(
            nn.Linear(total_concat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            obs = observations[key]

            if key == "ego_grid":
                # Behalte Shape (Batch, Z, H, W) für PyTorch Conv2d bei
                if len(obs.shape) == 3:  # Falls Batch-Dimension fehlt
                    obs = obs.unsqueeze(0)
                encoded = extractor(obs)
            else:
                if len(obs.shape) > 2:
                    obs = obs.flatten(start_dim=1)
                encoded = extractor(obs)

            encoded_tensor_list.append(encoded)

        combined = th.cat(encoded_tensor_list, dim=1)
        return self.policy_net(combined)
