import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Lidar2DCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor für SB3:
    - Verarbeitet 2D-LiDAR-Scans (Vertical x Horizontal) über ein CNN mit Circular Padding.
    - Verbindet die extrahierten LiDAR-Features mit Attitüde und Zielkoordinaten.
    """

    def __init__(
            self,
            observation_space: spaces.Dict,
            num_rays_v: int,
            num_rays_h: int,
            features_dim: int = 256,
    ):
        # 1 Feature-Dim vorübergehend 1 setzen, wird unten aktualisiert
        super().__init__(observation_space, features_dim=features_dim)

        self.num_rays_v = num_rays_v
        self.num_rays_h = num_rays_h

        # 1. LiDAR CNN-Zweig
        # Input-Shape für Conv2D: (Batch, Channels=1, Height=num_rays_v, Width=num_rays_h)
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                padding_mode="circular"  # Schließt 360°-Rundumblick nahtlos zusammen!
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                padding_mode="circular"
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Größe des CNN-Outputs berechnen
        with torch.no_grad():
            dummy_lidar = torch.zeros(1, 1, self.num_rays_v, self.num_rays_h)
            cnn_out_dim = self.cnn(dummy_lidar).shape[1]

        # 2. Skalare Features (Attitude + Target Deltas)
        attitude_dim = observation_space["attitude"].shape[0]

        # Berechnung der Target-Deltas Dim (falls 2D/3D Vektor)
        target_space = observation_space["target_deltas"]
        if isinstance(target_space, spaces.Box):
            target_dim = target_space.shape[0] * target_space.shape[1]
        else:
            target_dim = target_space.shape[0]

        total_concat_dim = cnn_out_dim + attitude_dim + target_dim

        # 3. Gemeinsamer MLP-Head zur Ausgabe der finale Feature-Größe
        self.linear_head = nn.Sequential(
            nn.Linear(total_concat_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # A) LiDAR Daten in 2D Tensor umformen: (B, C=1, H=num_rays_v, W=num_rays_h)
        lidar_flat = observations["lidar"]
        batch_size = lidar_flat.shape[0]
        lidar_2d = lidar_flat.view(batch_size, 1, self.num_rays_v, self.num_rays_h)

        # B) CNN durchlaufen
        cnn_feats = self.cnn(lidar_2d)

        # C) Skalare Daten flachklopfen
        attitude = observations["attitude"]
        target_deltas = observations["target_deltas"].reshape(batch_size, -1)

        # D) Late Fusion (Zusammenfügen)
        combined = torch.cat([cnn_feats, attitude, target_deltas], dim=1)

        # E) Auf Ziel-Dimension projizieren
        return self.linear_head(combined)