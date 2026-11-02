from collections.abc import Sequence
from typing import List, Tuple

import numpy as np

from uav_nav_obstacle_avoidance_rl import config

logger = config.logger


class OccupancyGrid:
    """3d grid for discretizing the XYZ space and placing obstacles"""

    def __init__(self, grid_sizes: Sequence[float], cell_size: float, rng):
        """
        initialize grid

        args:
            grid_sizes: the size of each grid dimension in meter (x_size, y_size)
            cell_size: size of each cell in meter
            rng: np random generator
        """
        self.x_size, self.y_size, self.z_size = grid_sizes  # Jetzt 3 Dimensionen (x, y, z)
        self.rng = rng
        self.cell_size = cell_size

        self.x_min, self.x_max = -self.x_size / 2, self.x_size / 2
        self.y_min, self.y_max = -self.y_size / 2, self.y_size / 2
        self.z_min, self.z_max = 0.0, self.z_size  # Z geht von 0 bis z_size

        # Grid-Dimensionen (anzahl Voxel pro Achse)
        self.nx = int(np.floor(self.x_size / self.cell_size))
        self.ny = int(np.floor(self.y_size / self.cell_size))
        self.nz = int(np.floor(self.z_size / self.cell_size))

        # 3D Occupancy Grid (nx, ny, nz)
        self.grid = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """convert world XY coordinates into grid cell indices"""
        i = int(np.floor((x - self.x_min) / self.cell_size))
        j = int(np.floor((y - self.y_min) / self.cell_size))
        return i, j

    def cell_to_world(self, cell_idx: Tuple[int, int]) -> Tuple[float, float]:
        """convert cell index to world XY coordinates (center of cell)"""
        i, j = cell_idx
        x = self.x_min + (i + 0.5) * self.cell_size
        y = self.y_min + (j + 0.5) * self.cell_size
        return x, y

    def are_cells_free(self, cell_indices: List[Tuple]) -> List[bool]:
        """
        check if cells are free
        supports (i, j) as well as (i, j, k)
        """
        result = []
        for cell in cell_indices:
            if len(cell) == 3:
                i, j, k = cell
                if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
                    result.append(bool(self.grid[i, j, k] == 0.0))
                else:
                    result.append(False)
            elif len(cell) == 2:
                i, j = cell
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    result.append(bool(np.all(self.grid[i, j, :] == 0.0)))
                else:
                    result.append(False)
        return result

    def get_occupancy(self):
        """
        Return:
            grid of current occupancy status of all cells in 2D: ndarray
        """
        return self.grid

    def get_random_free_cell(self) -> Tuple[int, int]:
        """get a random free grid cell"""
        free_cells = np.argwhere(self.grid == 0.0)
        if len(free_cells) == 0:
            logger.error("No free cells available!")
            raise ValueError("No free cells available!")
        idx = self.rng.choice(len(free_cells))
        return tuple(free_cells[idx])

    def get_random_free_position(self) -> Tuple[float, float]:
        """get random free XY world position"""
        cell_idx = self.get_random_free_cell()
        return self.cell_to_world(cell_idx)

    def mark_cells(self, cell_indices: List[Tuple], occupied: bool):
        """
        mark cells as occupied (1.0) or free (0.0)
        unterstuetzt sowohl (i, j) als auch (i, j, k)
        """
        for cell in cell_indices:
            if len(cell) == 3:
                i, j, k = cell
                if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
                    self.grid[i, j, k] = 1.0 if occupied else 0.0
            elif len(cell) == 2:
                i, j = cell
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    self.grid[i, j, :] = 1.0 if occupied else 0.0

    def reset_occupancy(self):
        """reset all cells to free (unoccupied) state"""
        self.grid.fill(False)

    def world_to_cell_3d(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """convert world XYZ coordinates into 3D grid cell indices"""
        i = int(np.floor((x - self.x_min) / self.cell_size))
        j = int(np.floor((y - self.y_min) / self.cell_size))
        k = int(np.floor((z - self.z_min) / self.cell_size))
        return i, j, k
