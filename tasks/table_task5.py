import numpy as np
from typing import Optional

from stacking_task import UR10MultiPickPlace


class TableTask5(UR10MultiPickPlace):
    """Pick green cubes from the dropzone table and place onto 6 red rectangles arranged in a circle."""

    def __init__(
        self,
        task_name: str = "table_task_5",
        initial_positions=None,
        initial_orientations=None,
        obj_size: Optional[np.ndarray] = None,
        stack_target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        # Lazily import Isaac utilities and scene helpers
        from isaacsim.core.utils.stage import get_stage_units
        from isaacsim.cortex.framework.cortex_utils import get_assets_root_path_or_die
        from table_setup import (
            setup_two_tables,
            generate_target_positions,
            generate_circular_positions,
            DROPZONE_CENTER_POINT,
        )

        resolved_obj_size = obj_size if obj_size is not None else np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()

        # Place pick objects (green cubes) directly on the dropzone table as a compact 3x2 grid (6 cubes)
        if initial_positions is None:
            BLOCK_SIZE = 0.0515
            dz_positions = generate_target_positions(grid_width=3, grid_height=2, block_size=BLOCK_SIZE)
            initial_positions = np.array(dz_positions)

        super().__init__(
            task_name=task_name,
            initial_positions=initial_positions,
            initial_orientations=initial_orientations,
            stack_target_position=stack_target_position,
            obj_size=resolved_obj_size,
            offset=offset,
        )

        BLOCK_SIZE = 0.0515
        # Sources: green cubes
        self.source_asset_type = "cube"
        self.source_colors = ["green"]

        # Targets: 6 red rectangles arranged in a circle on the same table
        self.target_asset_type = "rect"
        self.target_colors = ["red"]
        self._target_positions = generate_circular_positions(
            num_positions=6, radius=0.18, center=DROPZONE_CENTER_POINT, block_size=BLOCK_SIZE
        )
        # Make rectangles wider and thin (on Z)
        self._target_scale = np.array([BLOCK_SIZE * 1.8, BLOCK_SIZE * 0.9, BLOCK_SIZE * 0.02]) / get_stage_units()

        self._assets_root_path = get_assets_root_path_or_die()
        if self._assets_root_path is None:
            import carb
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # Store for workspace setup
        self._setup_two_tables = setup_two_tables

    def setup_workspace(self, scene) -> None:
        self._setup_two_tables(scene, self._assets_root_path)

