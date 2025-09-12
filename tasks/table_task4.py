import numpy as np
from typing import Optional

from stacking_task import UR10MultiPickPlace


class TableTask4(UR10MultiPickPlace):
    """Task using UR10 robot to stack cubes onto cylinders."""

    def __init__(
        self,
        task_name: str = "table_task_4",
        initial_positions=None,
        initial_orientations=None,
        obj_size: Optional[np.ndarray] = None,
        stack_target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        # Lazily import Isaac utilities to avoid import-order issues
        from isaacsim.core.utils.stage import get_stage_units
        from isaacsim.cortex.framework.cortex_utils import get_assets_root_path_or_die
        from table_setup import (
            setup_two_tables,
            generate_grid_positions,
            generate_circular_positions,
            DROPZONE_CENTER_POINT,
        )

        resolved_obj_size = obj_size if obj_size is not None else np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
        if initial_positions is None:
            initial_positions = generate_grid_positions(obj_size=resolved_obj_size, rows=3, cols=3)

        super().__init__(
            task_name=task_name,
            initial_positions=initial_positions,
            initial_orientations=initial_orientations,
            stack_target_position=stack_target_position,
            obj_size=resolved_obj_size,
            offset=offset,
        )

        BLOCK_SIZE = 0.0515
        # Sources: cubes to be picked
        self.source_asset_type = "cube"
        # Targets: cylinders arranged in a circle
        self.target_asset_type = "rect"
        self.target_colors = ["yellow"]
        
        self._target_positions = generate_circular_positions(
            num_positions=8, radius=0.2, center=DROPZONE_CENTER_POINT, block_size=BLOCK_SIZE
        )
        self._target_scale = np.array([BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE]) / get_stage_units()

        self._assets_root_path = get_assets_root_path_or_die()
        if self._assets_root_path is None:
            import carb
            carb.log_error("Could not find Isaac Sim assets folder")
            return

        # Store for workspace setup
        self._setup_two_tables = setup_two_tables

    def setup_workspace(self, scene) -> None:
        self._setup_two_tables(scene, self._assets_root_path)
