import numpy as np
from typing import Optional

from stacking_task import UR10MultiPickPlace


class TableTask3(UR10MultiPickPlace):
    """Task using UR10 robot to pick-place multiple discs."""

    def __init__(
        self,
        task_name: str = "table_task_3",
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
            generate_target_positions,
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
        # Targets: discs arranged in the DROPZONE grid
        self.target_asset_type = "disc"
        self.target_colors = ["purple", "cyan", "black", "yellow"]
        self._target_positions = generate_target_positions(grid_width=3, grid_height=4, block_size=BLOCK_SIZE)
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
        #self.add_source_objects(scene)  #invoked automatically by base class during set_up_scene(), after call to setup_workspace()
        #self.add_target_objects(scene)  #invoked automatically by base class during set_up_scene(), after call to setup_workspace()


