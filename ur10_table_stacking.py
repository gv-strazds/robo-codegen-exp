# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np
from typing import Optional
import random

def main() -> None:
    # Defer Isaac imports until after SimulationApp is created
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    import carb
    import omni.log

    from isaacsim.cortex.framework.cortex_utils import get_assets_root_path_or_die

    from isaacsim.core.api import World
    from isaacsim.core.api.scenes.scene import Scene
    from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
    import isaacsim.robot.manipulators.controllers as manipulators_controllers
    # from isaacsim.robot.manipulators.examples.universal_robots.controllers import StackingController
    from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import (
        PickPlaceController,
    )
    from isaacsim.robot.manipulators.grippers import SurfaceGripper
    from isaacsim.core.prims import SingleArticulation

    from table_setup import (
        setup_two_tables,
        BIN_X_COORD,
        BIN_Y_COORD,
        BIN_Z_COORD,
        BIN_SIZE,
        TABLETOP_Z_COORD,
        DROPZONE_X, DROPZONE_Y, DROPZONE_Z,
    )
    from stacking_task import UR10MultiPickPlace

    DROPZONE_GRID_WIDTH = 3
    DROPZONE_GRID_HEIGHT = 4
    DROPZONE_GRID_DEPTH = 4

    GRID_DX = -0.15
    x_shift = 0.05 - 0.2
    GRID_DY = 0.15
    z_shift = 0.06
    GRID_DZ = 0.135

    DROPZONE_GRID_XS = [DROPZONE_X+i*GRID_DX+x_shift for i in range(DROPZONE_GRID_WIDTH)]  # [1.00, 0.79, 0.58])   
    DROPZONE_GRID_YS = [DROPZONE_Y+(i*GRID_DY) for i in range(DROPZONE_GRID_HEIGHT)]  #[-0.62, -0.31, 0]
    h = GRID_DZ
    DROPZONE_GRID_ZS = [DROPZONE_Z + ((i) * h) + h / 2 + z_shift for i in range(DROPZONE_GRID_DEPTH)]
    BLOCK_SIZE=0.0515

    class TableTask2(UR10MultiPickPlace):
        """Task using UR10 robot to pick-place multiple cubes.

        Args:
            name (str, optional): Task name identifier. Should be unique if added to the World.
        """

        def __init__(
            self,
            task_name: str = "table_task_2",
            initial_positions=None,
            initial_orientations=None,
            target_positions=None,
            obj_size: Optional[np.ndarray] = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units(),
            stack_target_position: Optional[np.ndarray] = None,
            offset: Optional[np.ndarray] = None,
        ) -> None:
            super().__init__(
                task_name=task_name,
                initial_positions=initial_positions,
                initial_orientations=initial_orientations,
                stack_target_position=stack_target_position,
                obj_size=obj_size,
                offset=offset,
            )

            self.target_asset_type="cube"
            self.target_colors=["blue"]
            self._target_positions = [[x, y, DROPZONE_Z+0.001+BLOCK_SIZE/2] for y in DROPZONE_GRID_YS for x in DROPZONE_GRID_XS]
            # Explicit target scale separate from source object size
            self._target_scale = np.array([BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE]) / get_stage_units()
            self._assets_root_path = get_assets_root_path_or_die()
            if self._assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
                return
            omni.log.warn(f"TableTask init stack_target_position={self._stack_target_position}")
            return


        def setup_workspace(self, scene: Scene) -> None:
            setup_two_tables(scene, self._assets_root_path)

    class TableTask3(UR10MultiPickPlace):
        """Task using UR10 robot to pick-place multiple cubes.

        Args:
            name (str, optional): Task name identifier. Should be unique if added to the World.
        """

        def __init__(
            self,
            task_name: str = "table_task_3",
            initial_positions=None,
            initial_orientations=None,
            obj_size: Optional[np.ndarray] = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units(),
            stack_target_position: Optional[np.ndarray] = None,
            offset: Optional[np.ndarray] = None,
        ) -> None:
            super().__init__(
                task_name=task_name,
                initial_positions=initial_positions,
                initial_orientations=initial_orientations,
                stack_target_position=stack_target_position,
                obj_size=obj_size,
                offset=offset,
            )

            # Targets: discs arranged in the same DROPZONE grid
            self.target_asset_type = "disc"
            self.target_colors = ["purple", "cyan", "black", "yellow"]
            self._target_positions = [[x, y, DROPZONE_Z + 0.001 + BLOCK_SIZE / 2] for y in DROPZONE_GRID_YS for x in DROPZONE_GRID_XS]
            self._target_scale = np.array([BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE]) / get_stage_units()

            self._assets_root_path = get_assets_root_path_or_die()
            if self._assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
                return
            omni.log.warn(f"TableTask init stack_target_position={self._stack_target_position}")
            return

        def setup_workspace(self, scene: Scene) -> None:
            setup_two_tables(scene, self._assets_root_path)


    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Choose the task to run.")
    parser.add_argument(
        "--task",
        choices=["TableTask2", "TableTask3"],
        default="TableTask3",
        help="Specify the task to run: TableTask2 or TableTask3.",
    )
    args = parser.parse_args()

    cube_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()

    my_world = World(stage_units_in_meters=1.0)
    # Choose the task based on the command-line argument
    if args.task == "TableTask2":
        cube_initial_positions = (
            np.array([[0.4, 0.3 + i*(cube_size[1] + 0.01), cube_size[2]/2] for i in range(7)]) / get_stage_units()
        )
        my_task = TableTask2(
            initial_positions=cube_initial_positions,
            obj_size=cube_size,
        )
    else:
        # Define bin constants
        bin_width = BIN_SIZE[0]
        bin_height = BIN_SIZE[1]

        min_x = BIN_X_COORD - bin_width / 2 + 0.05
        max_x = BIN_X_COORD + bin_width / 2 - 0.05
        min_y = BIN_Y_COORD - bin_height / 2 + 0.05
        max_y = BIN_Y_COORD + bin_height / 2 - 0.05

        # Create a 3x3 grid of cube positions
        x_coords = np.linspace(min_x + cube_size[0], max_x - cube_size[0], 3)
        y_coords = np.linspace(min_y + cube_size[1], max_y - cube_size[1], 3)

        CUBE_POS_Z = cube_size[2]/2 + 0.025  # Start a bit higher, above the floor of the picking bin
        _cube_initial_positions = [
            [x, y, TABLETOP_Z_COORD + CUBE_POS_Z] for x in x_coords for y in y_coords]

        my_task = TableTask3(
            initial_positions=np.array(_cube_initial_positions),
            obj_size=cube_size,
        )

    my_world.add_task(my_task)
    my_world.reset()

    reset_needed = False
    while simulation_app.is_running():
        my_world.step(render=True)  # invokes Task.pre_step() on all tasks, then Simulation.step()
        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                # my_controller.reset()
                reset_needed = False
            current_tasks = my_world.get_current_tasks()
            for task_name in current_tasks:
                task = current_tasks[task_name]
                if hasattr(task,"task_step"):
                    task.task_step()
            # The following has been moved into UR10MultiPickPlace.task_step()
            # observations = my_world.get_observations()  #merges observations from all currently running tasks
            # actions = my_controller.forward(
            #     observations=observations, end_effector_offset=np.array([0.0, 0.0, 0.02])
            # )
            # articulation_controller.apply_action(actions)

    simulation_app.close()


if __name__ == "__main__":
    main()
