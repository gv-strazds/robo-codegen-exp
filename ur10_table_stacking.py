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
    )
    from stacking_task import UR10MultiPickPlace

    class TableTask2(UR10MultiPickPlace):
        """Task using UR10 robot to pick-place multiple cubes.

        Args:
            name (str, optional): Task name identifier. Should be unique if added to the World.
        """

        def __init__(
            self,
            name: str = "table_task_2",
            initial_positions=np.array([[0.4, 0.3, 0.03], [0.45, 0.6, 0.03]]) / get_stage_units(),
            initial_orientations=None,
            obj_size: Optional[np.ndarray] = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units(),
            stack_target_position: Optional[np.ndarray] = None,
            offset: Optional[np.ndarray] = None,
        ) -> None:
            super().__init__(
                task_name=name,
                initial_positions=initial_positions,
                initial_orientations=initial_orientations,
                stack_target_position=stack_target_position,
                obj_size=obj_size,
                offset=offset,
            )

            self._packing_bin = None
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
            name: str = "table_task_3",
            initial_positions=np.array([[0.4, 0.3, 0.03], [0.45, 0.6, 0.03]]) / get_stage_units(),
            initial_orientations=None,
            obj_size: Optional[np.ndarray] = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units(),
            stack_target_position: Optional[np.ndarray] = None,
            offset: Optional[np.ndarray] = None,
        ) -> None:
            super().__init__(
                task_name=name,
                initial_positions=initial_positions,
                initial_orientations=initial_orientations,
                stack_target_position=stack_target_position,
                obj_size=obj_size,
                offset=offset,
            )

            self._packing_bin = None
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

    CUBE_SIZE_X = 0.0515
    CUBE_SIZE_Y = 0.0515
    CUBE_SIZE_Z = 0.0515
    CUBE_POS_Z = CUBE_SIZE_Z / 2 + 0.025

    my_world = World(stage_units_in_meters=1.0)
    cube_size = np.array([CUBE_SIZE_X, CUBE_SIZE_Y, CUBE_SIZE_Z]) / get_stage_units()
    cube_initial_positions = (
        np.array([[0.4, 0.3 + i * (CUBE_SIZE_Y + 0.01), CUBE_POS_Z] for i in range(7)])
        / get_stage_units()
    )
    stack_target_position = np.array([0.4, 0.8, cube_size[2] / 2.0])
    stack_target_position[0] = stack_target_position[0] / get_stage_units()
    stack_target_position[1] = stack_target_position[1] / get_stage_units()

    # Define bin constants
    bin_width = BIN_SIZE[0]
    bin_height = BIN_SIZE[1]

    min_x = BIN_X_COORD - bin_width / 2 + 0.05
    max_x = BIN_X_COORD + bin_width / 2 - 0.05
    min_y = BIN_Y_COORD - bin_height / 2 + 0.05
    max_y = BIN_Y_COORD + bin_height / 2 - 0.05

    # Create a 3x3 grid of cube positions
    x_coords = np.linspace(min_x + CUBE_SIZE_X, max_x - CUBE_SIZE_X, 3)
    y_coords = np.linspace(min_y + CUBE_SIZE_Y, max_y - CUBE_SIZE_Y, 3)

    new_cube_initial_positions = []
    for x in x_coords:
        for y in y_coords:
            new_cube_initial_positions.append([x, y, TABLETOP_Z_COORD + CUBE_POS_Z])

    new_cube_initial_positions = np.array(new_cube_initial_positions) / get_stage_units()

    # Choose the task based on the command-line argument
    if args.task == "TableTask2":
        my_task = TableTask2(
            initial_positions=cube_initial_positions,
            obj_size=cube_size,
            stack_target_position=stack_target_position,
        )
    else:
        my_task = TableTask3(
            initial_positions=new_cube_initial_positions,
            obj_size=cube_size,
            stack_target_position=stack_target_position,
        )

    my_world.add_task(my_task)
    my_world.reset()
    robot_name = my_task.get_params()["robot_name"]["value"]
    my_ur10 = my_world.scene.get_object(robot_name)

    STACKING_CONTROLLER_NAME = "ur10_stacking_controller"
    pick_place_controller = PickPlaceController(
        name=STACKING_CONTROLLER_NAME + "_pick_place_controller",
        gripper=my_ur10.gripper,
        robot_articulation=my_ur10,
        events_dt=[
            1.0 / 125,  # Move above obj
            1.0 / 100,  # Down
            1.0 / 10,   # Wait
            1.0 / 4,    # Close gripper
            1.0 / 50,   # Lift
            1.0 / 200,  # Move above target
            1.0 / 100,  # Down
            1.0,        # Release gripper
            1.0 / 50,   # Move up
            1.0 / 2,    # Return towards start
        ],
    )
    my_controller = manipulators_controllers.StackingController(
        name=STACKING_CONTROLLER_NAME,
        pick_place_controller=pick_place_controller,
        picking_order_cube_names=my_task.get_obj_names(),
        robot_observation_name=robot_name,
    )
    articulation_controller = my_ur10.get_articulation_controller()

    reset_needed = False
    while simulation_app.is_running():
        my_world.step(render=True)
        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                my_controller.reset()
                reset_needed = False
            observations = my_world.get_observations()
            actions = my_controller.forward(
                observations=observations, end_effector_offset=np.array([0.0, 0.0, 0.02])
            )
            articulation_controller.apply_action(actions)

    simulation_app.close()


if __name__ == "__main__":
    main()
