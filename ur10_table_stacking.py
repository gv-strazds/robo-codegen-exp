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
        from tasks.table_task2 import TableTask2

        my_task = TableTask2(obj_size=cube_size)
    else:
        from tasks.table_task3 import TableTask3

        my_task = TableTask3(obj_size=cube_size)

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
