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

# from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import random
import isaacsim.core.api.tasks as tasks  # tasks.BaseTask, tasks.PickPlace, 
import isaacsim.robot.manipulators.controllers as manipulators_controllers
# from isaacsim.core.api.tasks import BaseTask

from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.robot.manipulators.examples.universal_robots import UR10
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid, DynamicCylinder
from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import (
    PickPlaceController,
)
from asset_utils import add_prim_asset


class UR10MultiPickPlace(tasks.BaseTask):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to "ur10_stacking".
        target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
        obj_size (Optional[np.ndarray], optional): [description]. Defaults to None.
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        task_name: str = "ur10_stacking",
        initial_positions=None,
        initial_orientations=None,
        stack_target_position: Optional[np.ndarray] = None,
        obj_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        # BEGIN --- isaacsim.core.api.tasks.Stacking (BaseStacking) .__init__
        super().__init__(name=task_name, offset=offset) #super=isaacsim.core.api.tasks.BaseTask
        self._robot = None
        self._num_of_pick_objs = initial_positions.shape[0]
        self._initial_positions = initial_positions
        self._initial_orientations = initial_orientations
        if self._initial_orientations is None:
            self._initial_orientations = [None] * self._num_of_pick_objs
        self._stack_target_position = stack_target_position
        self._obj_size = obj_size
        if self._obj_size is None:
            self._obj_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
        if stack_target_position is not None:
            self._stack_target_position = self._stack_target_position + self._offset
        self._pick_objs = []
        self._target_objs = []
        self._num_of_target_objs = 0
        # END --- isaacsim.core.api.tasks.Stacking (BaseStacking) .__init__
        self._ur10_asset_path = "/home/gstrazds/workspaces/sim_experiments/SimEnvs/" \
            "Collected_ur10_bin_filling/ur10_bin_filling.usd"
        return

    def set_up_scene(self, scene: Scene) -> None:
        """Loads the stage USD and adds the robot and task objects to the World's scene.

        Args:
            scene (Scene): The world's scene.
        """
        # super().set_up_scene(scene)
        self._scene = scene  # isaacsim/core/api/tasks/base_task.py

        # BEGIN --- isaacsim.core.api.tasks.Stacking (BaseStacking)
        # INCLUDED in ur10_table_scene .usd  #scene.add_default_ground_plane() z_position=-0.5)
        self._robot = self.set_robot()
        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        # END --- isaacsim.core.api.tasks.Stacking (BaseStacking)
        self.setup_workspace(scene)
        self.add_source_objects(scene)
        self.add_target_objects(scene)
        self._move_task_objects_to_their_frame()

    def add_source_objects(self, scene: Scene) -> None:
        """Add source (pickable) objects to the scene.

        - Uses ``asset_utils.add_prim_asset`` for object creation.
        - Honors optional ``self.source_asset_type`` (defaults to "cube").
        - If ``self.source_colors`` is provided, randomly selects from it; otherwise random RGB.
        - Names objects using the selected ``asset_type`` as the base (e.g., "disc_01").
        """
        asset_type = getattr(self, "source_asset_type", "cube")
        source_colors = getattr(self, "source_colors", None)
        for i in range(self._num_of_pick_objs):
            if source_colors is None:
                color = np.random.uniform(size=(3,))
            else:
                color = random.choice(source_colors)
            obj_name = find_unique_string_name(
                initial_name=asset_type, is_unique_fn=lambda x: not self.scene.object_exists(x)
            )
            prim = add_prim_asset(
                scene,
                asset_type=asset_type,
                obj_name=obj_name,
                position=self._initial_positions[i],
                orientation=self._initial_orientations[i],
                scale=self._obj_size,
                scene_path_root="/World/",
                color=color,
            )
            self._pick_objs.append(prim)
            self._task_objects[prim.name] = prim

    def add_target_objects(self, scene: Scene) -> None:
        """Add target/drop-off objects to the scene.

        Subclasses should define:
        - self._target_positions: iterable of [x, y, z]
        - self.target_asset_type: string in asset_utils.PRIMS_MAP (e.g., "cube", "disc")
        - self.target_colors: list of color names or RGB triples

        Uses self._obj_size for scale unless self._target_scale exists.
        Populates self._target_objs and self._task_objects; updates count.
        """
        if not hasattr(self, "_target_positions") or self._target_positions is None:
            return
        asset_type = getattr(self, "target_asset_type", "cube")
        colors = getattr(self, "target_colors", ["blue"])  # defaults
        target_scale = getattr(self, "_target_scale", self._obj_size)
        self._target_objs = []
        for i, target_pos in enumerate(self._target_positions):
            block_name = f"target_{i+1}"
            prim = add_prim_asset(
                scene,
                asset_type=asset_type,
                obj_name=block_name,
                prim_path="/World/" + block_name,
                position=np.array(target_pos),
                orientation=None,
                scale=target_scale,
                color=random.choice(colors),
            )
            self._target_objs.append(prim)
            self._task_objects[prim.name] = prim
        self._num_of_target_objs = len(self._target_objs)

    def set_robot(self) -> UR10:
        """[summary]

        Returns:
            UR10: [description]
        """

        # BEGIN --- isaacsim.robot.manipulators.examples.universal_robots.tasks.Stacking
        # ur10_prim_path = find_unique_string_name(
        #     initial_name="/World/Scene/ur10", is_unique_fn=lambda x: not is_prim_path_valid(x)
        # )
        ur10_robot_name = find_unique_string_name(
            initial_name="my_ur10", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        #MODIFIED:
        # self._ur10_robot = UR10(prim_path=ur10_prim_path, name=ur10_robot_name, attach_gripper=True)
        add_reference_to_stage(usd_path=self._ur10_asset_path, prim_path="/World/Scene")
        self._ur10_robot = UR10(prim_path="/World/Scene/ur10", name=ur10_robot_name, attach_gripper=True)
        #---- end-MODIFIED

        self._ur10_robot.set_joints_default_state(
            positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        )
        # END --- isaacsim.robot.manipulators.examples.universal_robots.tasks.Stacking
        return self._ur10_robot

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """[summary]

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        # BEGIN --- isaacsim.robot.manipulators.examples.universal_robots.tasks.Stacking
        # super() <-> isaacsim.core.api.tasks.Stacking(=BaseStacking)
        super().pre_step(time_step_index=time_step_index, simulation_time=simulation_time) #does nothing
        self._ur10_robot.gripper.update()
        # END --- isaacsim.robot.manipulators.examples.universal_robots.tasks.Stacking
        return

    # BEGIN ---  merge base class methods: isaacsim.core.api.tasks.Stacking(=BaseStacking)
    def set_params(
        self,
        obj_name: Optional[str] = None,
        obj_position: Optional[str] = None,
        obj_orientation: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[str] = None,
        target_orientation: Optional[str] = None,
        stack_target_position: Optional[str] = None,
    ) -> None:
        """[summary]

        Args:
            obj_name (Optional[str], optional): [description]. Defaults to None.
            obj_position (Optional[str], optional): [description]. Defaults to None.
            obj_orientation (Optional[str], optional): [description]. Defaults to None.
            stack_target_position (Optional[str], optional): [description]. Defaults to None.
        """
        if stack_target_position is not None:
            self._stack_target_position = stack_target_position
        if obj_name is not None:
            self._task_objects[obj_name].set_local_pose(position=obj_position, orientation=obj_orientation)
        if target_name is not None:
            self._task_objects[target_name].set_local_pose(position=target_position, orientation=target_orientation)
        return

    def get_params(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        params_representation = dict()
        params_representation["stack_target_position"] = {"value": self._stack_target_position, "modifiable": True}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation

    def post_reset(self) -> None:
        """[summary]"""
        # NOTE: if using SurfaceGripper, the following code (from core.api.tasks.Stacking) does nothing
        from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
        if isinstance(self._robot.gripper, ParallelGripper):
            self._robot.gripper.set_joint_positions(self._robot.gripper.joint_opened_positions)
        my_ur10 = self._robot

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
        self._task_controller = manipulators_controllers.StackingController(
            name=STACKING_CONTROLLER_NAME,
            pick_place_controller=pick_place_controller,
            picking_order_cube_names=self.get_obj_names(),
            robot_observation_name=self._robot.name,
        )
        self._articulation_controller = self._robot.get_articulation_controller()
        self._task_controller.reset()
        return

    def task_step(self):
        observations = self.get_observations()
        actions = self._task_controller.forward(
            observations=observations, end_effector_offset=np.array([0.0, 0.0, 0.02])
        )
        self._articulation_controller.apply_action(actions)
        return

    def get_obj_names(self) -> List[str]:
        """[summary]

        Returns:
            List[str]: [description]
        """
        obj_names = []
        for i in range(self._num_of_pick_objs):
            obj_names.append(self._pick_objs[i].name)
        return obj_names

    def get_target_names(self) -> List[str]:
        """[summary]

        Returns:
            List[str]: [description]
        """
        target_names = []
        for i in range(self._num_of_target_objs):
            target_names.append(self._target_objs[i].name)
        return target_names

    def calculate_metrics(self) -> dict:
        """[summary]

        Raises:
            NotImplementedError: [description]

        Returns:
            dict: [description]
        """
        raise NotImplementedError

    def is_done(self) -> bool:
        """[summary]

        Raises:
            NotImplementedError: [description]

        Returns:
            bool: [description]
        """
        raise NotImplementedError
    # END ---  merge base class methods: isaacsim.core.api.tasks.Stacking

    # MODIFIED from base class(isaacsim.core.api.tasks.Stacking) method 
    def get_observations(self) -> dict:  
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()
        observations = {
            self._robot.name: {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
            }
        }
        for i in range(self._num_of_pick_objs):
            obj_name = self._pick_objs[i].name
            obj_position, obj_orientation = self._pick_objs[i].get_local_pose()
            target_obj = self._scene.get_object(f"target_{i+1}")
            if target_obj:
                target_name = target_obj.name
                target_obj_pos, target_obj_orientation = target_obj.get_world_pose()
                target_position = np.array(
                    [
                        target_obj_pos[0], #self._stack_target_position[0],
                        target_obj_pos[1], # self._stack_target_position[1],
                        self._obj_size[2] * 1.5 # (obj_size_z + obj_size_z/2)
                    ]
                )
            else:
                target_name = None
                target_orientation = None
                if self._stack_target_position is not None:
                    target_position = np.array(
                        [
                            self._stack_target_position[0],
                            self._stack_target_position[1],
                            (self._obj_size[2] * i) + self._obj_size[2] / 2.0,
                        ]
                    )
                else:
                    target_position = np.array([0.3, 0.3, 0]) / get_stage_units()
            observations[self._pick_objs[i].name] = {
                "position": obj_position,
                "orientation": obj_orientation,
                "target_name": target_name,
                "target_position": target_position,
                "target_orientation": target_obj_orientation,
            }
        return observations
