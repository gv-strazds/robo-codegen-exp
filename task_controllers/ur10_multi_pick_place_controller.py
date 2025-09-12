from typing import List

from isaacsim.core.utils.types import ArticulationAction  # noqa: F401 (kept for future overrides)
from isaacsim.robot.manipulators.controllers import (
    StackingController as BaseStackingController,
)
from isaacsim.robot.manipulators.controllers.pick_place_controller import PickPlaceController


class UR10MultiPickPlaceController(BaseStackingController):
    """UR10-specific thin wrapper around the base StackingController.

    Currently identical in behavior to the base class. This class exists so we
    can customize task-specific picking/placing logic without modifying the
    reference controller implementation.
    """

    def __init__(
        self,
        name: str,
        pick_place_controller: PickPlaceController,
        picking_order_cube_names: List[str],
        robot_observation_name: str,
    ) -> None:
        super().__init__(
            name=name,
            pick_place_controller=pick_place_controller,
            picking_order_cube_names=picking_order_cube_names,
            robot_observation_name=robot_observation_name,
        )

