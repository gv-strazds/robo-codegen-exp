from typing import List, Optional

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
        # Track whether we've exhausted target objects
        self._targets_exhausted: bool = False

    def forward(self, observations: dict, end_effector_orientation=None, end_effector_offset=None) -> ArticulationAction:
        """Step the controller.

        If the current pick object lacks a corresponding target object ("target_name" is
        missing or None in observations), halt pick/place by returning a no-op action.
        """
        # If we've run out of targets, return a no-op immediately
        if self._targets_exhausted:
            target_joint_positions = [
                None
            ] * observations[self._robot_observation_name]["joint_positions"].shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        # If already completed, defer to base behavior (no-op action)
        if self._current_cube >= len(self._picking_order_cube_names):
            target_joint_positions = [
                None
            ] * observations[self._robot_observation_name]["joint_positions"].shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        # Check whether the current pick object has an associated target
        current_pick_name = self._picking_order_cube_names[self._current_cube]
        pick_obs = observations.get(current_pick_name, {})
        has_target = pick_obs.get("target_name") not in (None, "")

        if not has_target:
            # No corresponding target for the current pick object: mark done due to
            # exhausted targets and idle (no-op). Do not advance to next pick.
            if not self._targets_exhausted:
                # Log once when transitioning to exhausted state
                try:
                    import carb
                    carb.log_info(
                        f"UR10MultiPickPlaceController: No target for '{current_pick_name}'. Marking done."
                    )
                except Exception:
                    pass
            self._targets_exhausted = True
            self._pick_place_controller.reset()
            target_joint_positions = [
                None
            ] * observations[self._robot_observation_name]["joint_positions"].shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        # Otherwise, use the base stacking behavior
        return super().forward(
            observations=observations,
            end_effector_orientation=end_effector_orientation,
            end_effector_offset=end_effector_offset,
        )

    def reset(self, picking_order_cube_names: Optional[List[str]] = None) -> None:
        """Reset controller state and clear target exhaustion flag."""
        super().reset(picking_order_cube_names=picking_order_cube_names)
        self._targets_exhausted = False
        return

    def is_done(self) -> bool:
        """Return True if either picks are finished or targets are exhausted."""
        return self._targets_exhausted or super().is_done()
