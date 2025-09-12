"""Local task controllers used by this repository.

Exposes UR10MultiPickPlaceController, a thin subclass of the base
StackingController for future customization.
"""

from .ur10_multi_pick_place_controller import UR10MultiPickPlaceController

__all__ = ["UR10MultiPickPlaceController"]

