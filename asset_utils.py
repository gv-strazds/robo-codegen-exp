import numpy as np
from colors import *

import omni.log
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid, DynamicCylinder, DynamicSphere
# from isaacsim.core.api.objects import VisualCapsule, VisualSphere
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils import (  # noqa E402
    extensions,
    prims,
    rotations,
    stage,
    viewports,
    # transformations
)

PRIMS_MAP = {
    "cube": DynamicCuboid,
    "disc": DynamicCylinder,
    "cylinder": DynamicCylinder,
    "rect": FixedCuboid,
    "marker": VisualCuboid,
    "ball": DynamicSphere,
}
def add_usd_asset(scene,
                  asset_path,
                  obj_name=None,
                  position=None,
                  orientation=None,
                  scale=None,
                  assets_root_path=None,
                  prim_path=None,
                  scene_path_root="/"
                  ):
    if not prim_path:
        assert obj_name is not None
        prim_scene_path = f"{scene_path_root}{obj_name}"
    elif not prim_path.startswith("/"):
        prim_scene_path = f"{scene_path_root}{prim_path}"
    else:
        prim_scene_path = prim_path
    if not obj_name:
        obj_name = prim_scene_path.split("/")[-1]
    if assets_root_path and not asset_path.startswith(assets_root_path):
        abs_file_path = assets_root_path + asset_path
    else:
        abs_file_path = asset_path
    obj_prim = prims.create_prim(
            prim_scene_path,
            "Xform",
            position=position,
            orientation=orientation,
            scale=scale,
            usd_path=abs_file_path
        )
    xform_prim = SingleXFormPrim(prim_scene_path, name=obj_name)
    omni.log.warn(f"add_usd_asset: {abs_file_path} XFormPrim.name={xform_prim.name} {prim_scene_path}")
    scene.add(xform_prim)
    return xform_prim

def add_prim_asset(scene,
                   asset_type="cube",
                   obj_name=None,
                   position=None,
                   orientation=None,
                   scale=None,
                   prim_path=None,
                   scene_path_root="/",
                   color="blue",
                   ):
    if obj_name is None:
        obj_name = find_unique_string_name(
            initial_name=asset_type, is_unique_fn=lambda x: not scene.object_exists(x)
        )
    if not prim_path:
           prim_scene_path = f"{scene_path_root}{obj_name}"
    elif not prim_path.startswith("/"):
        prim_scene_path = f"{scene_path_root}{prim_path}"
    else:
        prim_scene_path = prim_path

    obj_prim_path = find_unique_string_name(
        initial_name=prim_scene_path, is_unique_fn=lambda x: not is_prim_path_valid(x)
    )
    # Accept both named colors and explicit RGB triples
    if isinstance(color, (list, tuple, np.ndarray)):
        color_value = np.array(color)
    elif isinstance(color, str) and color.lower() in COLOR_MAP:
        color_value = np.array(COLOR_MAP[color.lower()].value)
    else:
        color_value = np.array(BasicColor.RED.value)
    # omni.log.warn(f"add_prim_asset: {asset_type} {obj_name} {obj_prim_path}")
    prim = scene.add(
            PRIMS_MAP[asset_type](
                name=obj_name,
                position=position,
                orientation=orientation,
                prim_path=obj_prim_path,
                scale=scale,
                # size=1.0,
                color=color_value
            )
        )
    return prim
