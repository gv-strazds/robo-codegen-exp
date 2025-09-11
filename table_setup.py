import math
import random
import sys
import numpy as np
from typing import Optional

import omni.log

import carb
from collections import namedtuple

from isaacsim.core.api import World
from isaacsim.core.utils.string import find_unique_string_name
# from isaacsim.core.utils.prims import is_prim_path_valid

from isaacsim.core.utils.collisions import ray_cast
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.examples.universal_robots.controllers.stacking_controller import StackingController

from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.robot.manipulators.examples.universal_robots import UR10
from isaacsim.storage.native import get_assets_root_path

# from isaacsim.cortex.framework.cortex_utils import get_assets_root_path_or_die

from isaacsim.core.api.objects import VisualCapsule, VisualSphere
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid, DynamicCylinder
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils import (  # noqa E402
    extensions,
    prims,
    rotations,
    stage,
    viewports,
    # transformations
)
from pxr import Gf, UsdGeom  # noqa E402

GROUND_PLANE_Z_OFFSET = -0.5

UR_COORDS = np.array([0.0, 0.0, 0.0])  # position of the base of the robot

# coordinates of UR Robot (relative to original Scene layout - before shifting everyting with UR robot at origin)
#UR_X_COORD_0 = 0.8 #1.0
#UR_Y_COORD_0 = -0.5 #-0.3
UR_Z_COORD_0 = -GROUND_PLANE_Z_OFFSET # height of the base of the robot above the ground plane


TABLE_THICKNESS = 0.1
TABLETOP_Z_OFFSET = 0.1 #-0.2  # height of the (top surface of the) table upon which the objects are placed, relative to the base of the robot
TABLETOP_Z_COORD = UR_COORDS[2] + TABLETOP_Z_OFFSET
TABLETOP_HEIGHT = TABLETOP_Z_COORD - GROUND_PLANE_Z_OFFSET  # height of the top surface of the table above the ground plane
TABLE_LENGTH = 1.2
TABLE_WIDTH = 0.7
TABLE_SIZE = np.array([TABLE_WIDTH, TABLE_LENGTH, TABLE_THICKNESS])
TABLETOP_CENTER_POINT = np.array([-0.81, 0.33, TABLETOP_Z_COORD]) #[-0.01-UR_X_COORD_0, -0.17-UR_Y_COORD_0, TABLETOP_Z_COORD])  # 
TABLE_COORDS = TABLETOP_CENTER_POINT - [0, 0, TABLE_THICKNESS/2]


BIN_COORDS = TABLETOP_CENTER_POINT + [0.19, 0.27, 0.1]
BIN_X_COORD = BIN_COORDS[0]
BIN_Y_COORD = BIN_COORDS[1]
BIN_Z_COORD = BIN_COORDS[2] #
BIN_SCALE = [1.5, 1.5, 0.5]
BIN_SIZE = [ 0.35, 0.5, 0.05 ] # not correct, just a placeholder for now

Region2D = namedtuple('Region2D', ['min_x', 'max_x', 'min_y', 'max_y'])
PICK_REGION = Region2D(BIN_X_COORD-BIN_SIZE[0]/2, BIN_X_COORD+BIN_SIZE[0]/2,
                       BIN_Y_COORD-BIN_SIZE[1]/2, BIN_Y_COORD+BIN_SIZE[1]/2)

def is_in_pick_region(x, y, z):
    return z >= TABLETOP_Z_COORD and \
           (PICK_REGION.min_y-0.3 < y < PICK_REGION.max_y+0.1) and \
           (PICK_REGION.min_x-0.05 < x < PICK_REGION.max_x+0.2)

DROPZONE_X = 1.00-0.6
DROPZONE_Y = -0.62+1.0
DROPZONE_Z = 0 # -0.59374
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

cylinder_specs = [
([-0.05+BIN_X_COORD, BIN_Y_COORD+0.0, BIN_Z_COORD+0.44], [0.0, 75.8, 0.0]),
([-0.01+BIN_X_COORD, BIN_Y_COORD+0.065, BIN_Z_COORD+0.50], [0.0, 75.8, 0.0]),
([-0.05+BIN_X_COORD, BIN_Y_COORD+0.09, BIN_Z_COORD+0.48], [0.0, 75.8, 0.0]),
([-0.02+BIN_X_COORD, BIN_Y_COORD+0.065, BIN_Z_COORD+0.57], [0.0, 75.8, 39.0]),
]

def random_spawn_transform(spawn_region: Region2D, z_baseline=BIN_Z_COORD):
    _SPAWN_MIN_Z = 0.25   # 1.0
    _SPAWN_MAX_Z = 0.55  # 1.5
    x = random.uniform(spawn_region.min_x+0.2, spawn_region.max_x-0.2)
    y = random.uniform(spawn_region.min_y+0.2, spawn_region.max_y-0.2)
    z = random.uniform(z_baseline + _SPAWN_MIN_Z, z_baseline + _SPAWN_MAX_Z)  # high enough to be out of the way
    position = np.array([x, y, z])
    # position = np.array([0.3, 0.3, 0.3]) / get_stage_units()
    # jj = random.random() * 0.02 - 0.01
    w = (random.random()-0.5) * 1.5
    # norm = np.sqrt(jj**2 + w**2)
    # quat = math_util.Quaternion([w / norm, 0, jj / norm, 0]).vals
    # quat = math_util.Quaternion([1.0, 0, 0, 0]).vals
    quat = rotations.euler_angles_to_quat(np.array([0.0, math.pi/2+w, 0.0]))
    if False and random.random() > 0.5:
        print("<flip>")
        # flip the bottle so it's upside down
        quat = quat * math_util.Quaternion([0, 0, 1, 0]).vals
    else:
        print("<no flip>")

    return position, quat


def setup_two_tables(scene:Scene, assets_root_path=None, standard_objs=True, add_bin=True) -> None:
    if assets_root_path is None:
        assets_root_path = get_assets_root_path()
    # GroundPlane(prim_path="/World/groundPlane", size=3, color=np.array([0.1, 0.15, 0.25]))

    viewports.set_camera_view(
        eye=np.array([0.5, 2.0, 1.2]),
        target=TABLETOP_CENTER_POINT+[0.5, 0, 0])

    table_1 = FixedCuboid(prim_path="/World/table_2", name="table_1",
        position=TABLE_COORDS,
        scale=TABLE_SIZE,
        color=np.array([0.2, 0.3, 0.]))

    _support_height = TABLETOP_HEIGHT-TABLE_THICKNESS/2
    table_support = VisualCuboid(prim_path="/World/table_support",
        position=TABLE_COORDS - [0, 0, _support_height/2],  # position is the cuboid center point
        scale=np.array([.3, .3, _support_height]),
        color=np.array([.2, .3, 0.]))

    drop_zone = FixedCuboid(prim_path="/World/work_zone", name="table_1",
        position=np.array([DROPZONE_X+.05-0.2-(2*0.21)/2, DROPZONE_Y+(2*0.31)/2, DROPZONE_Z-TABLE_THICKNESS/2]),
        scale=np.array([0.21*3+0.3, (0.31*3), TABLE_THICKNESS]),
        color=np.array([.2, 0, .3]))

    # add a stand for the UR robot so that it doesn't appear to be hovering above the ground plane
    stand_prim = prims.create_prim(
        "/World/UR_mount",
        "Xform",
        position=UR_COORDS,
        # orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
        scale=np.array([1.0, 1.0, UR_Z_COORD_0]),
        usd_path=assets_root_path
        + "/Isaac/Props/Mounts/ur10_mount.usd",
    )
    # # Attach Rigid Body and Collision Preset
    # rigid_api = UsdPhysics.RigidBodyAPI.Apply(stand_prim)
    # rigid_api.CreateRigidBodyEnabledAttr(True)
    # UsdPhysics.CollisionAPI.Apply(stand_prim)

    n = 0
    for y in DROPZONE_GRID_YS:
        for x in DROPZONE_GRID_XS:
            n += 1
            block_name=f"target_{n}"
            BLOCK_SIZE=0.0515
            scene.add(
                DynamicCuboid(
                    name=block_name,
                    position=np.array([x, y, DROPZONE_Z+0.001+BLOCK_SIZE/2]),
                    # orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), -90)),
                    prim_path="/World/"+block_name,
                    scale=np.array([BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE]) / get_stage_units(),
                    size=1.0,
                    color=np.array([0, 0, 1]),
                )
            )
            # prim_pad = prims.create_prim(
            #     prim_path="/World/"+block_name,
            #     prim_type="Xform",
            #     position=np.array([x, y, DROPZONE_Z+0.001]),
            #     orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 0, 1), -90)),
            #     scale=np.array([1.0, 1.0, 1.0]),
            #     # usd_path="/home/gstrazds/workspaces/sim_experiments/SimEnvs/assets/madara_pad.usd",
            #     usd_path="/home/gstrazds/workspaces/sim_experiments/SimEnvs/assets/pad_v3.usd",
            # )
    if standard_objs:
        # add some objects on the table
        add_usd_asset(
            scene,
            asset_path="/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd", 
            obj_name="cracker_box",
            position=TABLETOP_CENTER_POINT+[-0.19, -0.08, 0.15], #np.array([-0.2-UR_X_COORD_0, -0.25-UR_Y_COORD_0, TABLETOP_Z_COORD+0.15]),
            orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
            assets_root_path=assets_root_path
        )
        add_usd_asset(
            scene,
            asset_path="/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            obj_name="sugar_box",
            position=TABLETOP_CENTER_POINT+[-0.06, -0.08, 0.01], #np.array([-0.07-UR_X_COORD_0, -0.25-UR_Y_COORD_0, TABLETOP_Z_COORD+0.1]),
            orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(0, 1, 0), -90)),
            assets_root_path=assets_root_path
        )
        add_usd_asset(
            scene,
            asset_path="/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            obj_name="soup_can",
            position=TABLETOP_CENTER_POINT+[0.11, -0.08, 0.10], #np.array([0.1-UR_X_COORD_0, -0.25-UR_Y_COORD_0, TABLETOP_Z_COORD+0.10]),
            orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
            assets_root_path=assets_root_path
        )
        add_usd_asset(
            scene,
            asset_path="/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            obj_name="mustard_bottle",
            position=TABLETOP_CENTER_POINT+[-0.055, 0.235, 0.12], #np.array([-0.065-UR_X_COORD_0, 0.065-UR_Y_COORD_0, TABLETOP_Z_COORD+0.12]),
            orientation=rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
            assets_root_path=assets_root_path
        )

    if add_bin:
        add_usd_asset(
            scene,
            asset_path=assets_root_path+"/Isaac/Props/KLT_Bin/small_KLT.usd",
            obj_name="KLT_Bin",
            position=np.array([BIN_X_COORD, BIN_Y_COORD, TABLETOP_Z_COORD+0.05]),
            orientation=None, #rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)),
            scale=np.array(BIN_SCALE),  # a shallow bin, to make it easier to pick up the bottles
        )


def add_usd_asset(scene,
                  asset_path,
                  obj_name,
                  position,
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

