bl_info = {
    "name": "BodyMove_test",
    "author": "John Peter",
    "version": (0, 16, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > BodyMove",
    "description": "AutoPhysics interval solver with Fulcrums, COM trajectories, Physics Ghost and pose preservation",
    "category": "Animation",
}

import bpy
import math
import sys
import logging

from . import dynamics, interval, reactive, balance_preview

from mathutils import Vector, Matrix
from bpy.props import (
    FloatProperty,
    BoolProperty,
    StringProperty,
    EnumProperty,
    PointerProperty,
)
from bpy_extras import view3d_utils
from bpy.app.handlers import persistent


# =========================================================
# CONSTANTS
# =========================================================

HELPER = "_bodymove_helper"
KIND = "_bodymove_kind"

MASS_POINT = "MASS_POINT"
SUPPORT_POINT = "SUPPORT_POINT"
GLOBAL_COM = "GLOBAL_COM"
COM_PROJECTION = "COM_PROJECTION"
TARGET_COM = "TARGET_COM"
SUPPORT_ZONE = "SUPPORT_ZONE"

CATEGORY = "bodymove_category"
MASS = "bodymove_mass"
BONE = "bodymove_bone"

ROLE = "bodymove_role"
RADIUS = "bodymove_radius"
SUPPORT_ACTIVE = "bodymove_support_active"
GROUND_DISTANCE = "bodymove_ground_distance"

MASS_INFLUENCE = "bodymove_mass_influence"
DISTANCE_COM = "bodymove_distance_com"
MOTION_INFLUENCE = "bodymove_motion_influence"

BASE_X = "bodymove_base_x"
BASE_Y = "bodymove_base_y"
BASE_Z = "bodymove_base_z"

SUPPORT_PREV_X = "bodymove_support_prev_x"
SUPPORT_PREV_Y = "bodymove_support_prev_y"
SUPPORT_PREV_Z = "bodymove_support_prev_z"
SUPPORT_STABLE = "bodymove_support_stable"
SUPPORT_PREV_TIME = "bodymove_support_prev_time"
SUPPORT_SPEED = "bodymove_support_speed"

_ANALYSIS_KINDS = {
    GLOBAL_COM,
    COM_PROJECTION,
    TARGET_COM,
    SUPPORT_ZONE,
}

_LIVE_GUARD = False


MASS_CATEGORIES = (
    ("HEAD", "Head", "Tête"),
    ("NECK", "Neck", "Cou"),
    ("TORSO", "Torso", "Torse"),
    ("PELVIS", "Pelvis", "Bassin"),
    ("LEFT_UPPER_ARM", "Left Upper Arm", ""),
    ("RIGHT_UPPER_ARM", "Right Upper Arm", ""),
    ("LEFT_FOREARM", "Left Forearm", ""),
    ("RIGHT_FOREARM", "Right Forearm", ""),
    ("LEFT_HAND", "Left Hand", ""),
    ("RIGHT_HAND", "Right Hand", ""),
    ("LEFT_THIGH", "Left Thigh", ""),
    ("RIGHT_THIGH", "Right Thigh", ""),
    ("LEFT_SHIN", "Left Shin", ""),
    ("RIGHT_SHIN", "Right Shin", ""),
    ("LEFT_FOOT", "Left Foot", ""),
    ("RIGHT_FOOT", "Right Foot", ""),
    ("CUSTOM", "Custom", ""),
)

HUMAN_MASS_PERCENT = {
    "HEAD": 0.068,
    "NECK": 0.013,
    "TORSO": 0.370,
    "PELVIS": 0.127,
    "LEFT_UPPER_ARM": 0.028,
    "RIGHT_UPPER_ARM": 0.028,
    "LEFT_FOREARM": 0.016,
    "RIGHT_FOREARM": 0.016,
    "LEFT_HAND": 0.006,
    "RIGHT_HAND": 0.006,
    "LEFT_THIGH": 0.100,
    "RIGHT_THIGH": 0.100,
    "LEFT_SHIN": 0.0465,
    "RIGHT_SHIN": 0.0465,
    "LEFT_FOOT": 0.0145,
    "RIGHT_FOOT": 0.0145,
}


SUPPORT_ROLES = (
    ("LEFT_FOOT", "Left Foot", ""),
    ("RIGHT_FOOT", "Right Foot", ""),
    ("LEFT_HAND", "Left Hand", ""),
    ("RIGHT_HAND", "Right Hand", ""),
    ("LEFT_KNEE", "Left Knee", ""),
    ("RIGHT_KNEE", "Right Knee", ""),
    ("CUSTOM", "Custom", ""),
)


# =========================================================
# GENERIC HELPERS
# =========================================================

def is_helper(obj):
    return bool(obj and obj.get(HELPER, False))


def helper_kind(obj):
    return obj.get(KIND) if is_helper(obj) else None


def label_from(items, identifier):
    for item_id, label, _ in items:
        if item_id == identifier:
            return label
    return identifier


def ensure_collection(scene):
    name = scene.bodymove_collection_name

    if name:
        col = bpy.data.collections.get(name)
        if col is not None:
            return col

    source = scene.bodymove_source_object
    suffix = source.name.replace(" ", "_") if source else "Body"
    name = f"BodyMove_{suffix}"

    col = bpy.data.collections.get(name)

    if col is None:
        col = bpy.data.collections.new(name)
        scene.collection.children.link(col)

    scene.bodymove_collection_name = col.name
    return col


def get_collection(scene):
    if not scene.bodymove_collection_name:
        return None
    return bpy.data.collections.get(scene.bodymove_collection_name)


def helpers(scene, wanted_kind=None):
    col = get_collection(scene)

    if col is None:
        return []

    out = []

    for obj in col.objects:
        if not is_helper(obj):
            continue

        if wanted_kind and helper_kind(obj) != wanted_kind:
            continue

        out.append(obj)

    return out


def unique_name(prefix, label):
    base = f"{prefix}_{label.replace(' ', '_')}"

    if bpy.data.objects.get(base) is None:
        return base

    i = 2
    while bpy.data.objects.get(f"{base}_{i}") is not None:
        i += 1

    return f"{base}_{i}"


def set_world_location(obj, location):
    matrix = obj.matrix_world.copy()
    matrix.translation = location
    obj.matrix_world = matrix


def select_only(context, obj):
    for selected in list(context.selected_objects):
        selected.select_set(False)

    obj.hide_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj


def bind_to_bone(obj, armature, bone_name):
    """Preserve world position while parenting marker to a bone."""
    if (
        armature is None
        or armature.type != "ARMATURE"
        or bone_name not in armature.data.bones
    ):
        return False

    world = obj.matrix_world.copy()

    obj.parent = armature
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name

    obj.matrix_world = world

    obj[BONE] = bone_name
    return True


def create_empty(
    scene,
    name,
    wanted_kind,
    location,
    display_type,
    size,
):
    col = ensure_collection(scene)

    obj = bpy.data.objects.new(name, None)
    col.objects.link(obj)

    obj[HELPER] = True
    obj[KIND] = wanted_kind

    obj.empty_display_type = display_type
    obj.empty_display_size = size
    obj.show_in_front = True
    obj.show_name = True
    obj.hide_render = True

    set_world_location(obj, location)

    return obj


def find_analysis_helper(scene, wanted_kind):
    found = helpers(scene, wanted_kind)
    return found[0] if found else None


def get_or_create_analysis_empty(
    scene,
    wanted_kind,
    name,
    location,
    display_type,
    size,
    color,
):
    obj = find_analysis_helper(scene, wanted_kind)

    if obj is None:
        obj = create_empty(
            scene,
            name,
            wanted_kind,
            location,
            display_type,
            size,
        )
    else:
        set_world_location(obj, location)
        obj.empty_display_type = display_type
        obj.empty_display_size = size

    obj.name = name
    obj.show_in_front = True
    obj.show_name = True
    obj.color = color

    return obj


def tag_redraw():
    wm = bpy.context.window_manager

    for window in wm.windows:
        screen = window.screen
        if not screen:
            continue

        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# =========================================================
# RAYCAST SOURCE MESH
# =========================================================

def raycast_source(context, coord, source):
    if source is None or source.type != "MESH":
        return None

    region = context.region
    rv3d = context.region_data

    origin_world = view3d_utils.region_2d_to_origin_3d(
        region,
        rv3d,
        coord,
    )

    direction_world = view3d_utils.region_2d_to_vector_3d(
        region,
        rv3d,
        coord,
    ).normalized()

    depsgraph = context.evaluated_depsgraph_get()
    source_eval = source.evaluated_get(depsgraph)

    inv = source_eval.matrix_world.inverted()

    origin_local = inv @ origin_world
    direction_local = (
        inv.to_3x3() @ direction_world
    ).normalized()

    hit, location_local, normal_local, face_index = (
        source_eval.ray_cast(
            origin_local,
            direction_local,
        )
    )

    if not hit:
        return None

    location_world = (
        source_eval.matrix_world @ location_local
    )

    normal_world = (
        source_eval.matrix_world.to_3x3()
        @ normal_local
    ).normalized()

    return location_world, normal_world, face_index


# =========================================================
# GROUND CONTACT
# =========================================================

def support_ground_distance(scene, support, depsgraph):
    ground = scene.bodymove_ground_object

    if ground is None:
        return None

    if ground.type != "MESH":
        return None

    try:
        ground_eval = ground.evaluated_get(depsgraph)

        world_point = (
            support.matrix_world.translation.copy()
        )

        inv = ground_eval.matrix_world.inverted()
        local_point = inv @ world_point

        hit, closest_local, normal, face_index = (
            ground_eval.closest_point_on_mesh(
                local_point
            )
        )

        if not hit:
            return None

        closest_world = (
            ground_eval.matrix_world
            @ closest_local
        )

        return (
            world_point - closest_world
        ).length

    except Exception:
        return None


def update_support_contacts(scene, depsgraph):
    # V0.5: distance + optional stability + virtual ground fallback.
    return update_support_contacts_v5(
        scene,
        depsgraph,
    )


# =========================================================
# SUPPORT GEOMETRY
# =========================================================

def distance_point_segment_xy(point, a, b):
    p = Vector((point.x, point.y))
    va = Vector((a.x, a.y))
    vb = Vector((b.x, b.y))

    ab = vb - va

    if ab.length_squared <= 1e-12:
        return (p - va).length

    t = (p - va).dot(ab) / ab.length_squared
    t = max(0.0, min(1.0, t))

    nearest = va + ab * t
    return (p - nearest).length


def convex_hull_xy(points):
    raw = sorted(
        set(
            (
                round(p.x, 8),
                round(p.y, 8),
            )
            for p in points
        )
    )

    if len(raw) <= 1:
        return [
            Vector((p[0], p[1]))
            for p in raw
        ]

    def cross(o, a, b):
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            - (a[1] - o[1]) * (b[0] - o[0])
        )

    lower = []
    for p in raw:
        while (
            len(lower) >= 2
            and cross(
                lower[-2],
                lower[-1],
                p,
            ) <= 0
        ):
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(raw):
        while (
            len(upper) >= 2
            and cross(
                upper[-2],
                upper[-1],
                p,
            ) <= 0
        ):
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]

    return [
        Vector((p[0], p[1]))
        for p in hull
    ]


def point_in_convex_xy(point, polygon):
    if len(polygon) < 3:
        return False

    p = Vector((point.x, point.y))
    sign = None

    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]

        value = (
            (b.x - a.x) * (p.y - a.y)
            - (b.y - a.y) * (p.x - a.x)
        )

        if abs(value) <= 1e-9:
            continue

        current = value > 0.0

        if sign is None:
            sign = current
        elif current != sign:
            return False

    return True


def analyze_support(projection, active_supports):
    if not active_supports:
        return {
            "balanced": False,
            "mode": "No Active Support",
            "outside": 0.0,
        }

    positions = [
        s.matrix_world.translation.copy()
        for s in active_supports
    ]

    radii = [
        max(float(s.get(RADIUS, 0.12)), 0.001)
        for s in active_supports
    ]

    if len(active_supports) == 1:
        p = positions[0]

        distance = Vector((
            projection.x - p.x,
            projection.y - p.y,
        )).length

        radius = radii[0]

        return {
            "balanced": distance <= radius,
            "mode": "Single Support",
            "outside": max(
                0.0,
                distance - radius,
            ),
        }

    if len(active_supports) == 2:
        distance = distance_point_segment_xy(
            projection,
            positions[0],
            positions[1],
        )

        radius = max(radii)

        return {
            "balanced": distance <= radius,
            "mode": "Double Support",
            "outside": max(
                0.0,
                distance - radius,
            ),
        }

    hull = convex_hull_xy(positions)

    if point_in_convex_xy(
        projection,
        hull,
    ):
        return {
            "balanced": True,
            "mode": f"{len(active_supports)} Supports",
            "outside": 0.0,
        }

    distances = [
        distance_point_segment_xy(
            projection,
            hull[i],
            hull[(i + 1) % len(hull)],
        )
        for i in range(len(hull))
    ]

    min_distance = min(distances)
    margin = max(radii)

    return {
        "balanced": min_distance <= margin,
        "mode": f"{len(active_supports)} Supports",
        "outside": max(
            0.0,
            min_distance - margin,
        ),
    }


# =========================================================
# SUPPORT ZONE VISUAL
# =========================================================

def support_zone_geometry(active_supports):
    if not active_supports:
        return [], []

    positions = [
        s.matrix_world.translation.copy()
        for s in active_supports
    ]

    z = sum(p.z for p in positions) / len(positions)

    if len(active_supports) == 1:
        center = positions[0]
        radius = max(
            float(
                active_supports[0].get(
                    RADIUS,
                    0.12,
                )
            ),
            0.01,
        )

        verts = []

        for i in range(24):
            a = math.tau * i / 24
            verts.append((
                center.x + math.cos(a) * radius,
                center.y + math.sin(a) * radius,
                z,
            ))

        return verts, [tuple(range(24))]

    if len(active_supports) == 2:
        a = positions[0]
        b = positions[1]

        radius = max(
            float(active_supports[0].get(RADIUS, 0.12)),
            float(active_supports[1].get(RADIUS, 0.12)),
            0.01,
        )

        direction = Vector((
            b.x - a.x,
            b.y - a.y,
        ))

        if direction.length <= 1e-8:
            direction = Vector((1.0, 0.0))
        else:
            direction.normalize()

        perp = Vector((
            -direction.y,
            direction.x,
        )) * radius

        verts = [
            (
                a.x + perp.x,
                a.y + perp.y,
                z,
            ),
            (
                b.x + perp.x,
                b.y + perp.y,
                z,
            ),
            (
                b.x - perp.x,
                b.y - perp.y,
                z,
            ),
            (
                a.x - perp.x,
                a.y - perp.y,
                z,
            ),
        ]

        return verts, [(0, 1, 2, 3)]

    hull = convex_hull_xy(positions)

    verts = [
        (p.x, p.y, z)
        for p in hull
    ]

    faces = (
        [tuple(range(len(verts)))]
        if len(verts) >= 3
        else []
    )

    return verts, faces


def update_support_zone(scene, active_supports):
    obj = find_analysis_helper(
        scene,
        SUPPORT_ZONE,
    )

    if not active_supports:
        if obj:
            obj.hide_viewport = True
        return

    verts, faces = support_zone_geometry(
        active_supports
    )

    if not verts:
        if obj:
            obj.hide_viewport = True
        return

    if obj is None:
        col = ensure_collection(scene)

        mesh = bpy.data.meshes.new(
            "BodyMove_SupportZoneMesh"
        )

        obj = bpy.data.objects.new(
            "BodyMove_SUPPORT_ZONE",
            mesh,
        )

        col.objects.link(obj)

        obj[HELPER] = True
        obj[KIND] = SUPPORT_ZONE

        obj.show_in_front = True
        obj.hide_render = True
        obj.display_type = "WIRE"
        obj.color = (0.1, 0.8, 0.2, 1.0)

    obj.hide_viewport = False

    mesh = obj.data

    try:
        mesh.clear_geometry()
    except Exception:
        pass

    mesh.from_pydata(
        verts,
        [],
        faces,
    )

    mesh.update()


# =========================================================
# CORE ANALYSIS
# =========================================================

def compute_analysis(
    scene,
    depsgraph,
    set_baseline=False,
):
    mass_points = helpers(
        scene,
        MASS_POINT,
    )

    valid = []

    for point in mass_points:
        try:
            mass = float(
                point.get(MASS, 0.0)
            )
        except (TypeError, ValueError):
            mass = 0.0

        if mass > 0.0:
            valid.append((point, mass))

    if not valid:
        return False

    total_mass = sum(
        mass
        for point, mass in valid
    )

    weighted = Vector((0.0, 0.0, 0.0))

    for point, mass in valid:
        weighted += (
            point.matrix_world.translation.copy()
            * mass
        )

    com = weighted / total_mass

    if set_baseline:
        scene.bodymove_base_com_x = com.x
        scene.bodymove_base_com_y = com.y
        scene.bodymove_base_com_z = com.z

    base_com = Vector((
        scene.bodymove_base_com_x,
        scene.bodymove_base_com_y,
        scene.bodymove_base_com_z,
    ))

    scene.bodymove_com_shift_live = (
        com - base_com
    ).length

    # Per-point influence.
    for point, mass in valid:
        share = mass / total_mass
        pos = point.matrix_world.translation.copy()

        point[MASS_INFLUENCE] = (
            share * 100.0
        )

        point[DISTANCE_COM] = (
            pos - com
        ).length

        if set_baseline or BASE_X not in point:
            point[BASE_X] = pos.x
            point[BASE_Y] = pos.y
            point[BASE_Z] = pos.z

        base_pos = Vector((
            float(point.get(BASE_X, pos.x)),
            float(point.get(BASE_Y, pos.y)),
            float(point.get(BASE_Z, pos.z)),
        ))

        displacement = (
            pos - base_pos
        ).length

        # Contribution teyorik point sa a nan COM shift.
        point[MOTION_INFLUENCE] = (
            share * displacement
        )

    # Visible Global COM.
    com_marker = get_or_create_analysis_empty(
        scene,
        GLOBAL_COM,
        "GLOBAL COM",
        com,
        "SPHERE",
        scene.bodymove_com_size,
        (1.0, 0.05, 0.03, 1.0),
    )

    com_marker.empty_display_size = (
        scene.bodymove_com_size
    )

    # Active support detection.
    active_supports = update_support_contacts(
        scene,
        depsgraph,
    )

    if active_supports:
        support_z = sum(
            s.matrix_world.translation.z
            for s in active_supports
        ) / len(active_supports)
    else:
        support_z = min(
            p.matrix_world.translation.z
            for p, mass in valid
        )

    projection = Vector((
        com.x,
        com.y,
        support_z,
    ))

    projection_marker = (
        get_or_create_analysis_empty(
            scene,
            COM_PROJECTION,
            "COM PROJECTION",
            projection,
            "CIRCLE",
            scene.bodymove_projection_size,
            (1.0, 0.8, 0.05, 1.0),
        )
    )

    projection_marker.empty_display_size = (
        scene.bodymove_projection_size
    )

    support_result = analyze_support(
        projection,
        active_supports,
    )

    target_projection = (
        calculate_target_projection(
            scene,
            projection,
            active_supports,
        )
    )

    create_target_com_marker(
        scene,
        target_projection,
        com.z,
    )

    update_support_zone(
        scene,
        active_supports,
    )

    # Scene outputs.
    scene.bodymove_total_mass = total_mass

    scene.bodymove_com_x = com.x
    scene.bodymove_com_y = com.y
    scene.bodymove_com_z = com.z

    scene.bodymove_mass_point_count = len(valid)
    scene.bodymove_support_count = len(
        helpers(
            scene,
            SUPPORT_POINT,
        )
    )
    scene.bodymove_active_support_count = len(
        active_supports
    )

    scene.bodymove_support_mode = (
        support_result["mode"]
    )

    scene.bodymove_balance_status = (
        "BALANCED"
        if support_result["balanced"]
        else "UNBALANCED"
    )

    scene.bodymove_distance_outside = (
        support_result["outside"]
    )

    heaviest = max(
        valid,
        key=lambda item: item[1],
    )

    scene.bodymove_heaviest_name = (
        heaviest[0].name
    )
    scene.bodymove_heaviest_mass = (
        heaviest[1]
    )

    return True






# =========================================================
# V0.6 - AUTOPHYSICS INTERVAL / PHYSICS GHOST
# =========================================================

GHOST_COLLECTION_PREFIX = "BodyMove_PhysicsPreview"
GHOST_ARMATURE_TAG = "_bodymove_ghost_armature"
GHOST_MESH_TAG = "_bodymove_ghost_mesh"
TRAJECTORY_TAG = "_bodymove_trajectory"


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def ease01(value):
    value = clamp01(value)
    return value * value * (3.0 - 2.0 * value)


def scene_interval(scene):
    if scene.bodymove_autophysics_use_scene_range:
        start = int(scene.frame_start)
        end = int(scene.frame_end)
    else:
        start = int(scene.bodymove_autophysics_start)
        end = int(scene.bodymove_autophysics_end)

    if end < start:
        start, end = end, start

    return start, end


def reset_support_tracking(scene):
    for support in helpers(scene, SUPPORT_POINT):
        for key in (
            SUPPORT_PREV_X,
            SUPPORT_PREV_Y,
            SUPPORT_PREV_Z,
            SUPPORT_STABLE,
            SUPPORT_PREV_TIME,
            SUPPORT_SPEED,
        ):
            if key in support:
                del support[key]


def active_support_roles(scene):
    return [
        str(support.get(ROLE, ""))
        for support in helpers(scene, SUPPORT_POINT)
        if bool(
            support.get(
                SUPPORT_ACTIVE,
                False,
            )
        )
    ]


def pose_depth(pb):
    depth = 0
    parent = pb.parent

    while parent is not None:
        depth += 1
        parent = parent.parent

    return depth


def sorted_pose_bones(armature):
    return sorted(
        list(armature.pose.bones),
        key=pose_depth,
    )


def capture_evaluated_pose(
    armature,
    depsgraph,
):
    evaluated = armature.evaluated_get(
        depsgraph
    )

    matrices = {}

    for pb in evaluated.pose.bones:
        matrices[pb.name] = (
            pb.matrix.copy()
        )

    return matrices


def capture_rotation_modes(armature):
    return {
        pb.name: pb.rotation_mode
        for pb in armature.pose.bones
    }


def copy_pose_to_ghost(
    ghost_armature,
    matrices,
    rotation_modes,
):
    # Parent bones first.
    for pb in sorted_pose_bones(
        ghost_armature
    ):
        matrix = matrices.get(pb.name)

        if matrix is None:
            continue

        try:
            mode = rotation_modes.get(
                pb.name,
                pb.rotation_mode,
            )

            pb.rotation_mode = mode
            pb.matrix = matrix.copy()
            bpy.context.view_layer.update()

        except Exception as exc:
            raise RuntimeError('Could not bake physics pose') from exc


def keyframe_pose_bone(
    pb,
    frame,
):
    try:
        pb.keyframe_insert(
            data_path="location",
            frame=frame,
            group=pb.name,
        )

        if pb.rotation_mode == "QUATERNION":
            pb.keyframe_insert(
                data_path="rotation_quaternion",
                frame=frame,
                group=pb.name,
            )

        elif pb.rotation_mode == "AXIS_ANGLE":
            pb.keyframe_insert(
                data_path="rotation_axis_angle",
                frame=frame,
                group=pb.name,
            )

        else:
            pb.keyframe_insert(
                data_path="rotation_euler",
                frame=frame,
                group=pb.name,
            )

        pb.keyframe_insert(
            data_path="scale",
            frame=frame,
            group=pb.name,
        )

    except Exception as exc:
        raise RuntimeError('Could not bake physics pose') from exc


def keyframe_armature_object(
    armature,
    frame,
):
    try:
        armature.keyframe_insert(
            data_path="location",
            frame=frame,
        )

        if armature.rotation_mode == "QUATERNION":
            armature.keyframe_insert(
                data_path="rotation_quaternion",
                frame=frame,
            )

        elif armature.rotation_mode == "AXIS_ANGLE":
            armature.keyframe_insert(
                data_path="rotation_axis_angle",
                frame=frame,
            )

        else:
            armature.keyframe_insert(
                data_path="rotation_euler",
                frame=frame,
            )

        armature.keyframe_insert(
            data_path="scale",
            frame=frame,
        )

    except Exception as exc:
        raise RuntimeError('Could not bake physics pose') from exc


def clear_pose_constraints(armature):
    # The Physics Ghost is baked.
    # Removing constraints avoids dependency loops and double-solving.
    for pb in armature.pose.bones:
        for constraint in list(
            pb.constraints
        ):
            try:
                pb.constraints.remove(
                    constraint
                )
            except Exception:
                pass

    for constraint in list(
        armature.constraints
    ):
        try:
            armature.constraints.remove(
                constraint
            )
        except Exception:
            pass


def get_ghost_collection(scene):
    name = scene.bodymove_ghost_collection_name

    if not name:
        return None

    return bpy.data.collections.get(name)


def get_ghost_armature(scene):
    name = scene.bodymove_ghost_armature_name

    if not name:
        return None

    return bpy.data.objects.get(name)


def remove_ghost_preview(scene):
    collection = get_ghost_collection(
        scene
    )

    if collection is not None:
        objects = list(collection.objects)

        actions_to_check = []

        for obj in objects:
            if (
                obj.animation_data is not None
                and obj.animation_data.action
                is not None
            ):
                actions_to_check.append(
                    obj.animation_data.action
                )

            data = getattr(
                obj,
                "data",
                None,
            )

            try:
                bpy.data.objects.remove(
                    obj,
                    do_unlink=True,
                )
            except Exception:
                continue

            if data is not None:
                try:
                    if data.users == 0:
                        if isinstance(
                            data,
                            bpy.types.Armature,
                        ):
                            bpy.data.armatures.remove(
                                data
                            )

                        elif isinstance(
                            data,
                            bpy.types.Curve,
                        ):
                            bpy.data.curves.remove(
                                data
                            )

                except Exception:
                    pass

        try:
            bpy.data.collections.remove(
                collection
            )
        except Exception:
            pass

        for action in actions_to_check:
            try:
                if action.users == 0:
                    bpy.data.actions.remove(
                        action
                    )
            except Exception:
                pass

    scene.bodymove_ghost_collection_name = ""
    scene.bodymove_ghost_armature_name = ""
    scene.bodymove_ghost_mesh_name = ""
    scene.bodymove_preview_ready = False


def create_ghost_preview_objects(
    scene,
):
    remove_ghost_preview(scene)

    source = scene.bodymove_source_object
    armature = (
        scene.bodymove_armature_object
    )

    if (
        source is None
        or armature is None
    ):
        return None, None, None

    collection = (
        bpy.data.collections.new(
            GHOST_COLLECTION_PREFIX
        )
    )

    scene.collection.children.link(
        collection
    )

    scene.bodymove_ghost_collection_name = (
        collection.name
    )

    # Ghost armature.
    ghost_armature = armature.copy()
    ghost_armature.data = (
        armature.data.copy()
    )
    ghost_armature.name = (
        "BodyMove_PhysicsGhost_Rig"
    )
    ghost_armature[GHOST_ARMATURE_TAG] = (
        True
    )

    ghost_armature.animation_data_clear()
    collection.objects.link(
        ghost_armature
    )

    ghost_armature.show_in_front = True

    try:
        ghost_armature.data.display_type = (
            "STICK"
        )
    except Exception:
        pass

    clear_pose_constraints(
        ghost_armature
    )

    scene.bodymove_ghost_armature_name = (
        ghost_armature.name
    )

    # Ghost mesh.
    ghost_mesh = source.copy()
    ghost_mesh.name = (
        "BodyMove_PhysicsGhost_Mesh"
    )
    ghost_mesh[GHOST_MESH_TAG] = True

    # Share the geometry data.
    # We only change object-level modifier targets and display.
    ghost_mesh.data = source.data

    try:
        ghost_mesh.animation_data_clear()
    except Exception:
        pass

    collection.objects.link(
        ghost_mesh
    )

    source_world = (
        source.matrix_world.copy()
    )

    if source.parent == armature:
        ghost_mesh.parent = ghost_armature

    ghost_mesh.matrix_world = source_world

    for modifier in ghost_mesh.modifiers:
        if (
            modifier.type == "ARMATURE"
            and modifier.object == armature
        ):
            modifier.object = (
                ghost_armature
            )

    ghost_mesh.display_type = "WIRE"
    ghost_mesh.show_in_front = True
    ghost_mesh.hide_render = True

    try:
        ghost_mesh.show_wire = True
        ghost_mesh.show_all_edges = True
        ghost_mesh.color = (
            0.12,
            0.55,
            1.0,
            0.35,
        )
    except Exception:
        pass

    scene.bodymove_ghost_mesh_name = (
        ghost_mesh.name
    )

    action = bpy.data.actions.new(
        name="BodyMove_PhysicsGhost_Action"
    )

    ghost_armature.animation_data_create()
    ghost_armature.animation_data.action = (
        action
    )

    return (
        collection,
        ghost_armature,
        ghost_mesh,
    )


def create_trajectory_curve(
    collection,
    name,
    points,
    color,
    bevel_depth,
):
    if (
        collection is None
        or len(points) < 2
    ):
        return None

    curve = bpy.data.curves.new(
        name=f"{name}_Curve",
        type="CURVE",
    )

    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 2

    spline = curve.splines.new(
        "POLY"
    )

    spline.points.add(
        len(points) - 1
    )

    for index, point in enumerate(
        points
    ):
        spline.points[index].co = (
            point.x,
            point.y,
            point.z,
            1.0,
        )

    obj = bpy.data.objects.new(
        name,
        curve,
    )

    obj[TRAJECTORY_TAG] = True
    obj.show_in_front = True
    obj.hide_render = True

    try:
        obj.color = color
    except Exception:
        pass

    collection.objects.link(obj)

    return obj


def moving_average_vectors(
    values,
    radius,
):
    if (
        radius <= 0
        or len(values) <= 2
    ):
        return [
            value.copy()
            for value in values
        ]

    result = []

    for i in range(len(values)):
        start = max(
            0,
            i - radius,
        )
        end = min(
            len(values),
            i + radius + 1,
        )

        total = Vector((
            0.0,
            0.0,
            0.0,
        ))

        count = 0

        for j in range(
            start,
            end,
        ):
            total += values[j]
            count += 1

        result.append(
            total / max(count, 1)
        )

    return result


def blend_vector_lists(
    original,
    smoothed,
    factor,
):
    factor = clamp01(factor)

    return [
        a.lerp(b, factor)
        for a, b
        in zip(
            original,
            smoothed,
        )
    ]


def temporal_filter_corrections(
    scene,
    raw,
):
    if not raw:
        return [], [], []

    separation = clamp01(
        scene.bodymove_motion_separation
    )

    trajectory_smooth = clamp01(
        scene.bodymove_smooth_trajectory
    )

    rotation_smooth = clamp01(
        scene.bodymove_smooth_rotation
    )

    max_radius = max(
        1,
        scene.bodymove_smoothing_radius,
    )

    low_radius = max(
        1,
        int(
            round(
                1
                + separation
                * max_radius
            )
        ),
    )

    low_frequency = moving_average_vectors(
        raw,
        low_radius,
    )

    separated = []

    for current, low in zip(
        raw,
        low_frequency,
    ):
        high = current - low

        separated.append(
            low
            + high
            * (
                1.0 - separation
            )
        )

    trajectory_radius = max(
        1,
        int(
            round(
                1
                + trajectory_smooth
                * max_radius
            )
        ),
    )

    trajectory_average = (
        moving_average_vectors(
            separated,
            trajectory_radius,
        )
    )

    trajectory = blend_vector_lists(
        separated,
        trajectory_average,
        trajectory_smooth,
    )

    rotation_radius = max(
        1,
        int(
            round(
                1
                + rotation_smooth
                * max_radius
            )
        ),
    )

    rotation_average = (
        moving_average_vectors(
            trajectory,
            rotation_radius,
        )
    )

    rotation = blend_vector_lists(
        trajectory,
        rotation_average,
        rotation_smooth,
    )

    secondary = []

    secondary_amount = clamp01(
        scene.bodymove_secondary_motion
    )

    for i, current in enumerate(
        rotation
    ):
        if i == 0:
            secondary.append(
                Vector((
                    0.0,
                    0.0,
                    0.0,
                ))
            )
            continue

        velocity = (
            current - rotation[i - 1]
        )

        delayed = -velocity * (
            0.55
            * secondary_amount
        )

        secondary.append(delayed)

    # Fade correction at interval edges.
    blend_frames = max(
        0,
        int(
            scene.bodymove_edge_blend_frames
        ),
    )

    if blend_frames > 0:
        count = len(trajectory)

        for i in range(count):
            fade_in = min(
                1.0,
                i
                / float(blend_frames),
            )

            fade_out = min(
                1.0,
                (count - 1 - i)
                / float(blend_frames),
            )

            fade = ease01(
                min(
                    fade_in,
                    fade_out,
                )
            )

            trajectory[i] *= fade
            rotation[i] *= fade
            secondary[i] *= fade

    return (
        trajectory,
        rotation,
        secondary,
    )


def mass_point_bone_offsets(
    scene,
    armature,
):
    data = []

    for point in helpers(
        scene,
        MASS_POINT,
    ):
        bone_name = str(
            point.get(
                BONE,
                "",
            )
        )

        if not bone_name:
            continue

        pb = armature.pose.bones.get(
            bone_name
        )

        if pb is None:
            continue

        mass = float(
            point.get(
                MASS,
                0.0,
            )
        )

        if mass <= 0.0:
            continue

        bone_world = (
            armature.matrix_world
            @ pb.matrix
        )

        try:
            local = (
                bone_world.inverted()
                @ point.matrix_world.translation
            )
        except Exception:
            continue

        data.append({
            "bone": bone_name,
            "mass": mass,
            "local": local.copy(),
        })

    return data


def calculate_armature_com_from_offsets(
    armature,
    offsets,
):
    total_mass = 0.0

    weighted = Vector((
        0.0,
        0.0,
        0.0,
    ))

    for item in offsets:
        pb = armature.pose.bones.get(
            item["bone"]
        )

        if pb is None:
            continue

        bone_world = (
            armature.matrix_world
            @ pb.matrix
        )

        position = (
            bone_world
            @ item["local"]
        )

        mass = item["mass"]

        weighted += position * mass
        total_mass += mass

    if total_mass <= 0.0:
        return None

    return weighted / total_mass


def armature_bone_world_location(
    armature,
    bone_name,
):
    if not bone_name:
        return None

    pb = armature.pose.bones.get(
        bone_name
    )

    if pb is None:
        return None

    matrix = (
        armature.matrix_world
        @ pb.matrix
    )

    return matrix.translation.copy()


def translate_ghost_foot_to_world(
    ghost_armature,
    bone_name,
    desired_world,
):
    current = armature_bone_world_location(
        ghost_armature,
        bone_name,
    )

    if current is None:
        return

    delta = desired_world - current

    if delta.length <= 1e-8:
        return

    translate_pose_bone_world(
        ghost_armature,
        bone_name,
        delta,
    )


def role_to_foot_bone(
    scene,
    role,
):
    if role == "LEFT_FOOT":
        return scene.bodymove_map_left_foot

    if role == "RIGHT_FOOT":
        return scene.bodymove_map_right_foot

    return ""


def role_support_leg(
    scene,
    roles,
):
    role_set = set(roles)

    if (
        "LEFT_FOOT" in role_set
        and "RIGHT_FOOT"
        not in role_set
    ):
        return (
            scene.bodymove_map_left_thigh,
            scene.bodymove_map_left_shin,
            scene.bodymove_map_left_foot,
        )

    if (
        "RIGHT_FOOT" in role_set
        and "LEFT_FOOT"
        not in role_set
    ):
        return (
            scene.bodymove_map_right_thigh,
            scene.bodymove_map_right_shin,
            scene.bodymove_map_right_foot,
        )

    return ("", "", "")


def apply_interval_physics_pose(
    scene,
    ghost_armature,
    translation_error,
    rotation_error,
    secondary_error,
    support_roles,
    desired_foot_world,
):
    source = scene.bodymove_source_object

    height = source_world_height(
        source
    )

    height = max(
        height,
        0.1,
    )

    physics_strength = clamp01(
        scene.bodymove_interval_physics_strength
    )

    pose_preservation = clamp01(
        scene.bodymove_pose_preservation
    )

    # Preserve a meaningful part of the animator's pose,
    # while still allowing correction.
    preservation_factor = (
        1.0
        - pose_preservation
    )

    strength = (
        physics_strength
        * preservation_factor
    )

    compensation = clamp01(
        scene.bodymove_compensation_motion
    )

    translation_error = (
        translation_error * strength
    )

    rotation_error = (
        rotation_error * strength
    )

    secondary_error = (
        secondary_error
        * strength
    )

    # Pelvis translation is intentionally limited.
    if scene.bodymove_map_pelvis:
        pelvis_factor = (
            scene.bodymove_physics_pelvis_translation
            * (
                1.0
                - 0.45
                * compensation
            )
        )

        pelvis_delta = (
            translation_error
            * pelvis_factor
        )

        pelvis_delta = clamp_vector_length(
            pelvis_delta,
            max(
                scene.bodymove_physics_max_pelvis_step,
                0.001,
            ),
        )

        translate_pose_bone_world(
            ghost_armature,
            scene.bodymove_map_pelvis,
            pelvis_delta,
        )

    max_angle = math.radians(
        max(
            0.1,
            scene.bodymove_physics_max_angle_deg,
        )
    )

    # Rotation-first compensation.
    rotate_pose_bone_world_lean(
        ghost_armature,
        scene.bodymove_map_spine,
        rotation_error,
        (
            scene.bodymove_physics_spine_gain
            * (
                1.0
                + 0.30
                * compensation
            )
        ),
        max_angle,
    )

    rotate_pose_bone_world_lean(
        ghost_armature,
        scene.bodymove_map_chest,
        rotation_error,
        (
            scene.bodymove_physics_chest_gain
            * (
                1.0
                + 0.45
                * compensation
            )
        ),
        max_angle,
    )

    if scene.bodymove_map_head:
        rotate_pose_bone_world_lean(
            ghost_armature,
            scene.bodymove_map_head,
            (
                rotation_error
                + secondary_error
            ),
            (
                scene.bodymove_physics_head_gain
                * (
                    1.0
                    + compensation
                )
            ),
            max_angle * 0.60,
        )

    # Active support leg.
    thigh, shin, foot = role_support_leg(
        scene,
        support_roles,
    )

    if thigh:
        rotate_pose_bone_world_lean(
            ghost_armature,
            thigh,
            rotation_error,
            scene.bodymove_physics_leg_gain,
            max_angle * 0.65,
        )

    if shin:
        rotate_pose_bone_world_lean(
            ghost_armature,
            shin,
            -rotation_error,
            scene.bodymove_physics_leg_gain
            * 0.45,
            max_angle * 0.50,
        )

    # Arms add subtle counter-balance + secondary motion.
    left_arm = (
        scene.bodymove_map_left_upper_arm
    )
    right_arm = (
        scene.bodymove_map_right_upper_arm
    )

    arm_motion = (
        -rotation_error
        + secondary_error
    )

    if left_arm:
        rotate_pose_bone_world_lean(
            ghost_armature,
            left_arm,
            arm_motion,
            (
                scene.bodymove_physics_arm_gain
                * (
                    1.0
                    + 0.60
                    * scene.bodymove_secondary_motion
                )
            ),
            max_angle * 0.45,
        )

    if right_arm:
        rotate_pose_bone_world_lean(
            ghost_armature,
            right_arm,
            -arm_motion,
            (
                scene.bodymove_physics_arm_gain
                * (
                    1.0
                    + 0.60
                    * scene.bodymove_secondary_motion
                )
            ),
            max_angle * 0.45,
        )




def sample_autophysics_interval(
    context,
    scene,
    start,
    end,
):
    armature = (
        scene.bodymove_armature_object
    )

    if armature is None:
        return []

    current_frame = (
        scene.frame_current
    )

    current_subframe = scene.frame_subframe

    saved_live = (
        scene.bodymove_live_analysis
    )

    saved_physics = (
        scene.bodymove_physics_enabled
    )

    saved_auto_influence = (
        scene.bodymove_auto_influence
    )

    scene.bodymove_live_analysis = False
    scene.bodymove_physics_enabled = False
    scene.bodymove_auto_influence = False

    reset_support_tracking(
        scene
    )

    rotation_modes = capture_rotation_modes(
        armature
    )

    data = []

    substeps = max(
        1,
        min(
            4,
            int(
                scene.bodymove_autophysics_substeps
            ),
        ),
    )

    try:
        for frame in range(
            start,
            end + 1,
        ):
            com_total = Vector((
                0.0,
                0.0,
                0.0,
            ))

            target_total = Vector((
                0.0,
                0.0,
                0.0,
            ))

            sample_count = 0
            exact_pose = None
            exact_object_matrix = None
            exact_roles = []
            desired_foot_world = {}
            exact_segments = {}
            reference_rotation = armature.matrix_world.to_quaternion()

            for sub_index in range(
                substeps
            ):
                if (
                    frame == end
                    and sub_index > 0
                ):
                    break

                subframe = (
                    sub_index
                    / float(substeps)
                )

                scene.frame_set(
                    frame,
                    subframe=subframe,
                )

                depsgraph = (
                    context.evaluated_depsgraph_get()
                )

                compute_analysis(
                    scene,
                    depsgraph,
                    set_baseline=False,
                )

                current_com = Vector((
                    scene.bodymove_com_x,
                    scene.bodymove_com_y,
                    scene.bodymove_com_z,
                ))

                target_com = Vector((
                    scene.bodymove_target_com_x,
                    scene.bodymove_target_com_y,
                    scene.bodymove_target_com_z,
                ))

                com_total += current_com
                target_total += target_com
                sample_count += 1

                if sub_index == 0:
                    exact_segments, reference_rotation = interval.capture_segments(
                        scene, sys.modules[__name__], depsgraph,
                    )
                    exact_pose = (
                        capture_evaluated_pose(
                            armature,
                            depsgraph,
                        )
                    )

                    exact_object_matrix = (
                        armature.matrix_world.copy()
                    )

                    exact_roles = (
                        active_support_roles(
                            scene
                        )
                    )

                    for role in exact_roles:
                        foot_bone = role_to_foot_bone(
                            scene,
                            role,
                        )

                        position = (
                            armature_bone_world_location(
                                armature,
                                foot_bone,
                            )
                        )

                        if position is not None:
                            desired_foot_world[
                                role
                            ] = position

            count = max(
                sample_count,
                1,
            )

            average_com = (
                com_total / count
            )

            average_target = (
                target_total / count
            )

            correction = (
                average_target
                - average_com
            )

            correction.z = 0.0

            data.append({
                "frame": frame,
                "com": average_com,
                "target": average_target,
                "raw_correction": correction,
                "pose": exact_pose or {},
                "rotation_modes": rotation_modes,
                "object_matrix": exact_object_matrix,
                "support_roles": exact_roles,
                "desired_foot_world": desired_foot_world,
                "segments": exact_segments,
                "reference_rotation": reference_rotation,
            })

    finally:
        scene.frame_set(
            current_frame, subframe=current_subframe,
        )
        reset_support_tracking(scene)

        scene.bodymove_live_analysis = (
            saved_live
        )

        scene.bodymove_physics_enabled = (
            saved_physics
        )

        scene.bodymove_auto_influence = (
            saved_auto_influence
        )

        try:
            compute_analysis(
                scene,
                context.evaluated_depsgraph_get(),
                set_baseline=False,
            )
        except Exception:
            pass

    return data


def build_autophysics_ghost(
    context,
    scene,
    samples,
):
    if not samples:
        return False

    (
        collection,
        ghost_armature,
        ghost_mesh,
    ) = create_ghost_preview_objects(
        scene
    )

    if (
        ghost_armature is None
        or collection is None
    ):
        return False

    raw = [
        item["raw_correction"]
        for item in samples
    ]

    (
        trajectory_corrections,
        rotation_corrections,
        secondary_corrections,
    ) = temporal_filter_corrections(
        scene,
        raw,
    )

    reactions = dynamics.solve_segment_reactions(
        samples, interval.reaction_receivers(scene, ghost_armature),
        fps=scene.render.fps / scene.render.fps_base,
        height=source_world_height(scene.bodymove_source_object),
        strength=scene.bodymove_interval_physics_strength if scene.bodymove_segment_dynamics else 0.0,
        preservation=scene.bodymove_pose_preservation,
        compensation=scene.bodymove_compensation_motion,
        secondary=scene.bodymove_secondary_motion,
        frequency=scene.bodymove_reaction_frequency,
        damping=scene.bodymove_reaction_damping,
        response_time=scene.bodymove_reaction_time,
        max_angle=math.radians(scene.bodymove_physics_max_angle_deg),
        edge_frames=scene.bodymove_edge_blend_frames,
    )
    ghost_armature['bodymove_interval_start'] = samples[0]['frame']
    ghost_armature['bodymove_interval_end'] = samples[-1]['frame']
    ghost_armature['bodymove_source_rig'] = scene.bodymove_armature_object.name
    contact_fallbacks = 0
    current_subframe = scene.frame_subframe

    current_frame = (
        scene.frame_current
    )

    saved_live = (
        scene.bodymove_live_analysis
    )

    saved_physics = (
        scene.bodymove_physics_enabled
    )

    saved_auto_influence = (
        scene.bodymove_auto_influence
    )

    scene.bodymove_live_analysis = False
    scene.bodymove_physics_enabled = False
    scene.bodymove_auto_influence = False

    original_trajectory = [
        item["com"].copy()
        for item in samples
    ]

    physics_trajectory = []

    offsets = None

    try:
        first_frame = (
            samples[0]["frame"]
        )

        scene.frame_set(
            first_frame
        )

        offsets = mass_point_bone_offsets(
            scene,
            scene.bodymove_armature_object,
        )

        for index, item in enumerate(
            samples
        ):
            frame = item["frame"]

            scene.frame_set(frame)

            if (
                item["object_matrix"]
                is not None
            ):
                ghost_armature.matrix_world = (
                    item[
                        "object_matrix"
                    ].copy()
                )

            copy_pose_to_ghost(
                ghost_armature,
                item["pose"],
                item["rotation_modes"],
            )

            context.view_layer.update()
            before_pose = {pb.name: pb.matrix_basis.copy() for pb in ghost_armature.pose.bones}
            feet = interval.capture_feet(scene, ghost_armature, item['support_roles'], sys.modules[__name__])

            # Apply time-smoothed correction.
            apply_interval_physics_pose(
                scene,
                ghost_armature,
                trajectory_corrections[index],
                rotation_corrections[index],
                Vector((0.0, 0.0, 0.0)),
                item["support_roles"],
                item["desired_foot_world"],
            )
            interval.apply_rotations(ghost_armature, reactions[index])
            if scene.bodymove_physics_foot_lock:
                tolerance = max(source_world_height(scene.bodymove_source_object) * 0.001, 0.0001)
                residual = interval.restore_feet(scene, ghost_armature, feet, sys.modules[__name__], tolerance)
                if residual > tolerance:
                    for pb in ghost_armature.pose.bones:
                        pb.matrix_basis = before_pose[pb.name]
                    context.view_layer.update()
                    contact_fallbacks += 1

            try:
                context.view_layer.update()
            except Exception:
                pass

            for pb in ghost_armature.pose.bones:
                keyframe_pose_bone(
                    pb,
                    frame,
                )

            keyframe_armature_object(
                ghost_armature,
                frame,
            )

            if offsets:
                com = (
                    calculate_armature_com_from_offsets(
                        ghost_armature,
                        offsets,
                    )
                )

                if com is not None:
                    physics_trajectory.append(
                        com.copy()
                    )

                else:
                    physics_trajectory.append(
                        item["com"]
                        + trajectory_corrections[
                            index
                        ]
                    )

            else:
                physics_trajectory.append(
                    item["com"]
                    + trajectory_corrections[
                        index
                    ]
                )

        height = source_world_height(
            scene.bodymove_source_object
        )

        thickness = max(
            0.002,
            height * 0.002,
        )

        create_trajectory_curve(
            collection,
            "COM_TRAJECTORY_ORIGINAL",
            original_trajectory,
            (
                1.0,
                0.12,
                0.12,
                1.0,
            ),
            thickness,
        )

        create_trajectory_curve(
            collection,
            "COM_TRAJECTORY_PHYSICS",
            physics_trajectory,
            (
                0.15,
                1.0,
                0.25,
                1.0,
            ),
            thickness,
        )

    finally:
        scene.frame_set(
            current_frame, subframe=current_subframe,
        )

        scene.bodymove_live_analysis = (
            saved_live
        )

        scene.bodymove_physics_enabled = (
            saved_physics
        )

        scene.bodymove_auto_influence = (
            saved_auto_influence
        )

    # Match the applied Action interpolation and avoid Bezier overshoot.
    bag = interval.channelbag(ghost_armature.animation_data)
    if bag is not None:
        for curve in bag.fcurves:
            for key in curve.keyframe_points:
                key.interpolation = 'LINEAR'
    scene.bodymove_preview_ready = True
    scene.bodymove_contact_fallbacks = contact_fallbacks
    scene.bodymove_preview_status = (
        f"READY | "
        f"{samples[0]['frame']}"
        f"-{samples[-1]['frame']} | "
        f"{len(samples)} frames"
    )

    return True


class BODYMOVE_OT_PreviewAutoPhysics(
    bpy.types.Operator
):
    bl_idname = (
        "bodymove.preview_autophysics"
    )
    bl_label = "Preview AutoPhysics"
    bl_description = "Create a non-destructive Physics Ghost over the selected animation interval"
    bl_options = {
        "REGISTER",
        "UNDO",
    }

    def execute(self, context):
        scene = context.scene
        if scene.bodymove_reactive_enabled:
            self.report({'ERROR'}, 'Use Keep Pose or Cancel in Live Balance before Preview')
            return {'CANCELLED'}

        if not scene.bodymove_initialized:
            self.report(
                {"ERROR"},
                "Run Quick Human Setup first"
            )
            return {"CANCELLED"}

        if (
            scene.bodymove_armature_object
            is None
        ):
            self.report(
                {"ERROR"},
                "Armature missing"
            )
            return {"CANCELLED"}

        if not helpers(
            scene,
            MASS_POINT,
        ):
            self.report(
                {"ERROR"},
                "No Mass Points"
            )
            return {"CANCELLED"}

        start, end = scene_interval(
            scene
        )

        frame_count = (
            end - start + 1
        )

        if frame_count <= 0:
            return {"CANCELLED"}

        max_frames = max(
            10,
            int(
                scene.bodymove_autophysics_max_frames
            ),
        )

        if frame_count > max_frames:
            self.report(
                {"ERROR"},
                (
                    f"Interval too long: "
                    f"{frame_count} frames. "
                    f"Max = {max_frames}"
                ),
            )
            return {"CANCELLED"}

        scene.bodymove_preview_status = (
            "SAMPLING..."
        )

        remove_ghost_preview(
            scene
        )

        try:
            samples = (
                sample_autophysics_interval(
                    context,
                    scene,
                    start,
                    end,
                )
            )

            if not samples:
                raise RuntimeError(
                    "No interval samples"
                )

            scene.bodymove_preview_status = (
                "BUILDING GHOST..."
            )

            if not build_autophysics_ghost(
                context,
                scene,
                samples,
            ):
                raise RuntimeError(
                    "Ghost build failed"
                )

        except Exception as exc:
            scene.bodymove_preview_status = (
                f"ERROR: {exc}"
            )

            remove_ghost_preview(
                scene
            )

            self.report(
                {"ERROR"},
                str(exc),
            )

            return {"CANCELLED"}

        self.report(
            {"INFO"},
            "Physics Ghost ready"
        )

        return {"FINISHED"}


class BODYMOVE_OT_ClearAutoPhysicsPreview(
    bpy.types.Operator
):
    bl_idname = (
        "bodymove.clear_autophysics_preview"
    )
    bl_label = "Clear Physics Preview"
    bl_options = {
        "REGISTER",
        "UNDO",
    }

    def execute(self, context):
        remove_ghost_preview(
            context.scene
        )

        context.scene.bodymove_preview_status = (
            "No preview"
        )

        return {"FINISHED"}


class BODYMOVE_OT_ApplyAutoPhysics(
    bpy.types.Operator
):
    bl_idname = (
        "bodymove.apply_autophysics"
    )
    bl_label = "Apply AutoPhysics"
    bl_description = "Apply the Physics Ghost as a new Blender Action while keeping the original Action"
    bl_options = {
        "REGISTER",
        "UNDO",
    }

    def execute(self, context):
        scene = context.scene

        armature = (
            scene.bodymove_armature_object
        )

        ghost = get_ghost_armature(
            scene
        )

        if (
            armature is None
            or ghost is None
            or ghost.animation_data
            is None
            or ghost.animation_data.action
            is None
        ):
            self.report(
                {"ERROR"},
                "Preview Physics Ghost first"
            )
            return {"CANCELLED"}

        reason = interval.apply_block_reason(armature)
        if reason:
            self.report({'ERROR'}, reason)
            return {'CANCELLED'}
        if ghost.get('bodymove_source_rig') != armature.name:
            self.report({'ERROR'}, 'Source rig changed; regenerate preview')
            return {'CANCELLED'}
        original_action = armature.animation_data.action if armature.animation_data else None
        try:
            new_action, slot = interval.merge_action_interval(
                armature, ghost, int(ghost['bodymove_interval_start']), int(ghost['bodymove_interval_end']),
            )
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        if original_action is not None:
            original_action.use_fake_user = True
            # Preserve the first original across repeated applications.
            if original_action.name != scene.bodymove_last_applied_action_name:
                scene.bodymove_original_action_name = original_action.name
                scene.bodymove_original_slot = armature.animation_data.action_slot.identifier if armature.animation_data.action_slot else ''
        new_action.name = f"{original_action.name if original_action else armature.name}_BodyMovePhysics"
        armature.animation_data_create()
        armature.animation_data.action = new_action
        armature.animation_data.action_slot = slot

        scene.bodymove_last_applied_action_name = (
            new_action.name
        )

        remove_ghost_preview(
            scene
        )

        scene.bodymove_preview_status = (
            f"APPLIED: {new_action.name}"
        )

        self.report(
            {"INFO"},
            (
                f"New Action: "
                f"{new_action.name}"
            ),
        )

        return {"FINISHED"}


class BODYMOVE_OT_RestoreOriginalAction(
    bpy.types.Operator
):
    bl_idname = (
        "bodymove.restore_original_action"
    )
    bl_label = "Restore Original Action"
    bl_options = {
        "REGISTER",
        "UNDO",
    }

    def execute(self, context):
        scene = context.scene
        armature = (
            scene.bodymove_armature_object
        )

        if armature is None:
            return {"CANCELLED"}

        name = (
            scene.bodymove_original_action_name
        )

        action = (
            bpy.data.actions.get(name)
            if name
            else None
        )

        if action is None:
            self.report(
                {"ERROR"},
                "Original Action not found"
            )
            return {"CANCELLED"}

        armature.animation_data_create()
        armature.animation_data.action = (
            action
        )

        slot = next((s for s in action.slots if s.identifier == scene.bodymove_original_slot), None)
        if slot is not None:
            armature.animation_data.action_slot = slot
        scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
        self.report(
            {"INFO"},
            f"Restored: {action.name}"
        )

        return {"FINISHED"}


def support_by_role(
    scene,
    role,
):
    for support in helpers(
        scene,
        SUPPORT_POINT,
    ):
        if support.get(ROLE, "") == role:
            return support

    return None


# =========================================================
# V0.5 - SMART HUMAN SETUP / PHYSICS CORE
# =========================================================

def source_world_height(source):
    if source is None:
        return 1.75

    try:
        points = [
            source.matrix_world @ Vector(corner)
            for corner in source.bound_box
        ]

        if not points:
            return 1.75

        z_values = [p.z for p in points]
        return max(
            max(z_values) - min(z_values),
            0.01,
        )

    except Exception:
        return 1.75


def source_lowest_world_z(source):
    if source is None:
        return 0.0

    try:
        points = [
            source.matrix_world @ Vector(corner)
            for corner in source.bound_box
        ]

        if not points:
            return 0.0

        return min(p.z for p in points)

    except Exception:
        return 0.0


def find_ground_candidate(scene, source):
    keywords = (
        "ground",
        "floor",
        "plane",
        "sol",
        "terrain",
    )

    candidates = []

    for obj in scene.objects:
        if obj == source:
            continue

        if obj.type != "MESH":
            continue

        name = obj.name.lower()

        if any(key in name for key in keywords):
            candidates.append(obj)

    if not candidates:
        return None

    source_z = (
        source.matrix_world.translation.z
        if source is not None
        else 0.0
    )

    return min(
        candidates,
        key=lambda obj: abs(
            obj.matrix_world.translation.z
            - source_z
        ),
    )


def pose_bone_center_world(armature, bone_name):
    if (
        armature is None
        or armature.type != "ARMATURE"
        or not bone_name
    ):
        return None

    pb = armature.pose.bones.get(bone_name)

    if pb is None:
        return None

    center_armature = (
        pb.head + pb.tail
    ) * 0.5

    return (
        armature.matrix_world
        @ center_armature
    )


def pose_bone_tail_world(armature, bone_name):
    if (
        armature is None
        or armature.type != "ARMATURE"
        or not bone_name
    ):
        return None

    pb = armature.pose.bones.get(bone_name)

    if pb is None:
        return None

    return (
        armature.matrix_world
        @ pb.tail
    )


def ground_closest_world(scene, world_point, depsgraph):
    ground = scene.bodymove_ground_object

    if (
        ground is None
        or ground.type != "MESH"
    ):
        return Vector((
            world_point.x,
            world_point.y,
            scene.bodymove_virtual_ground_z,
        ))

    try:
        ground_eval = ground.evaluated_get(
            depsgraph
        )

        inv = ground_eval.matrix_world.inverted()
        local = inv @ world_point

        hit, closest_local, normal, face_index = (
            ground_eval.closest_point_on_mesh(
                local
            )
        )

        if hit:
            return (
                ground_eval.matrix_world
                @ closest_local
            )

    except Exception:
        pass

    return Vector((
        world_point.x,
        world_point.y,
        scene.bodymove_virtual_ground_z,
    ))


def generic_human_bone_map(armature):
    profile = detect_rig_profile(armature)

    if profile == "MIXAMO":
        exact = {
            "HEAD": ["mixamorig:Head"],
            "NECK": ["mixamorig:Neck"],
            "TORSO": ["mixamorig:Spine2", "mixamorig:Spine1"],
            "PELVIS": ["mixamorig:Hips"],
            "LEFT_UPPER_ARM": ["mixamorig:LeftArm"],
            "RIGHT_UPPER_ARM": ["mixamorig:RightArm"],
            "LEFT_FOREARM": ["mixamorig:LeftForeArm"],
            "RIGHT_FOREARM": ["mixamorig:RightForeArm"],
            "LEFT_HAND": ["mixamorig:LeftHand"],
            "RIGHT_HAND": ["mixamorig:RightHand"],
            "LEFT_THIGH": ["mixamorig:LeftUpLeg"],
            "RIGHT_THIGH": ["mixamorig:RightUpLeg"],
            "LEFT_SHIN": ["mixamorig:LeftLeg"],
            "RIGHT_SHIN": ["mixamorig:RightLeg"],
            "LEFT_FOOT": ["mixamorig:LeftFoot"],
            "RIGHT_FOOT": ["mixamorig:RightFoot"],
        }

    elif profile == "RIGIFY":
        exact = {
            "HEAD": ["head"],
            "NECK": ["neck"],
            "TORSO": ["chest", "spine_fk.003", "torso"],
            "PELVIS": ["hips", "pelvis", "torso"],
            "LEFT_UPPER_ARM": ["upper_arm_fk.L", "upper_arm.L"],
            "RIGHT_UPPER_ARM": ["upper_arm_fk.R", "upper_arm.R"],
            "LEFT_FOREARM": ["forearm_fk.L", "forearm.L"],
            "RIGHT_FOREARM": ["forearm_fk.R", "forearm.R"],
            "LEFT_HAND": ["hand_fk.L", "hand.L"],
            "RIGHT_HAND": ["hand_fk.R", "hand.R"],
            "LEFT_THIGH": ["thigh_fk.L", "thigh.L"],
            "RIGHT_THIGH": ["thigh_fk.R", "thigh.R"],
            "LEFT_SHIN": ["shin_fk.L", "shin.L"],
            "RIGHT_SHIN": ["shin_fk.R", "shin.R"],
            "LEFT_FOOT": ["foot_ik.L", "foot_fk.L", "foot.L"],
            "RIGHT_FOOT": ["foot_ik.R", "foot_fk.R", "foot.R"],
        }

    elif profile == "AUTO_RIG_PRO":
        # Auto-Rig Pro has two naming layers: c_* controls and deform bones.
        # Mass points must follow deform bones; prefer those exact names here.
        exact = {
            "HEAD": ["head", "head.x"],
            "NECK": ["neck", "neck.x", "neck_twist.x"],
            "TORSO": ["spine_02", "spine_01", "spine_02.x", "spine_01.x"],
            "PELVIS": ["root.x", "pelvis", "hips", "pelvis.x", "hips.x"],
            "LEFT_UPPER_ARM": ["arm_stretch.l", "arm.l", "upper_arm.l", "arm_stretch.L"],
            "RIGHT_UPPER_ARM": ["arm_stretch.r", "arm.r", "upper_arm.r", "arm_stretch.R"],
            "LEFT_FOREARM": ["forearm.l", "forearm_stretch.l", "forearm.L"],
            "RIGHT_FOREARM": ["forearm.r", "forearm_stretch.r", "forearm.R"],
            "LEFT_HAND": ["hand.l", "hand.L"],
            "RIGHT_HAND": ["hand.r", "hand.R"],
            "LEFT_THIGH": ["thigh.l", "thigh.L"],
            "RIGHT_THIGH": ["thigh.r", "thigh.R"],
            "LEFT_SHIN": ["leg.l", "shin.l", "shin.L"],
            "RIGHT_SHIN": ["leg.r", "shin.r", "shin.R"],
            "LEFT_FOOT": ["foot.l", "foot.L", "c_foot_ik.l"],
            "RIGHT_FOOT": ["foot.r", "foot.R", "c_foot_ik.r"],
        }

    else:
        exact = {
            "HEAD": ["head"],
            "NECK": ["neck"],
            "TORSO": ["chest", "upperchest", "spine2", "spine_02", "torso"],
            "PELVIS": ["hips", "pelvis", "hip"],
            "LEFT_UPPER_ARM": ["leftarm", "upperarm.l", "upper_arm.l", "arm_l"],
            "RIGHT_UPPER_ARM": ["rightarm", "upperarm.r", "upper_arm.r", "arm_r"],
            "LEFT_FOREARM": ["leftforearm", "forearm.l", "fore_arm.l"],
            "RIGHT_FOREARM": ["rightforearm", "forearm.r", "fore_arm.r"],
            "LEFT_HAND": ["lefthand", "hand.l", "hand_l"],
            "RIGHT_HAND": ["righthand", "hand.r", "hand_r"],
            "LEFT_THIGH": [
                "thigh_l", "upperleg_l", "upper_leg_l", "upleg_l",
                "left_thigh", "leftupleg", "thigh.l", "upperleg.l", "leg_upper.l",
            ],
            "RIGHT_THIGH": [
                "thigh_r", "upperleg_r", "upper_leg_r", "upleg_r",
                "right_thigh", "rightupleg", "thigh.r", "upperleg.r", "leg_upper.r",
            ],
            "LEFT_SHIN": [
                "calf_l", "shin_l", "lowerleg_l", "lower_leg_l", "leg_l",
                "left_shin", "leftleg", "shin.l", "lowerleg.l", "calf.l",
            ],
            "RIGHT_SHIN": [
                "calf_r", "shin_r", "lowerleg_r", "lower_leg_r", "leg_r",
                "right_shin", "rightleg", "shin.r", "lowerleg.r", "calf.r",
            ],
            "LEFT_FOOT": ["leftfoot", "foot_ik.l", "foot.l", "foot_l"],
            "RIGHT_FOOT": ["rightfoot", "foot_ik.r", "foot.r", "foot_r"],
        }

    return {
        category: find_bone_fuzzy(
            armature,
            candidates,
        )
        for category, candidates
        in exact.items()
    }


def existing_mass_categories(scene):
    return {
        point.get(CATEGORY, "")
        for point in helpers(
            scene,
            MASS_POINT,
        )
    }


def existing_support_roles(scene):
    return {
        point.get(ROLE, "")
        for point in helpers(
            scene,
            SUPPORT_POINT,
        )
    }


def create_missing_human_mass_points(
    scene,
    armature,
):
    mapping = generic_human_bone_map(
        armature
    )

    existing = existing_mass_categories(
        scene
    )

    body_weight = max(
        scene.bodymove_body_weight,
        1.0,
    )

    created = 0

    for category, fraction in HUMAN_MASS_PERCENT.items():
        if category in existing:
            continue

        bone_name = mapping.get(
            category,
            "",
        )

        if not bone_name:
            continue

        location = pose_bone_center_world(
            armature,
            bone_name,
        )

        if location is None:
            continue

        label = label_from(
            MASS_CATEGORIES,
            category,
        )

        mass = body_weight * fraction

        obj = create_empty(
            scene,
            unique_name(
                "BM_MP",
                label,
            ),
            MASS_POINT,
            location,
            "SPHERE",
            scene.bodymove_mass_point_size,
        )

        obj[CATEGORY] = category
        obj[MASS] = float(mass)

        if bind_to_bone(
            obj,
            armature,
            bone_name,
        ):
            obj.name = (
                f"MP | {label} | "
                f"{mass:.2f}kg | {bone_name}"
            )
            created += 1
        else:
            bpy.data.objects.remove(
                obj,
                do_unlink=True,
            )

    return created, mapping


def create_missing_foot_supports(
    scene,
    armature,
    mapping,
    depsgraph,
):
    existing = existing_support_roles(
        scene
    )

    created = 0

    for role, category in (
        ("LEFT_FOOT", "LEFT_FOOT"),
        ("RIGHT_FOOT", "RIGHT_FOOT"),
    ):
        if role in existing:
            continue

        bone_name = mapping.get(
            category,
            "",
        )

        if not bone_name:
            continue

        bone_point = pose_bone_tail_world(
            armature,
            bone_name,
        )

        if bone_point is None:
            bone_point = pose_bone_center_world(
                armature,
                bone_name,
            )

        if bone_point is None:
            continue

        support_world = ground_closest_world(
            scene,
            bone_point,
            depsgraph,
        )

        label = label_from(
            SUPPORT_ROLES,
            role,
        )

        obj = create_empty(
            scene,
            unique_name(
                "BM_SP",
                label,
            ),
            SUPPORT_POINT,
            support_world,
            "CIRCLE",
            scene.bodymove_support_radius,
        )

        obj[ROLE] = role
        obj[RADIUS] = float(
            scene.bodymove_support_radius
        )
        obj[SUPPORT_ACTIVE] = True

        if bind_to_bone(
            obj,
            armature,
            bone_name,
        ):
            obj.name = (
                f"SP | {label} | {bone_name}"
            )
            created += 1
        else:
            bpy.data.objects.remove(
                obj,
                do_unlink=True,
            )

    return created


def set_extended_mapping_from_human_map(
    scene,
    mapping,
):
    scene.bodymove_map_left_upper_arm = mapping.get(
        "LEFT_UPPER_ARM",
        "",
    )
    scene.bodymove_map_right_upper_arm = mapping.get(
        "RIGHT_UPPER_ARM",
        "",
    )
    scene.bodymove_map_left_thigh = mapping.get(
        "LEFT_THIGH",
        "",
    )
    scene.bodymove_map_right_thigh = mapping.get(
        "RIGHT_THIGH",
        "",
    )
    scene.bodymove_map_left_shin = mapping.get(
        "LEFT_SHIN",
        "",
    )
    scene.bodymove_map_right_shin = mapping.get(
        "RIGHT_SHIN",
        "",
    )


class BODYMOVE_OT_QuickHumanSetup(bpy.types.Operator):
    bl_idname = "bodymove.quick_human_setup"
    bl_label = "Quick Human Setup"
    bl_description = "Auto-detect rig, create human Mass Points and Foot Supports, then calculate COM"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene

        source = scene.bodymove_source_object

        if source is None:
            active = context.active_object

            if (
                active is not None
                and active.type == "MESH"
            ):
                source = active

            else:
                selected_mesh = next(
                    (
                        obj
                        for obj in context.selected_objects
                        if obj.type == "MESH"
                    ),
                    None,
                )

                source = selected_mesh

            scene.bodymove_source_object = (
                source
            )

        if (
            source is None
            or source.type != "MESH"
        ):
            self.report(
                {"ERROR"},
                "Sélectionnez le Mesh humain"
            )
            return {"CANCELLED"}

        armature = scene.bodymove_armature_object

        if armature is None:
            try:
                armature = source.find_armature()
            except Exception:
                armature = None

            scene.bodymove_armature_object = (
                armature
            )

        if (
            armature is None
            or armature.type != "ARMATURE"
        ):
            self.report(
                {"ERROR"},
                "Armature non détectée"
            )
            return {"CANCELLED"}

        if scene.bodymove_ground_object is None:
            scene.bodymove_ground_object = (
                find_ground_candidate(
                    scene,
                    source,
                )
            )

        height = source_world_height(source)

        scene.bodymove_virtual_ground_z = (
            source_lowest_world_z(source)
        )

        # Auto-scale defaults according to character size.
        scene.bodymove_contact_tolerance = max(
            height * 0.02,
            0.005,
        )
        scene.bodymove_support_motion_tolerance = max(
            height * 0.015,
            0.005,
        )
        scene.bodymove_support_radius = max(
            height * 0.06,
            0.02,
        )
        scene.bodymove_mass_point_size = max(
            height * 0.018,
            0.01,
        )
        scene.bodymove_com_size = max(
            height * 0.045,
            0.02,
        )
        scene.bodymove_projection_size = max(
            height * 0.04,
            0.02,
        )

        ensure_collection(scene)

        scene.bodymove_initialized = True
        scene.bodymove_analysis_valid = False

        if (
            scene.bodymove_autophysics_start
            >= scene.bodymove_autophysics_end
        ):
            scene.bodymove_autophysics_start = (
                scene.frame_start
            )
            scene.bodymove_autophysics_end = (
                scene.frame_end
            )

        auto_map_rig(scene)

        created_mass, mapping = (
            create_missing_human_mass_points(
                scene,
                armature,
            )
        )

        set_extended_mapping_from_human_map(
            scene,
            mapping,
        )

        depsgraph = (
            context.evaluated_depsgraph_get()
        )

        created_support = (
            create_missing_foot_supports(
                scene,
                armature,
                mapping,
                depsgraph,
            )
        )

        success = compute_analysis(
            scene,
            depsgraph,
            set_baseline=True,
        )

        if success:
            scene.bodymove_analysis_valid = True
            scene.bodymove_physics_status = "READY"
            scene.bodymove_solver_status = "READY"
        scene.bodymove_physics_status = "READY"
        scene.bodymove_physics_error = (
            current_target_error(scene).length
        )

        tag_redraw()

        self.report(
            {"INFO"},
            (
                f"Quick Setup | "
                f"{created_mass} Mass Points | "
                f"{created_support} Supports"
            ),
        )

        return {"FINISHED"}


def virtual_ground_distance(scene, support):
    return abs(
        support.matrix_world.translation.z
        - scene.bodymove_virtual_ground_z
    )


def update_support_contacts_v5(
    scene,
    depsgraph,
):
    ground = scene.bodymove_ground_object

    tolerance = max(
        scene.bodymove_contact_tolerance,
        0.0001,
    )

    motion_tolerance = max(
        scene.bodymove_support_motion_tolerance,
        0.0001,
    )

    active = []

    for support in helpers(
        scene,
        SUPPORT_POINT,
    ):
        pos = (
            support.matrix_world.translation.copy()
        )

        if ground is None:
            distance = virtual_ground_distance(
                scene,
                support,
            )
        else:
            distance = support_ground_distance(
                scene,
                support,
                depsgraph,
            )

        if distance is None:
            support[SUPPORT_ACTIVE] = False
            support[GROUND_DISTANCE] = -1.0
            continue

        previous = None

        if SUPPORT_PREV_X in support:
            previous = Vector((
                float(
                    support.get(
                        SUPPORT_PREV_X,
                        pos.x,
                    )
                ),
                float(
                    support.get(
                        SUPPORT_PREV_Y,
                        pos.y,
                    )
                ),
                float(
                    support.get(
                        SUPPORT_PREV_Z,
                        pos.z,
                    )
                ),
            ))

        movement = (
            (pos - previous).length
            if previous is not None
            else 0.0
        )

        now = (scene.frame_current + scene.frame_subframe) / (scene.render.fps / scene.render.fps_base)
        previous_time = float(support.get(SUPPORT_PREV_TIME, now))
        dt = now - previous_time
        if dt > 1e-8 and previous is not None:
            speed = movement / dt
        elif dt < -1e-8:
            speed = 0.0  # Timeline seek: discard future history.
        else:
            speed = float(support.get(SUPPORT_SPEED, 0.0))
        # Existing tolerance was displacement per frame at 24 FPS.
        stable = not scene.bodymove_use_support_stability or speed <= motion_tolerance * 24.0
        support[SUPPORT_PREV_TIME] = now
        support[SUPPORT_SPEED] = speed

        if abs(dt) > 1e-8 or previous is None:
            support[SUPPORT_PREV_X] = pos.x
            support[SUPPORT_PREV_Y] = pos.y
            support[SUPPORT_PREV_Z] = pos.z

        support[GROUND_DISTANCE] = float(
            distance
        )
        support[SUPPORT_STABLE] = bool(
            stable
        )

        is_active = (
            distance <= tolerance
            and stable
        )

        support[SUPPORT_ACTIVE] = bool(
            is_active
        )

        if is_active:
            active.append(support)

    return active


def nearest_point_on_segment_xy(
    point,
    a,
    b,
):
    p = Vector((point.x, point.y))
    va = Vector((a.x, a.y))
    vb = Vector((b.x, b.y))

    ab = vb - va

    if ab.length_squared <= 1e-12:
        return va

    t = (
        (p - va).dot(ab)
        / ab.length_squared
    )

    t = max(
        0.0,
        min(1.0, t),
    )

    return va + ab * t


def calculate_target_projection(
    scene,
    current_projection,
    active_supports,
):
    if not active_supports:
        return current_projection.copy()

    margin_factor = max(
        0.1,
        min(
            0.95,
            scene.bodymove_target_margin,
        ),
    )

    positions = [
        s.matrix_world.translation.copy()
        for s in active_supports
    ]

    radii = [
        max(
            float(
                s.get(
                    RADIUS,
                    scene.bodymove_support_radius,
                )
            ),
            0.001,
        )
        for s in active_supports
    ]

    p = Vector((
        current_projection.x,
        current_projection.y,
    ))

    if len(active_supports) == 1:
        center = Vector((
            positions[0].x,
            positions[0].y,
        ))

        radius = radii[0]
        safe_radius = radius * margin_factor

        delta = p - center
        distance = delta.length

        if distance <= safe_radius:
            target_xy = p
        elif distance <= 1e-12:
            target_xy = center
        else:
            target_xy = (
                center
                + delta.normalized()
                * safe_radius
            )

        return Vector((
            target_xy.x,
            target_xy.y,
            current_projection.z,
        ))

    if len(active_supports) == 2:
        q = nearest_point_on_segment_xy(
            current_projection,
            positions[0],
            positions[1],
        )

        radius = max(radii)
        safe_radius = (
            radius * margin_factor
        )

        delta = p - q
        distance = delta.length

        if distance <= safe_radius:
            target_xy = p
        elif distance <= 1e-12:
            target_xy = q
        else:
            target_xy = (
                q
                + delta.normalized()
                * safe_radius
            )

        return Vector((
            target_xy.x,
            target_xy.y,
            current_projection.z,
        ))

    hull = convex_hull_xy(positions)

    if point_in_convex_xy(
        current_projection,
        hull,
    ):
        return current_projection.copy()

    center = Vector((
        sum(p.x for p in positions)
        / len(positions),
        sum(p.y for p in positions)
        / len(positions),
    ))

    return Vector((
        center.x,
        center.y,
        current_projection.z,
    ))


def create_target_com_marker(
    scene,
    target_projection,
    com_z,
):
    target = Vector((
        target_projection.x,
        target_projection.y,
        com_z,
    ))

    marker = get_or_create_analysis_empty(
        scene,
        TARGET_COM,
        "TARGET COM",
        target,
        "SPHERE",
        scene.bodymove_com_size * 0.8,
        (0.1, 0.8, 1.0, 1.0),
    )

    scene.bodymove_target_com_x = target.x
    scene.bodymove_target_com_y = target.y
    scene.bodymove_target_com_z = target.z

    return marker


def world_axis_to_armature(
    armature,
    axis,
):
    vector = (
        armature.matrix_world
        .inverted()
        .to_3x3()
        @ axis
    )

    if vector.length <= 1e-12:
        return axis.copy()

    return vector.normalized()


def rotate_pose_bone_world_lean(
    armature,
    bone_name,
    world_error,
    gain,
    max_angle,
):
    if (
        armature is None
        or armature.type != "ARMATURE"
        or not bone_name
    ):
        return False

    pb = armature.pose.bones.get(
        bone_name
    )

    if pb is None:
        return False

    source = bpy.context.scene.bodymove_source_object
    height = source_world_height(
        source
    )

    if height <= 1e-6:
        height = 1.75

    # Z is up.
    # +X COM target => lean through +Y rotation.
    # +Y COM target => lean through -X rotation.
    angle_y = (
        world_error.x
        / height
        * gain
    )

    angle_x = (
        -world_error.y
        / height
        * gain
    )

    angle_x = max(
        -max_angle,
        min(max_angle, angle_x),
    )

    angle_y = max(
        -max_angle,
        min(max_angle, angle_y),
    )

    if (
        abs(angle_x) <= 1e-8
        and abs(angle_y) <= 1e-8
    ):
        return True

    axis_x = world_axis_to_armature(
        armature,
        Vector((1.0, 0.0, 0.0)),
    )

    axis_y = world_axis_to_armature(
        armature,
        Vector((0.0, 1.0, 0.0)),
    )

    pivot = pb.head.copy()

    try:
        transform = (
            Matrix.Translation(pivot)
            @ Matrix.Rotation(
                angle_y,
                4,
                axis_y,
            )
            @ Matrix.Rotation(
                angle_x,
                4,
                axis_x,
            )
            @ Matrix.Translation(-pivot)
        )

        pb.matrix = (
            transform @ pb.matrix
        )

        return True

    except Exception:
        return False


def support_side(scene):
    active = get_active_supports(scene)

    roles = {
        support.get(ROLE, "")
        for support in active
    }

    if (
        "LEFT_FOOT" in roles
        and "RIGHT_FOOT" not in roles
    ):
        return "LEFT"

    if (
        "RIGHT_FOOT" in roles
        and "LEFT_FOOT" not in roles
    ):
        return "RIGHT"

    if (
        "LEFT_FOOT" in roles
        and "RIGHT_FOOT" in roles
    ):
        return "BOTH"

    return "NONE"


def mapped_support_leg(scene):
    side = support_side(scene)

    if side == "LEFT":
        return (
            scene.bodymove_map_left_thigh,
            scene.bodymove_map_left_shin,
            scene.bodymove_map_left_foot,
        )

    if side == "RIGHT":
        return (
            scene.bodymove_map_right_thigh,
            scene.bodymove_map_right_shin,
            scene.bodymove_map_right_foot,
        )

    return ("", "", "")


def mapped_counterbalance_arms(scene):
    side = support_side(scene)

    if side == "LEFT":
        return (
            scene.bodymove_map_right_upper_arm,
            scene.bodymove_map_left_upper_arm,
        )

    if side == "RIGHT":
        return (
            scene.bodymove_map_left_upper_arm,
            scene.bodymove_map_right_upper_arm,
        )

    return (
        scene.bodymove_map_left_upper_arm,
        scene.bodymove_map_right_upper_arm,
    )


def current_target_error(scene):
    current = Vector((
        scene.bodymove_com_x,
        scene.bodymove_com_y,
        scene.bodymove_com_z,
    ))

    target = Vector((
        scene.bodymove_target_com_x,
        scene.bodymove_target_com_y,
        scene.bodymove_target_com_z,
    ))

    error = target - current
    error.z = 0.0

    return error


def capture_active_support_world_positions(
    scene,
):
    return [
        (
            support,
            support.matrix_world.translation.copy(),
            support_role_foot_control(
                scene,
                support,
            ),
        )
        for support
        in get_active_supports(scene)
    ]


def restore_support_positions_with_foot_controls(
    scene,
    armature,
    captured,
):
    if not scene.bodymove_physics_foot_lock:
        return

    for (
        support,
        desired_world,
        foot_bone,
    ) in captured:
        if not foot_bone:
            continue

        current = (
            support.matrix_world.translation.copy()
        )

        delta = desired_world - current

        # Keep only horizontal stabilization.
        delta.z = 0.0

        max_lock = max(
            scene.bodymove_physics_max_pelvis_step
            * 1.5,
            0.001,
        )

        delta = clamp_vector_length(
            delta,
            max_lock,
        )

        if delta.length > 1e-8:
            translate_pose_bone_world(
                armature,
                foot_bone,
                delta,
            )


def physics_solver_iteration(
    scene,
    depsgraph,
):
    armature = (
        scene.bodymove_armature_object
    )

    if (
        armature is None
        or armature.type != "ARMATURE"
    ):
        scene.bodymove_physics_status = (
            "Armature invalide"
        )
        return False

    active = get_active_supports(scene)

    if not active:
        scene.bodymove_physics_status = (
            "No Active Fulcrum"
        )
        return False

    error = current_target_error(scene)
    error_length = error.length

    deadzone = max(
        scene.bodymove_physics_deadzone,
        0.0005,
    )

    if error_length <= deadzone:
        scene.bodymove_physics_status = (
            "STABLE"
        )
        return False

    captured_supports = (
        capture_active_support_world_positions(
            scene
        )
    )

    strength = max(
        0.0,
        min(
            1.0,
            scene.bodymove_physics_strength,
        ),
    )

    working_error = error * strength

    # 1. Small pelvis translation only.
    if scene.bodymove_map_pelvis:
        pelvis_delta = (
            working_error
            * scene.bodymove_physics_pelvis_translation
        )

        pelvis_delta = clamp_vector_length(
            pelvis_delta,
            max(
                scene.bodymove_physics_max_pelvis_step,
                0.001,
            ),
        )

        translate_pose_bone_world(
            armature,
            scene.bodymove_map_pelvis,
            pelvis_delta,
        )

    # 2. Rotation-first body chain.
    max_angle = math.radians(
        max(
            0.1,
            scene.bodymove_physics_max_angle_deg,
        )
    )

    rotate_pose_bone_world_lean(
        armature,
        scene.bodymove_map_spine,
        working_error,
        scene.bodymove_physics_spine_gain,
        max_angle,
    )

    rotate_pose_bone_world_lean(
        armature,
        scene.bodymove_map_chest,
        working_error,
        scene.bodymove_physics_chest_gain,
        max_angle,
    )

    # Head uses smaller counter-orientation.
    if scene.bodymove_map_head:
        rotate_pose_bone_world_lean(
            armature,
            scene.bodymove_map_head,
            working_error,
            scene.bodymove_physics_head_gain,
            max_angle * 0.6,
        )

    # 3. Support leg compensation.
    thigh, shin, foot = (
        mapped_support_leg(scene)
    )

    if thigh:
        rotate_pose_bone_world_lean(
            armature,
            thigh,
            working_error,
            scene.bodymove_physics_leg_gain,
            max_angle * 0.65,
        )

    if shin:
        rotate_pose_bone_world_lean(
            armature,
            shin,
            -working_error,
            scene.bodymove_physics_leg_gain * 0.45,
            max_angle * 0.5,
        )

    # 4. Small arm counterbalance.
    arm_a, arm_b = (
        mapped_counterbalance_arms(scene)
    )

    if arm_a:
        rotate_pose_bone_world_lean(
            armature,
            arm_a,
            -working_error,
            scene.bodymove_physics_arm_gain,
            max_angle * 0.5,
        )

    if arm_b:
        rotate_pose_bone_world_lean(
            armature,
            arm_b,
            working_error,
            scene.bodymove_physics_arm_gain * 0.5,
            max_angle * 0.35,
        )

    armature.update_tag()

    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    restore_support_positions_with_foot_controls(
        scene,
        armature,
        captured_supports,
    )

    armature.update_tag()

    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    compute_analysis(
        scene,
        depsgraph,
        set_baseline=False,
    )

    return True


def physics_solver_step(
    scene,
    depsgraph,
):
    if not scene.bodymove_physics_enabled:
        return False

    if not scene.bodymove_analysis_valid:
        return False

    iterations = max(
        1,
        min(
            12,
            scene.bodymove_physics_iterations,
        ),
    )

    changed = False

    start_error = (
        current_target_error(scene).length
    )

    for _ in range(iterations):
        before = (
            current_target_error(scene).length
        )

        if before <= scene.bodymove_physics_deadzone:
            break

        if not physics_solver_iteration(
            scene,
            depsgraph,
        ):
            break

        changed = True

        after = (
            current_target_error(scene).length
        )

        if after >= before - 1e-5:
            break

    final_error = (
        current_target_error(scene).length
    )

    scene.bodymove_physics_error = (
        final_error
    )

    if final_error <= scene.bodymove_physics_deadzone:
        scene.bodymove_physics_status = "STABLE"
    elif changed:
        scene.bodymove_physics_status = (
            f"SOLVING | {final_error:.3f}m"
        )
    else:
        scene.bodymove_physics_status = (
            f"WAITING | {final_error:.3f}m"
        )

    return changed


# =========================================================
# V0.4 - UNIVERSAL RIG MAPPING
# =========================================================

def armature_bone_names(armature):
    if armature is None or armature.type != "ARMATURE":
        return []
    return [bone.name for bone in armature.data.bones]


def find_bone_fuzzy(armature, candidates):
    """
    Returns the best matching bone name.
    Search order:
    1. exact case-sensitive
    2. exact case-insensitive
    3. suffix case-insensitive
    4. contains case-insensitive
    """
    names = armature_bone_names(armature)

    if not names:
        return ""

    lower_lookup = {
        name.lower(): name
        for name in names
    }

    for candidate in candidates:
        if candidate in names:
            return candidate

    for candidate in candidates:
        found = lower_lookup.get(candidate.lower())
        if found:
            return found

    for candidate in candidates:
        c = candidate.lower()
        for name in names:
            if name.lower().endswith(c):
                return name

    for candidate in candidates:
        c = candidate.lower()
        for name in names:
            if c in name.lower():
                return name

    return ""


def detect_rig_profile(armature):
    names = armature_bone_names(armature)
    lower = [name.lower() for name in names]

    if any(name.startswith("mixamorig:") for name in names):
        return "MIXAMO"

    if (
        any("foot_ik.l" in name for name in lower)
        and any(name == "torso" for name in lower)
    ):
        return "RIGIFY"

    if any(
        token in name
        for name in lower
        for token in (
            "c_root_master",
            "c_pos",
            "c_foot_ik",
            "c_hips",
        )
    ):
        return "AUTO_RIG_PRO"

    return "GENERIC"


def auto_map_rig(scene):
    armature = scene.bodymove_armature_object

    if armature is None or armature.type != "ARMATURE":
        return 0

    profile = detect_rig_profile(armature)
    scene.bodymove_rig_profile = profile

    if profile == "MIXAMO":
        candidates = {
            "root": [
                "mixamorig:Hips",
            ],
            "pelvis": [
                "mixamorig:Hips",
            ],
            "spine": [
                "mixamorig:Spine1",
                "mixamorig:Spine",
            ],
            "chest": [
                "mixamorig:Spine2",
                "mixamorig:Spine1",
            ],
            "head": [
                "mixamorig:Head",
            ],
            "left_foot": [
                "mixamorig:LeftFoot",
            ],
            "right_foot": [
                "mixamorig:RightFoot",
            ],
        }

    elif profile == "RIGIFY":
        candidates = {
            "root": [
                "root",
                "torso",
            ],
            "pelvis": [
                "hips",
                "pelvis",
                "torso",
            ],
            "spine": [
                "torso",
                "spine_fk.001",
                "spine_fk.002",
                "spine",
            ],
            "chest": [
                "chest",
                "chest_parent",
                "spine_fk.003",
            ],
            "head": [
                "head",
            ],
            "left_foot": [
                "foot_ik.L",
                "foot_fk.L",
                "foot.L",
            ],
            "right_foot": [
                "foot_ik.R",
                "foot_fk.R",
                "foot.R",
            ],
        }

    elif profile == "AUTO_RIG_PRO":
        candidates = {
            "root": [
                "c_root_master.x",
                "c_pos",
                "c_root.x",
            ],
            "pelvis": [
                "c_root.x",
                "c_hips.x",
                "c_hips",
                "c_pelvis.x",
            ],
            "spine": [
                "c_spine_fk.01.x",
                "c_spine_01.x",
                "c_spine_1.x",
                "c_spine_01",
            ],
            "chest": [
                "c_chest.x",
                "c_spine_02.x",
                "c_spine_2.x",
            ],
            "head": [
                "c_head.x",
                "c_head",
            ],
            "left_foot": [
                "c_foot_ik.l",
                "c_foot_fk.l",
                "c_foot.l",
            ],
            "right_foot": [
                "c_foot_ik.r",
                "c_foot_fk.r",
                "c_foot.r",
            ],
        }

    else:
        # Generic heuristic.
        candidates = {
            "root": [
                "root",
                "master",
                "global",
                "hips",
                "pelvis",
            ],
            "pelvis": [
                "pelvis",
                "hips",
                "hip",
                "root",
            ],
            "spine": [
                "spine1",
                "spine_01",
                "spine.001",
                "spine",
                "torso",
            ],
            "chest": [
                "chest",
                "spine2",
                "spine_02",
                "spine.002",
                "upperchest",
            ],
            "head": [
                "head",
                "neck_head",
            ],
            "left_foot": [
                "foot_ik.l",
                "leftfoot",
                "foot.l",
                "foot_l",
                "l_foot",
            ],
            "right_foot": [
                "foot_ik.r",
                "rightfoot",
                "foot.r",
                "foot_r",
                "r_foot",
            ],
        }

    mapping = {
        key: find_bone_fuzzy(
            armature,
            value,
        )
        for key, value in candidates.items()
    }

    scene.bodymove_map_root = mapping["root"]
    scene.bodymove_map_pelvis = mapping["pelvis"]
    scene.bodymove_map_spine = mapping["spine"]
    scene.bodymove_map_chest = mapping["chest"]
    scene.bodymove_map_head = mapping["head"]
    scene.bodymove_map_left_foot = mapping["left_foot"]
    scene.bodymove_map_right_foot = mapping["right_foot"]

    # V0.5: extended human roles for full-body response.
    extended = generic_human_bone_map(
        armature
    )

    set_extended_mapping_from_human_map(
        scene,
        extended,
    )

    required_mass_roles = tuple(HUMAN_MASS_PERCENT.keys())
    mapped_mass = sum(bool(extended.get(role)) for role in required_mass_roles)
    missing_mass = [role for role in required_mass_roles if not extended.get(role)]
    missing_controls = [key for key, value in mapping.items() if not value]
    scene.bodymove_map_report = (
        f"{profile}: controls {sum(bool(v) for v in mapping.values())}/7; "
        f"mass bones {mapped_mass}/{len(required_mass_roles)}"
    )
    if missing_controls:
        scene.bodymove_map_report += "; missing controls: " + ", ".join(missing_controls)
    if missing_mass:
        scene.bodymove_map_report += "; missing mass: " + ", ".join(missing_mass)

    # Re-use mapping for v0.3 one-shot Assist.
    if mapping["spine"]:
        scene.bodymove_assist_primary_bone = (
            mapping["spine"]
        )

    if mapping["chest"]:
        scene.bodymove_assist_secondary_bone = (
            mapping["chest"]
        )

    return sum(
        1
        for name in mapping.values()
        if name
    )


class BODYMOVE_OT_AutoMapRig(bpy.types.Operator):
    bl_idname = "bodymove.auto_map_rig"
    bl_label = "Auto Map Rig"
    bl_description = "Détecter le type de rig et proposer les controls BodyMove"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        armature = scene.bodymove_armature_object

        if armature is None or armature.type != "ARMATURE":
            self.report(
                {"ERROR"},
                "Choisissez une Armature"
            )
            return {"CANCELLED"}

        count = auto_map_rig(scene)

        self.report({"INFO"}, scene.bodymove_map_report[:240])

        return {"FINISHED"}


# =========================================================
# V0.4 - AUTOMATIC INFLUENCE SOLVER
# =========================================================

def mapped_response_bones(scene):
    """
    Bones that are allowed to react to COM movement.
    The weights are intentionally configurable.
    """

    mode = scene.bodymove_solver_mode

    response = []

    if mode in {"FULL_BODY", "FULL_BODY_LOCK"}:
        if scene.bodymove_map_pelvis:
            response.append((
                scene.bodymove_map_pelvis,
                scene.bodymove_solver_pelvis_weight,
            ))

    if scene.bodymove_map_spine:
        response.append((
            scene.bodymove_map_spine,
            scene.bodymove_solver_spine_weight,
        ))

    if scene.bodymove_map_chest:
        response.append((
            scene.bodymove_map_chest,
            scene.bodymove_solver_chest_weight,
        ))

    if scene.bodymove_map_head:
        response.append((
            scene.bodymove_map_head,
            scene.bodymove_solver_head_weight,
        ))

    # Remove duplicate bone names while preserving first occurrence.
    seen = set()
    unique = []

    for bone_name, weight in response:
        if not bone_name or bone_name in seen:
            continue

        seen.add(bone_name)
        unique.append((
            bone_name,
            max(0.0, min(1.0, weight)),
        ))

    return unique


def support_role_foot_control(scene, support):
    role = support.get(ROLE, "")

    if role == "LEFT_FOOT":
        return scene.bodymove_map_left_foot

    if role == "RIGHT_FOOT":
        return scene.bodymove_map_right_foot

    return ""


def capture_support_positions(scene):
    data = []

    for support in get_active_supports(scene):
        data.append((
            support,
            support.matrix_world.translation.copy(),
            support_role_foot_control(
                scene,
                support,
            ),
        ))

    return data


def clamp_vector_length(vector, max_length):
    if vector.length <= max_length:
        return vector

    if vector.length <= 1e-12:
        return vector

    result = vector.normalized()
    result *= max_length
    return result


def solver_error(scene):
    correction, target = (
        calculate_balance_correction(scene)
    )

    if correction is None:
        return None, None

    return correction, correction.length


def update_solver_stagnation(scene, error):
    last_error = scene.bodymove_solver_last_error

    if last_error <= 0.0:
        scene.bodymove_solver_stagnation = 0
    elif abs(last_error - error) <= 0.00005:
        scene.bodymove_solver_stagnation += 1
    else:
        scene.bodymove_solver_stagnation = 0

    scene.bodymove_solver_last_error = error


def auto_influence_solver_step(scene, depsgraph):
    """
    One proportional correction step.

    The depsgraph handler may call this repeatedly while the rig is being moved.
    Small clamped steps make the response converge instead of jumping.

    FULL_BODY_LOCK:
    after pelvis/spine/chest reaction, active foot controls receive a
    counter-translation so their Support Point tries to stay planted.
    """

    if not scene.bodymove_auto_influence:
        return False

    if not scene.bodymove_analysis_valid:
        return False

    armature = scene.bodymove_armature_object

    if armature is None or armature.type != "ARMATURE":
        scene.bodymove_solver_status = "Armature invalide"
        return False

    active = get_active_supports(scene)

    if not active:
        scene.bodymove_solver_status = "No Active Support"
        return False

    correction, error = solver_error(scene)

    if correction is None:
        scene.bodymove_solver_status = "No correction target"
        return False

    update_solver_stagnation(
        scene,
        error,
    )

    if (
        scene.bodymove_solver_stagnation
        >= scene.bodymove_solver_stagnation_limit
    ):
        scene.bodymove_auto_influence = False
        scene.bodymove_solver_status = (
            "PAUSED: rig/constraints block correction"
        )
        return False

    deadzone = max(
        scene.bodymove_solver_deadzone,
        0.0001,
    )

    if error <= deadzone:
        scene.bodymove_solver_status = "STABLE"
        scene.bodymove_solver_last_step = 0.0
        scene.bodymove_solver_stagnation = 0
        return False

    if (
        scene.bodymove_solver_only_unbalanced
        and scene.bodymove_balance_status == "BALANCED"
    ):
        scene.bodymove_solver_status = "Balanced"
        return False

    strength = max(
        0.0,
        min(
            1.0,
            scene.bodymove_solver_strength,
        ),
    )

    step = correction * strength

    step = clamp_vector_length(
        step,
        max(
            scene.bodymove_solver_max_step,
            0.001,
        ),
    )

    if step.length <= 1e-8:
        return False

    response = mapped_response_bones(scene)

    if not response:
        scene.bodymove_solver_status = (
            "Map response controls first"
        )
        return False

    support_before = (
        capture_support_positions(scene)
        if (
            scene.bodymove_solver_mode
            == "FULL_BODY_LOCK"
            and scene.bodymove_solver_foot_lock
        )
        else []
    )

    applied = 0

    for bone_name, weight in response:
        if weight <= 0.0:
            continue

        if translate_pose_bone_world(
            armature,
            bone_name,
            step * weight,
        ):
            applied += 1

    if applied == 0:
        scene.bodymove_solver_status = (
            "Response bones locked/invalid"
        )
        return False

    armature.update_tag()

    # Try to evaluate once before optional foot lock.
    try:
        if bpy.context.view_layer:
            bpy.context.view_layer.update()
    except Exception:
        pass

    # Experimental foot pinning for rigs that expose foot controls.
    if support_before:
        for support, wanted_world, foot_bone in support_before:
            if not foot_bone:
                continue

            current_world = (
                support.matrix_world.translation.copy()
            )

            lock_delta = (
                wanted_world - current_world
            )

            lock_delta.z = 0.0

            max_lock = max(
                scene.bodymove_solver_max_step
                * 1.5,
                0.001,
            )

            lock_delta = clamp_vector_length(
                lock_delta,
                max_lock,
            )

            if lock_delta.length > 1e-8:
                translate_pose_bone_world(
                    armature,
                    foot_bone,
                    lock_delta,
                )

    armature.update_tag()

    scene.bodymove_solver_last_step = (
        step.length
    )

    scene.bodymove_solver_status = (
        f"REACTING | error {error:.3f}m"
    )

    return True


# =========================================================
# BALANCE ASSIST
# =========================================================

def get_active_supports(scene):
    return [
        support
        for support in helpers(scene, SUPPORT_POINT)
        if bool(support.get(SUPPORT_ACTIVE, False))
    ]


def active_support_center(scene):
    active = get_active_supports(scene)

    if not active:
        return None

    center = Vector((0.0, 0.0, 0.0))

    for support in active:
        center += support.matrix_world.translation.copy()

    center /= len(active)
    return center


def translate_pose_bone_world(armature, bone_name, world_delta):
    """
    Déplace un PoseBone par une translation exprimée en world-space.
    Le delta est converti en espace Armature avant d'être appliqué.
    """

    if (
        armature is None
        or armature.type != "ARMATURE"
        or not bone_name
    ):
        return False

    pose_bone = armature.pose.bones.get(bone_name)

    if pose_bone is None:
        return False

    if world_delta.length <= 1e-12:
        return True

    try:
        delta_armature = (
            armature.matrix_world.inverted().to_3x3()
            @ world_delta
        )

        pose_bone.matrix = (
            Matrix.Translation(delta_armature)
            @ pose_bone.matrix
        )

        armature.update_tag()
        return True

    except Exception:
        return False


def calculate_balance_correction(scene):
    """
    Retourne le déplacement horizontal qui rapproche le COM
    du centre des supports actifs.
    """

    target = active_support_center(scene)

    if target is None:
        return None, None

    projection = find_analysis_helper(
        scene,
        COM_PROJECTION,
    )

    if projection is not None:
        current = (
            projection.matrix_world.translation.copy()
        )
    else:
        current = Vector((
            scene.bodymove_com_x,
            scene.bodymove_com_y,
            target.z,
        ))

    correction = Vector((
        target.x - current.x,
        target.y - current.y,
        0.0,
    ))

    return correction, target


class BODYMOVE_OT_ApplyBalanceAssist(bpy.types.Operator):
    bl_idname = "bodymove.apply_balance_assist"
    bl_label = "Appliquer Balance Assist"
    bl_description = "Déplacer les control bones choisis pour rapprocher le COM des supports actifs"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        global _LIVE_GUARD

        scene = context.scene
        armature = scene.bodymove_armature_object

        if (
            armature is None
            or armature.type != "ARMATURE"
        ):
            self.report(
                {"ERROR"},
                "Armature invalide"
            )
            return {"CANCELLED"}

        if not scene.bodymove_analysis_valid:
            self.report(
                {"ERROR"},
                "Calculez GLOBAL COM d'abord"
            )
            return {"CANCELLED"}

        depsgraph = context.evaluated_depsgraph_get()

        # Rafraîchir support / COM avant solver.
        try:
            _LIVE_GUARD = True
            compute_analysis(
                scene,
                depsgraph,
                set_baseline=False,
            )
        finally:
            _LIVE_GUARD = False

        active = get_active_supports(scene)

        if not active:
            self.report(
                {"ERROR"},
                "Aucun Support Point actif"
            )
            return {"CANCELLED"}

        if (
            scene.bodymove_assist_only_unbalanced
            and scene.bodymove_balance_status == "BALANCED"
        ):
            self.report(
                {"INFO"},
                "Le corps est déjà BALANCED"
            )
            return {"CANCELLED"}

        correction, target = calculate_balance_correction(
            scene
        )

        if correction is None:
            self.report(
                {"ERROR"},
                "Correction impossible"
            )
            return {"CANCELLED"}

        distance = correction.length

        if distance <= 1e-6:
            self.report(
                {"INFO"},
                "COM déjà centré sur les supports"
            )
            return {"CANCELLED"}

        max_distance = max(
            scene.bodymove_assist_max_correction,
            0.001,
        )

        if distance > max_distance:
            correction.normalize()
            correction *= max_distance

        correction *= max(
            0.0,
            min(
                1.0,
                scene.bodymove_assist_strength,
            ),
        )

        primary_name = (
            scene.bodymove_assist_primary_bone
        )
        secondary_name = (
            scene.bodymove_assist_secondary_bone
        )

        if not primary_name and not secondary_name:
            self.report(
                {"ERROR"},
                "Choisissez au moins un Assist Control Bone"
            )
            return {"CANCELLED"}

        applied = 0

        try:
            _LIVE_GUARD = True

            if primary_name:
                delta = (
                    correction
                    * max(
                        0.0,
                        min(
                            1.0,
                            scene.bodymove_assist_primary_weight,
                        ),
                    )
                )

                if translate_pose_bone_world(
                    armature,
                    primary_name,
                    delta,
                ):
                    applied += 1

            if (
                secondary_name
                and secondary_name != primary_name
            ):
                delta = (
                    correction
                    * max(
                        0.0,
                        min(
                            1.0,
                            scene.bodymove_assist_secondary_weight,
                        ),
                    )
                )

                if translate_pose_bone_world(
                    armature,
                    secondary_name,
                    delta,
                ):
                    applied += 1

        finally:
            _LIVE_GUARD = False

        if applied == 0:
            self.report(
                {"ERROR"},
                "Aucun control bone pa t ka deplase"
            )
            return {"CANCELLED"}

        # Fòse evaluated rig / markers yo rafraîchi.
        context.view_layer.update()

        try:
            _LIVE_GUARD = True
            compute_analysis(
                scene,
                context.evaluated_depsgraph_get(),
                set_baseline=False,
            )
        finally:
            _LIVE_GUARD = False

        scene.bodymove_last_assist_x = correction.x
        scene.bodymove_last_assist_y = correction.y
        scene.bodymove_last_assist_distance = (
            correction.length
        )

        tag_redraw()

        self.report(
            {"INFO"},
            (
                f"Balance Assist: "
                f"{correction.length:.3f} m"
            ),
        )

        return {"FINISHED"}


# =========================================================
# LIVE HANDLERS
# =========================================================

def live_update(scene, depsgraph):
    global _LIVE_GUARD

    if reactive.busy() or getattr(scene, 'bodymove_reactive_enabled', False):
        return

    if _LIVE_GUARD:
        return

    if not scene.bodymove_initialized:
        return

    if not scene.bodymove_analysis_valid:
        return

    if not scene.bodymove_live_analysis:
        return

    try:
        _LIVE_GUARD = True

        compute_analysis(
            scene,
            depsgraph,
            set_baseline=False,
        )

        # V0.5: iterative target-COM physics solver.
        changed = physics_solver_step(
            scene,
            depsgraph,
        )

        if changed:
            try:
                armature = scene.bodymove_armature_object
                if armature is not None:
                    armature.update_tag()
            except Exception:
                pass

        tag_redraw()
    except Exception:
        # Handler la pa dwe kraze Blender si yon objè efase.
        logging.getLogger(__name__).exception('BodyMove live update failed')
    finally:
        _LIVE_GUARD = False


@persistent
def bodymove_depsgraph_update(scene, depsgraph):
    live_update(
        scene,
        depsgraph,
    )


@persistent
def bodymove_frame_change(scene, depsgraph):
    live_update(
        scene,
        depsgraph,
    )


# =========================================================
# INITIALIZE BODY
# =========================================================

class BODYMOVE_OT_Initialize(bpy.types.Operator):
    bl_idname = "bodymove.initialize"
    bl_label = "Initialiser Body"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        source = scene.bodymove_source_object

        if source is None:
            source = context.active_object
            scene.bodymove_source_object = source

        if source is None or source.type != "MESH":
            self.report(
                {"ERROR"},
                "Sélectionnez un Source Mesh"
            )
            return {"CANCELLED"}

        # Auto-detect armature if possible.
        armature = scene.bodymove_armature_object

        if armature is None:
            try:
                armature = source.find_armature()
            except Exception:
                armature = None

            if armature is not None:
                scene.bodymove_armature_object = armature

        if (
            armature is None
            or armature.type != "ARMATURE"
        ):
            self.report(
                {"ERROR"},
                "BodyMove v0.2 bezwen yon Armature"
            )
            return {"CANCELLED"}

        ensure_collection(scene)

        scene.bodymove_initialized = True
        scene.bodymove_analysis_valid = False
        scene.bodymove_detail_name = ""

        # V0.4: propose a generic rig mapping immediately.
        auto_map_rig(scene)

        self.report(
            {"INFO"},
            (
                f"Body: {source.name} | "
                f"Rig: {armature.name}"
            ),
        )

        return {"FINISHED"}


# =========================================================
# PLACE POINT
# =========================================================

class BODYMOVE_OT_PlacePoint(bpy.types.Operator):
    bl_idname = "bodymove.place_point"
    bl_label = "Placer Point"
    bl_options = {"REGISTER", "UNDO"}

    point_type: EnumProperty(
        items=(
            ("MASS", "Mass Point", ""),
            ("SUPPORT", "Support Point", ""),
        ),
        default="MASS",
    )

    def invoke(self, context, event):
        scene = context.scene

        if not scene.bodymove_initialized:
            self.report(
                {"ERROR"},
                "Initialisez Body d'abord"
            )
            return {"CANCELLED"}

        if context.area.type != "VIEW_3D":
            return {"CANCELLED"}

        armature = scene.bodymove_armature_object

        if armature is None:
            self.report(
                {"ERROR"},
                "Armature manquant"
            )
            return {"CANCELLED"}

        bone_name = (
            scene.bodymove_mass_bone_name
            if self.point_type == "MASS"
            else scene.bodymove_support_bone_name
        )

        if (
            not bone_name
            or bone_name not in armature.data.bones
        ):
            self.report(
                {"ERROR"},
                "Choisissez un Bone valide"
            )
            return {"CANCELLED"}

        context.window.cursor_modal_set(
            "CROSSHAIR"
        )

        context.window_manager.modal_handler_add(
            self
        )

        self.report(
            {"INFO"},
            "Cliquez sur le Source Mesh. ESC annule."
        )

        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {
            "ESC",
            "RIGHTMOUSE",
        }:
            context.window.cursor_modal_restore()
            return {"CANCELLED"}

        if (
            event.type != "LEFTMOUSE"
            or event.value != "PRESS"
        ):
            return {"RUNNING_MODAL"}

        scene = context.scene
        source = scene.bodymove_source_object
        armature = scene.bodymove_armature_object

        hit = raycast_source(
            context,
            (
                event.mouse_region_x,
                event.mouse_region_y,
            ),
            source,
        )

        if hit is None:
            self.report(
                {"WARNING"},
                "Cliquez directement sur le Source Mesh"
            )
            return {"RUNNING_MODAL"}

        location, normal, face_index = hit

        location = (
            location
            + normal
            * scene.bodymove_surface_offset
        )

        if self.point_type == "MASS":
            category = scene.bodymove_mass_category
            label = label_from(
                MASS_CATEGORIES,
                category,
            )

            if category == "CUSTOM":
                label = (
                    scene.bodymove_custom_mass_name.strip()
                    or "Custom"
                )

            mass = scene.bodymove_mass_value

            if mass <= 0.0:
                context.window.cursor_modal_restore()
                return {"CANCELLED"}

            bone_name = (
                scene.bodymove_mass_bone_name
            )

            obj = create_empty(
                scene,
                unique_name("BM_MP", label),
                MASS_POINT,
                location,
                "SPHERE",
                scene.bodymove_mass_point_size,
            )

            obj[CATEGORY] = category
            obj[MASS] = float(mass)

            if not bind_to_bone(
                obj,
                armature,
                bone_name,
            ):
                bpy.data.objects.remove(
                    obj,
                    do_unlink=True,
                )

                context.window.cursor_modal_restore()

                self.report(
                    {"ERROR"},
                    "Impossible de lier au Bone"
                )

                return {"CANCELLED"}

            obj.name = (
                f"MP | {label} | "
                f"{mass:.2f}kg | {bone_name}"
            )

        else:
            role = scene.bodymove_support_role
            label = label_from(
                SUPPORT_ROLES,
                role,
            )

            if role == "CUSTOM":
                label = (
                    scene.bodymove_custom_support_name.strip()
                    or "Custom Support"
                )

            radius = max(
                scene.bodymove_support_radius,
                0.01,
            )

            bone_name = (
                scene.bodymove_support_bone_name
            )

            obj = create_empty(
                scene,
                unique_name("BM_SP", label),
                SUPPORT_POINT,
                location,
                "CIRCLE",
                radius,
            )

            obj[ROLE] = role
            obj[RADIUS] = float(radius)
            obj[SUPPORT_ACTIVE] = True

            if not bind_to_bone(
                obj,
                armature,
                bone_name,
            ):
                bpy.data.objects.remove(
                    obj,
                    do_unlink=True,
                )

                context.window.cursor_modal_restore()

                self.report(
                    {"ERROR"},
                    "Impossible de lier au Bone"
                )

                return {"CANCELLED"}

            obj.name = (
                f"SP | {label} | {bone_name}"
            )

        scene.bodymove_detail_name = obj.name
        scene.bodymove_analysis_valid = False

        select_only(
            context,
            obj,
        )

        context.window.cursor_modal_restore()

        return {"FINISHED"}


# =========================================================
# CALCULATE / BASELINE
# =========================================================

class BODYMOVE_OT_Calculate(bpy.types.Operator):
    bl_idname = "bodymove.calculate"
    bl_label = "Calculer"
    bl_options = {"REGISTER"}

    def execute(self, context):
        scene = context.scene

        if not helpers(
            scene,
            MASS_POINT,
        ):
            self.report(
                {"ERROR"},
                "Ajoutez des Mass Points"
            )
            return {"CANCELLED"}

        depsgraph = (
            context.evaluated_depsgraph_get()
        )

        success = compute_analysis(
            scene,
            depsgraph,
            set_baseline=True,
        )

        if not success:
            self.report(
                {"ERROR"},
                "Analyse impossible"
            )
            return {"CANCELLED"}

        scene.bodymove_analysis_valid = True

        scene.bodymove_solver_last_error = 0.0
        scene.bodymove_solver_stagnation = 0
        scene.bodymove_solver_last_step = 0.0
        scene.bodymove_solver_status = "READY"

        tag_redraw()

        self.report(
            {"INFO"},
            (
                "GLOBAL COM affiché. "
                "Live Analysis activable."
            ),
        )

        return {"FINISHED"}


# =========================================================
# DETAIL / DELETE
# =========================================================

class BODYMOVE_OT_Detail(bpy.types.Operator):
    bl_idname = "bodymove.detail"
    bl_label = "Detail"

    target_name: StringProperty()

    def execute(self, context):
        obj = bpy.data.objects.get(
            self.target_name
        )

        if obj is None:
            return {"CANCELLED"}

        context.scene.bodymove_detail_name = (
            obj.name
        )

        select_only(
            context,
            obj,
        )

        return {"FINISHED"}


class BODYMOVE_OT_DeletePoint(bpy.types.Operator):
    bl_idname = "bodymove.delete_point"
    bl_label = "Delete Point"
    bl_options = {"REGISTER", "UNDO"}

    target_name: StringProperty()

    def execute(self, context):
        scene = context.scene

        obj = bpy.data.objects.get(
            self.target_name
        )

        if obj is None:
            return {"CANCELLED"}

        if helper_kind(obj) not in {
            MASS_POINT,
            SUPPORT_POINT,
        }:
            return {"CANCELLED"}

        bpy.data.objects.remove(
            obj,
            do_unlink=True,
        )

        scene.bodymove_detail_name = ""
        scene.bodymove_analysis_valid = False

        return {"FINISHED"}


# =========================================================
# RESET
# =========================================================

class BODYMOVE_OT_ResetAnalysis(bpy.types.Operator):
    bl_idname = "bodymove.reset_analysis"
    bl_label = "Reset Analysis"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene

        for wanted_kind in _ANALYSIS_KINDS:
            for obj in list(
                helpers(
                    scene,
                    wanted_kind,
                )
            ):
                data = (
                    obj.data
                    if obj.type == "MESH"
                    else None
                )

                bpy.data.objects.remove(
                    obj,
                    do_unlink=True,
                )

                if (
                    data is not None
                    and data.users == 0
                ):
                    bpy.data.meshes.remove(data)

        scene.bodymove_analysis_valid = False
        scene.bodymove_com_shift_live = 0.0

        return {"FINISHED"}


class BODYMOVE_OT_ClearSetup(bpy.types.Operator):
    bl_idname = "bodymove.clear_setup"
    bl_label = "Clear Setup"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene

        try:
            remove_ghost_preview(scene)
        except Exception:
            pass

        col = get_collection(scene)

        if col is not None:
            for obj in list(col.objects):
                if is_helper(obj):
                    data = (
                        obj.data
                        if obj.type == "MESH"
                        else None
                    )

                    bpy.data.objects.remove(
                        obj,
                        do_unlink=True,
                    )

                    if (
                        data is not None
                        and data.users == 0
                    ):
                        bpy.data.meshes.remove(data)

        scene.bodymove_analysis_valid = False
        scene.bodymove_detail_name = ""

        return {"FINISHED"}


class BODYMOVE_OT_GenerateWalk(bpy.types.Operator):
    bl_idname = "bodymove.generate_walk"
    bl_label = "Generate Walk"
    bl_description = "Create a basic in-place walk cycle for a mapped Auto-Rig Pro IK rig"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        rig = scene.bodymove_armature_object
        if rig is None or rig.type != "ARMATURE":
            self.report({"ERROR"}, "Choose an armature and run Quick Human Setup first")
            return {"CANCELLED"}
        if scene.bodymove_rig_profile != "AUTO_RIG_PRO":
            self.report({"ERROR"}, "The first walk generator currently supports Auto-Rig Pro")
            return {"CANCELLED"}

        feet = [
            rig.pose.bones.get(scene.bodymove_map_left_foot),
            rig.pose.bones.get(scene.bodymove_map_right_foot),
        ]
        pelvis = rig.pose.bones.get(scene.bodymove_map_pelvis)
        if pelvis is None or any(bone is None for bone in feet):
            self.report({"ERROR"}, "Map the pelvis and both foot IK controls first")
            return {"CANCELLED"}
        if any("c_foot_ik" not in bone.name.lower() for bone in feet):
            self.report({"ERROR"}, "Map c_foot_ik.l and c_foot_ik.r for this walk generator")
            return {"CANCELLED"}

        stride = max(float(scene.bodymove_walk_stride), 0.05)
        speed = max(float(scene.bodymove_walk_speed), 0.05)
        cycles = max(int(scene.bodymove_walk_cycles), 1)
        fps = scene.render.fps / max(scene.render.fps_base, 1e-6)
        cycle_frames = max(4, int(round((2.0 * stride / speed) * fps)))
        start = int(scene.frame_start)
        end = start + cycle_frames * cycles

        axis = {
            "X": Vector((1.0, 0.0, 0.0)),
            "-X": Vector((-1.0, 0.0, 0.0)),
            "Y": Vector((0.0, 1.0, 0.0)),
            "-Y": Vector((0.0, -1.0, 0.0)),
        }[scene.bodymove_walk_forward_axis]
        rig_basis = rig.matrix_world.to_3x3()
        forward_scale = max((rig_basis @ axis).length, 1e-8)
        vertical_scale = max((rig_basis @ Vector((0.0, 0.0, 1.0))).length, 1e-8)
        stride_local = stride / forward_scale

        old_frame = scene.frame_current
        old_subframe = scene.frame_subframe
        anim = rig.animation_data_create()
        old_action = anim.action
        old_slot = anim.action_slot
        previous_generated = bool(old_action and old_action.get("_bodymove_walk_generated"))
        base_action = scene.bodymove_walk_base_action if previous_generated else old_action
        base_slot_id = scene.bodymove_walk_base_slot if previous_generated else (
            old_slot.identifier if old_slot else ""
        )
        if not previous_generated:
            scene.bodymove_walk_base_action = old_action
            scene.bodymove_walk_base_slot = base_slot_id

        try:
            scene.frame_set(start)
            context.view_layer.update()
            bone_matrices = {
                bone.name: bone.matrix.copy()
                for bone in [pelvis, *feet]
            }
            height = max(float(rig.dimensions.z), 0.5)
            if scene.bodymove_source_object is not None:
                height = max(float(scene.bodymove_source_object.dimensions.z), 0.5)

            if base_action is not None:
                action = base_action.copy()
                slot = next(
                    (item for item in action.slots if item.identifier == base_slot_id),
                    action.slots[0] if len(action.slots) else None,
                )
                if slot is None:
                    slot = action.slots.new(id_type="OBJECT", name=rig.name)
            else:
                action = bpy.data.actions.new(rig.name + "_BodyMove_Walk")
                slot = action.slots.new(id_type="OBJECT", name=rig.name)
                layer = action.layers.new("BodyMove Walk")
                strip = layer.strips.new(type="KEYFRAME")
                strip.channelbag(slot, ensure=True)

            action["_bodymove_walk_generated"] = True
            action["_bodymove_walk_source_action"] = base_action.name if base_action else ""
            rig.animation_data.action = action
            rig.animation_data.action_slot = slot

            swing = 0.38
            stance = 1.0 - swing
            lift_height = height * 0.075 / vertical_scale
            pelvis_bob = height * 0.018 / vertical_scale

            for frame in range(start, end + 1):
                scene.frame_set(frame)
                elapsed = frame - start
                gait_phase = (elapsed / cycle_frames) % 1.0
                pelvis_matrix = bone_matrices[pelvis.name].copy()
                pelvis_matrix.translation += Vector((0.0, 0.0, pelvis_bob * (1.0 - math.cos(2.0 * math.pi * gait_phase))))
                pelvis.matrix = pelvis_matrix
                context.view_layer.update()

                for side_index, foot in enumerate(feet):
                    phase = (gait_phase + (0.5 if side_index else 0.0)) % 1.0
                    if phase < stance:
                        travel = phase / stance
                        forward = (0.5 - travel) * stride_local
                        lift = 0.0
                    else:
                        travel = (phase - stance) / swing
                        eased = travel * travel * (3.0 - 2.0 * travel)
                        forward = (-0.5 + eased) * stride_local
                        lift = math.sin(math.pi * travel) * lift_height

                    foot_matrix = bone_matrices[foot.name].copy()
                    foot_matrix.translation += axis * forward
                    foot_matrix.translation.z += lift
                    foot.matrix = foot_matrix

                    foot.keyframe_insert(data_path="location", frame=frame, group=foot.name)
                    if foot.rotation_mode == "QUATERNION":
                        foot.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=foot.name)
                    else:
                        foot.keyframe_insert(data_path="rotation_euler", frame=frame, group=foot.name)

                pelvis.keyframe_insert(data_path="location", frame=frame, group=pelvis.name)
                if pelvis.rotation_mode == "QUATERNION":
                    pelvis.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=pelvis.name)
                else:
                    pelvis.keyframe_insert(data_path="rotation_euler", frame=frame, group=pelvis.name)

            scene.frame_set(start)
            scene.bodymove_walk_action_name = action.name
            if previous_generated and old_action != action and old_action.users == 0:
                bpy.data.actions.remove(old_action)
            self.report({"INFO"}, f"Walk cycle generated: frames {start}-{end}")
            return {"FINISHED"}

        except Exception as exc:
            rig.animation_data.action = old_action
            if old_slot is not None:
                rig.animation_data.action_slot = old_slot
            if 'action' in locals() and action.users == 0:
                bpy.data.actions.remove(action)
            if not previous_generated:
                scene.bodymove_walk_base_action = None
                scene.bodymove_walk_base_slot = ""
            scene.frame_set(old_frame, subframe=old_subframe)
            self.report({"ERROR"}, f"Walk generation failed: {exc}")
            return {"CANCELLED"}


class BODYMOVE_OT_RestoreWalkAction(bpy.types.Operator):
    bl_idname = "bodymove.restore_walk_action"
    bl_label = "Restore Previous Action"
    bl_description = "Restore the action that was active before generating the walk"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        rig = scene.bodymove_armature_object
        if rig is None or rig.type != "ARMATURE" or rig.animation_data is None:
            self.report({"ERROR"}, "No animated armature is available")
            return {"CANCELLED"}
        current = rig.animation_data.action
        if current is None or not current.get("_bodymove_walk_generated"):
            self.report({"ERROR"}, "The active action is not a BodyMove walk")
            return {"CANCELLED"}

        rig.animation_data.action = scene.bodymove_walk_base_action
        if scene.bodymove_walk_base_action is not None:
            rig.animation_data.action_slot = next(
                (item for item in scene.bodymove_walk_base_action.slots
                 if item.identifier == scene.bodymove_walk_base_slot),
                scene.bodymove_walk_base_action.slots[0]
                if len(scene.bodymove_walk_base_action.slots) else None,
            )
        scene.bodymove_walk_base_action = None
        scene.bodymove_walk_base_slot = ""
        scene.bodymove_walk_action_name = ""
        if current.users == 0:
            bpy.data.actions.remove(current)
        self.report({"INFO"}, "Previous action restored")
        return {"FINISHED"}


# =========================================================
# UI
# =========================================================

class BODYMOVE_PT_Main(bpy.types.Panel):
    bl_label = "BodyMove_test v0.16"
    bl_idname = "BODYMOVE_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "BodyMove"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        armature = (
            scene.bodymove_armature_object
        )

        # -------------------------------------------------
        # 1. QUICK SETUP
        # -------------------------------------------------

        quick = layout.box()

        quick.label(
            text="1. Configuration",
            icon="ARMATURE_DATA",
        )

        quick.prop(
            scene,
            "bodymove_body_weight",
            text="Body Weight",
        )

        row = quick.row()
        row.scale_y = 1.55

        row.operator(
            "bodymove.quick_human_setup",
            text="QUICK HUMAN SETUP",
            icon="PLAY",
        )

        if scene.bodymove_initialized:
            quick.label(
                text=(
                    f"Rig: "
                    f"{scene.bodymove_rig_profile}"
                ),
                icon="CHECKMARK",
            )

            quick.label(
                text=(
                    f"Mass Points: "
                    f"{len(helpers(scene, MASS_POINT))} | "
                    f"Supports: "
                    f"{len(helpers(scene, SUPPORT_POINT))}"
                )
            )

        # -------------------------------------------------
        # 2. COM + FULCRUM STATUS
        # -------------------------------------------------

        reactive.draw(layout, scene)
        balance_preview.draw(layout, scene)

        mass_points = helpers(
            scene,
            MASS_POINT,
        )

        if mass_points:
            analysis = layout.box()

            analysis.label(
                text="2. COM & Fulcrums",
                icon="PIVOT_MEDIAN",
            )

            row = analysis.row()
            row.scale_y = 1.25

            row.operator(
                "bodymove.calculate",
                text="RECALCULER",
                icon="PHYSICS",
            )

            analysis.prop(
                scene,
                "bodymove_live_analysis",
                text="Live Analysis",
            )

            if scene.bodymove_analysis_valid:
                left_support = support_by_role(
                    scene,
                    "LEFT_FOOT",
                )

                right_support = support_by_role(
                    scene,
                    "RIGHT_FOOT",
                )

                row = analysis.row(
                    align=True
                )

                if left_support is not None:
                    left_active = bool(
                        left_support.get(
                            SUPPORT_ACTIVE,
                            False,
                        )
                    )

                    row.label(
                        text=(
                            "Left: ACTIVE"
                            if left_active
                            else "Left: AIR"
                        ),
                        icon=(
                            "CHECKMARK"
                            if left_active
                            else "CANCEL"
                        ),
                    )

                if right_support is not None:
                    right_active = bool(
                        right_support.get(
                            SUPPORT_ACTIVE,
                            False,
                        )
                    )

                    row.label(
                        text=(
                            "Right: ACTIVE"
                            if right_active
                            else "Right: AIR"
                        ),
                        icon=(
                            "CHECKMARK"
                            if right_active
                            else "CANCEL"
                        ),
                    )

                analysis.label(
                    text=(
                        f"Support: "
                        f"{scene.bodymove_support_mode}"
                    )
                )

                analysis.label(
                    text=(
                        f"Balance: "
                        f"{scene.bodymove_balance_status}"
                    ),
                    icon=(
                        "CHECKMARK"
                        if scene.bodymove_balance_status
                        == "BALANCED"
                        else "ERROR"
                    ),
                )

                analysis.label(
                    text=(
                        f"Current COM: "
                        f"{scene.bodymove_com_x:.2f}, "
                        f"{scene.bodymove_com_y:.2f}, "
                        f"{scene.bodymove_com_z:.2f}"
                    )
                )

                analysis.label(
                    text=(
                        f"Target COM: "
                        f"{scene.bodymove_target_com_x:.2f}, "
                        f"{scene.bodymove_target_com_y:.2f}, "
                        f"{scene.bodymove_target_com_z:.2f}"
                    ),
                    icon="TRACKING",
                )

        # -------------------------------------------------
        # 4. LIVE POSE ASSIST
        # Keep V5 solver, but hide it by default.
        # -------------------------------------------------

        if scene.bodymove_analysis_valid:
            live = layout.box()

            live.prop(
                scene,
                "bodymove_show_live_pose_assist",
                text="Live Pose Assist",
            )

            if scene.bodymove_show_live_pose_assist:
                live.prop(
                    scene,
                    "bodymove_physics_enabled",
                    text="AUTO PHYSICS LIVE",
                )

                live.label(
                    text=(
                        f"Status: "
                        f"{scene.bodymove_physics_status}"
                    )
                )

                live.label(
                    text=(
                        f"COM Error: "
                        f"{scene.bodymove_physics_error:.3f} m"
                    )
                )

        # -------------------------------------------------
        # 5. ADVANCED RIG/MASS SETUP
        # -------------------------------------------------

        advanced_setup = layout.box()

        advanced_setup.prop(
            scene,
            "bodymove_show_advanced_setup",
            text="Rig / Mass Advanced",
        )

        if scene.bodymove_show_advanced_setup:
            box = advanced_setup.box()

            box.prop(
                scene,
                "bodymove_source_object",
                text="Source Mesh",
            )

            box.prop(
                scene,
                "bodymove_armature_object",
                text="Armature",
            )

            box.prop(
                scene,
                "bodymove_ground_object",
                text="Ground",
            )

            box.prop(
                scene,
                "bodymove_contact_tolerance",
                text="Contact Tolerance",
            )

            box.prop(
                scene,
                "bodymove_use_support_stability",
                text="Fulcrum Stability",
            )

            if scene.bodymove_use_support_stability:
                box.prop(
                    scene,
                    "bodymove_support_motion_tolerance",
                    text="Motion Tolerance",
                )

            row = box.row(
                align=True
            )

            row.operator(
                "bodymove.initialize",
                text="INITIALISER",
            )

            row.operator(
                "bodymove.auto_map_rig",
                text="AUTO MAP",
            )

            if (
                armature is not None
                and armature.type
                == "ARMATURE"
            ):
                mapping = advanced_setup.box()

                mapping.label(
                    text="Rig Mapping"
                )
                mapping.label(
                    text=scene.bodymove_map_report,
                    icon="INFO",
                )

                for prop_name, label in (
                    (
                        "bodymove_map_pelvis",
                        "Pelvis",
                    ),
                    (
                        "bodymove_map_spine",
                        "Spine",
                    ),
                    (
                        "bodymove_map_chest",
                        "Chest",
                    ),
                    (
                        "bodymove_map_head",
                        "Head",
                    ),
                    (
                        "bodymove_map_left_thigh",
                        "Left Thigh",
                    ),
                    (
                        "bodymove_map_right_thigh",
                        "Right Thigh",
                    ),
                    (
                        "bodymove_map_left_shin",
                        "Left Shin",
                    ),
                    (
                        "bodymove_map_right_shin",
                        "Right Shin",
                    ),
                    (
                        "bodymove_map_left_foot",
                        "Left Foot",
                    ),
                    (
                        "bodymove_map_right_foot",
                        "Right Foot",
                    ),
                ):
                    mapping.prop_search(
                        scene,
                        prop_name,
                        armature.data,
                        "bones",
                        text=label,
                    )

        # -------------------------------------------------
        # TOOLS
        # -------------------------------------------------

        if mass_points:
            tools = layout.box()

            tools.label(
                text="Tools",
                icon="TOOL_SETTINGS",
            )

            row = tools.row(
                align=True
            )

            row.operator(
                "bodymove.reset_analysis",
                text="Reset Analysis",
            )

            row.operator(
                "bodymove.clear_setup",
                text="Clear Setup",
            )


class BODYMOVE_PT_Locomotion(bpy.types.Panel):
    """Motion tools nested inside the existing BodyMove sidebar panel."""
    bl_label = "Locomotion"
    bl_idname = "BODYMOVE_PT_locomotion"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "BodyMove"
    bl_parent_id = "BODYMOVE_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        armature = scene.bodymove_armature_object

        walk = layout.box()
        walk.label(text="In-Place Walk Cycle", icon="ANIM_DATA")
        columns = walk.row(align=True).column_flow(columns=2, align=True)
        columns.prop(scene, "bodymove_walk_speed", text="Speed")
        columns.prop(scene, "bodymove_walk_stride", text="Stride")
        columns.prop(scene, "bodymove_walk_cycles", text="Cycles")
        columns.prop(scene, "bodymove_walk_forward_axis", text="Forward")
        cadence = 60.0 * scene.bodymove_walk_speed / max(scene.bodymove_walk_stride, 0.05)
        walk.label(text=f"Cadence: {cadence:.0f} steps/min")
        walk.label(text="Foot IK targets with pelvis bounce")
        row = walk.row(align=True)
        row.operator("bodymove.generate_walk", text="Generate Walk", icon="PLAY")
        restore = row.row(align=True)
        restore.enabled = bool(scene.bodymove_walk_action_name)
        restore.operator("bodymove.restore_walk_action", text="Restore", icon="LOOP_BACK")
        if scene.bodymove_walk_action_name:
            walk.label(text=f"Action: {scene.bodymove_walk_action_name}", icon="ACTION")

        intro = layout.box()
        intro.label(text="Motion Preview & Bake", icon="ANIM_DATA")
        intro.label(text="Apply physics response to an existing animation")

        # -------------------------------------------------
        # 3. AUTOPHYSICS INTERVAL
        # -------------------------------------------------

        if scene.bodymove_analysis_valid:
            auto = layout.box()

            auto.label(
                text="3. AutoPhysics Interval",
                icon="ACTION",
            )

            auto.prop(
                scene,
                "bodymove_autophysics_use_scene_range",
                text="Use Scene Range",
            )

            if scene.bodymove_autophysics_use_scene_range:
                auto.label(
                    text=(
                        f"Frames: "
                        f"{scene.frame_start}"
                        f" - {scene.frame_end}"
                    )
                )

            else:
                row = auto.row(
                    align=True
                )

                row.prop(
                    scene,
                    "bodymove_autophysics_start",
                    text="Start",
                )

                row.prop(
                    scene,
                    "bodymove_autophysics_end",
                    text="End",
                )

            auto.separator()

            settings = auto.column_flow(columns=2, align=True)
            settings.prop(scene, "bodymove_interval_physics_strength", text="Physics Strength")
            settings.prop(scene, "bodymove_pose_preservation", text="Pose Preservation")
            settings.prop(scene, "bodymove_smooth_trajectory", text="Smooth Trajectory")
            settings.prop(scene, "bodymove_smooth_rotation", text="Smooth Rotation")
            settings.prop(scene, "bodymove_compensation_motion", text="Compensation Motion")
            settings.prop(scene, "bodymove_motion_separation", text="Motion Separation")
            settings.prop(scene, "bodymove_secondary_motion", text="Secondary Motion")
            settings.prop(scene, "bodymove_segment_dynamics", text="Segment Reactions")
            if scene.bodymove_segment_dynamics:
                settings.prop(scene, 'bodymove_reaction_time', text='Reaction Time (s)')
                settings.prop(scene, 'bodymove_reaction_frequency', text='Spring Frequency (Hz)')
                settings.prop(scene, 'bodymove_reaction_damping', text='Damping')
            if scene.bodymove_preview_ready and scene.bodymove_contact_fallbacks:
                auto.label(text=f"Contact preserved: {scene.bodymove_contact_fallbacks} frames kept original", icon='INFO')
            if armature is not None:
                reason = interval.apply_block_reason(armature)
                if reason:
                    auto.label(text='Apply needs a baked FK rig (constraints/NLA/drivers)', icon='INFO')
            auto.separator()

            row = auto.row()
            row.scale_y = 1.55

            row.operator(
                "bodymove.preview_autophysics",
                text="PREVIEW PHYSICS GHOST",
                icon="PLAY",
            )

            auto.label(
                text=(
                    f"Status: "
                    f"{scene.bodymove_preview_status}"
                )
            )

            if scene.bodymove_preview_ready:
                row = auto.row(
                    align=True
                )

                row.operator(
                    "bodymove.apply_autophysics",
                    text="APPLY PHYSICS",
                    icon="CHECKMARK",
                )

                row.operator(
                    "bodymove.clear_autophysics_preview",
                    text="CLEAR",
                    icon="TRASH",
                )

            if scene.bodymove_original_action_name:
                auto.operator(
                    "bodymove.restore_original_action",
                    text="RESTORE ORIGINAL ACTION",
                    icon="LOOP_BACK",
                )

            auto.prop(
                scene,
                "bodymove_show_autophysics_advanced",
                text="Advanced",
            )

            if scene.bodymove_show_autophysics_advanced:
                advanced = auto.box()

                advanced.prop(
                    scene,
                    "bodymove_autophysics_substeps",
                    text="Subframes",
                )

                advanced.prop(
                    scene,
                    "bodymove_smoothing_radius",
                    text="Smoothing Radius",
                )

                advanced.prop(
                    scene,
                    "bodymove_edge_blend_frames",
                    text="Edge Blend",
                )

                advanced.prop(
                    scene,
                    "bodymove_autophysics_max_frames",
                    text="Max Preview Frames",
                )

                advanced.separator()

                advanced.label(
                    text="Body Response"
                )

                advanced.prop(
                    scene,
                    "bodymove_physics_pelvis_translation",
                    text="Pelvis Translation",
                )

                advanced.prop(
                    scene,
                    "bodymove_physics_max_pelvis_step",
                    text="Pelvis Max Step",
                )

                advanced.prop(
                    scene,
                    "bodymove_physics_max_angle_deg",
                    text="Max Rotation",
                )

                advanced.prop(
                    scene,
                    "bodymove_physics_spine_gain",
                    text="Spine",
                )

                advanced.prop(
                    scene,
                    "bodymove_physics_chest_gain",
                    text="Chest",
                )

                advanced.prop(
                    scene,
                    "bodymove_physics_leg_gain",
                    text="Support Leg",
                )

                advanced.prop(
                    scene,
                    "bodymove_physics_arm_gain",
                    text="Arms",
                )

                advanced.prop(
                    scene,
                    "bodymove_physics_head_gain",
                    text="Head",
                )

                advanced.prop(
                    scene,
                    "bodymove_physics_foot_lock",
                    text="Foot Lock",
                )

        else:
            layout.label(text="Run Quick Human Setup to enable motion tools", icon="INFO")


# =========================================================
# REGISTER
# =========================================================

classes = (
    BODYMOVE_OT_AutoMapRig,
    BODYMOVE_OT_QuickHumanSetup,
    BODYMOVE_OT_PreviewAutoPhysics,
    BODYMOVE_OT_ClearAutoPhysicsPreview,
    BODYMOVE_OT_ApplyAutoPhysics,
    BODYMOVE_OT_RestoreOriginalAction,
    BODYMOVE_OT_Initialize,
    BODYMOVE_OT_PlacePoint,
    BODYMOVE_OT_Calculate,
    BODYMOVE_OT_ApplyBalanceAssist,
    BODYMOVE_OT_Detail,
    BODYMOVE_OT_DeletePoint,
    BODYMOVE_OT_ResetAnalysis,
    BODYMOVE_OT_ClearSetup,
    BODYMOVE_OT_GenerateWalk,
    BODYMOVE_OT_RestoreWalkAction,
    BODYMOVE_PT_Main,
    BODYMOVE_PT_Locomotion,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.bodymove_source_object = (
        PointerProperty(
            name="Source Mesh",
            type=bpy.types.Object,
        )
    )

    bpy.types.Scene.bodymove_armature_object = (
        PointerProperty(
            name="Armature",
            type=bpy.types.Object,
        )
    )

    bpy.types.Scene.bodymove_ground_object = (
        PointerProperty(
            name="Ground",
            type=bpy.types.Object,
        )
    )

    bpy.types.Scene.bodymove_initialized = (
        BoolProperty(
            default=False,
        )
    )

    bpy.types.Scene.bodymove_collection_name = (
        StringProperty(
            default="",
        )
    )

    # V0.5 QUICK SETUP
    bpy.types.Scene.bodymove_body_weight = (
        FloatProperty(
            name="Body Weight",
            default=75.0,
            min=20.0,
            soft_max=250.0,
            precision=1,
        )
    )

    bpy.types.Scene.bodymove_walk_speed = FloatProperty(
        name="Speed", default=1.2, min=0.1, max=6.0,
        description="Walking speed in meters per second",
        unit="VELOCITY",
    )
    bpy.types.Scene.bodymove_walk_stride = FloatProperty(
        name="Stride", default=0.65, min=0.1, max=2.0,
        description="Distance covered by one step, in meters",
        unit="LENGTH",
    )
    bpy.types.Scene.bodymove_walk_cycles = bpy.props.IntProperty(
        name="Cycles", default=2, min=1, max=12,
        description="Number of complete left-right walk cycles to generate",
    )
    bpy.types.Scene.bodymove_walk_forward_axis = EnumProperty(
        name="Forward", items=[
            ("X", "+X", "Move along armature local +X"),
            ("-X", "-X", "Move along armature local -X"),
            ("Y", "+Y", "Move along armature local +Y"),
            ("-Y", "-Y", "Move along armature local -Y"),
        ], default="-Y",
    )
    bpy.types.Scene.bodymove_walk_base_action = PointerProperty(
        name="Previous Action", type=bpy.types.Action,
    )
    bpy.types.Scene.bodymove_walk_base_slot = StringProperty(default="")
    bpy.types.Scene.bodymove_walk_action_name = StringProperty(default="")

    # V0.7 SEGMENT DYNAMICS
    bpy.types.Scene.bodymove_segment_dynamics = BoolProperty(name='Segment Reactions', default=True)
    bpy.types.Scene.bodymove_reaction_time = FloatProperty(name='Reaction Time', default=0.12, min=0.01, max=0.5, description='Time scale for counter-motion, in seconds')
    bpy.types.Scene.bodymove_reaction_frequency = FloatProperty(name='Spring Frequency', default=3.0, min=0.5, max=12.0, description='Higher values follow the target faster')
    bpy.types.Scene.bodymove_reaction_damping = FloatProperty(name='Damping', default=1.0, min=0.2, max=2.0, description='1 is critically damped; lower values allow follow-through')
    bpy.types.Scene.bodymove_contact_fallbacks = bpy.props.IntProperty(default=0)
    bpy.types.Scene.bodymove_original_slot = StringProperty(default='')

    # V0.6 AUTOPHYSICS INTERVAL
    bpy.types.Scene.bodymove_autophysics_use_scene_range = (
        BoolProperty(
            name="Use Scene Range",
            default=True,
        )
    )

    bpy.types.Scene.bodymove_autophysics_start = (
        bpy.props.IntProperty(
            name="Start Frame",
            default=1,
        )
    )

    bpy.types.Scene.bodymove_autophysics_end = (
        bpy.props.IntProperty(
            name="End Frame",
            default=60,
        )
    )

    bpy.types.Scene.bodymove_interval_physics_strength = (
        FloatProperty(
            name="Physics Strength",
            default=0.70,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_pose_preservation = (
        FloatProperty(
            name="Pose Preservation",
            default=0.80,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_smooth_trajectory = (
        FloatProperty(
            name="Smooth Trajectory",
            default=0.90,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_smooth_rotation = (
        FloatProperty(
            name="Smooth Rotation",
            default=0.80,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_compensation_motion = (
        FloatProperty(
            name="Compensation Motion",
            default=0.75,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_motion_separation = (
        FloatProperty(
            name="Motion Separation",
            default=0.65,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_secondary_motion = (
        FloatProperty(
            name="Secondary Motion",
            default=0.45,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_autophysics_substeps = (
        bpy.props.IntProperty(
            name="Subframes",
            default=2,
            min=1,
            max=4,
        )
    )

    bpy.types.Scene.bodymove_smoothing_radius = (
        bpy.props.IntProperty(
            name="Smoothing Radius",
            default=4,
            min=1,
            max=20,
        )
    )

    bpy.types.Scene.bodymove_edge_blend_frames = (
        bpy.props.IntProperty(
            name="Edge Blend",
            default=4,
            min=0,
            max=30,
        )
    )

    bpy.types.Scene.bodymove_autophysics_max_frames = (
        bpy.props.IntProperty(
            name="Max Preview Frames",
            default=240,
            min=10,
            max=2000,
        )
    )

    bpy.types.Scene.bodymove_show_autophysics_advanced = (
        BoolProperty(
            name="AutoPhysics Advanced",
            default=False,
        )
    )

    bpy.types.Scene.bodymove_show_live_pose_assist = (
        BoolProperty(
            name="Live Pose Assist",
            default=False,
        )
    )

    bpy.types.Scene.bodymove_preview_ready = (
        BoolProperty(
            default=False,
        )
    )

    bpy.types.Scene.bodymove_preview_status = (
        StringProperty(
            default="No preview",
        )
    )

    bpy.types.Scene.bodymove_ghost_collection_name = (
        StringProperty(
            default="",
        )
    )

    bpy.types.Scene.bodymove_ghost_armature_name = (
        StringProperty(
            default="",
        )
    )

    bpy.types.Scene.bodymove_ghost_mesh_name = (
        StringProperty(
            default="",
        )
    )

    bpy.types.Scene.bodymove_original_action_name = (
        StringProperty(
            default="",
        )
    )

    bpy.types.Scene.bodymove_last_applied_action_name = (
        StringProperty(
            default="",
        )
    )

    bpy.types.Scene.bodymove_virtual_ground_z = (
        FloatProperty(
            name="Virtual Ground Z",
            default=0.0,
            precision=3,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_show_advanced_setup = (
        BoolProperty(
            name="Advanced Setup",
            default=False,
        )
    )

    bpy.types.Scene.bodymove_use_support_stability = (
        BoolProperty(
            name="Fulcrum Stability",
            default=True,
        )
    )

    bpy.types.Scene.bodymove_support_motion_tolerance = (
        FloatProperty(
            name="Support Motion Tolerance",
            default=0.025,
            min=0.001,
            soft_max=0.20,
            precision=3,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_target_margin = (
        FloatProperty(
            name="Support Safety",
            default=0.70,
            min=0.10,
            max=0.95,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_target_com_x = (
        FloatProperty(default=0.0)
    )

    bpy.types.Scene.bodymove_target_com_y = (
        FloatProperty(default=0.0)
    )

    bpy.types.Scene.bodymove_target_com_z = (
        FloatProperty(default=0.0)
    )

    # Extended rig mapping
    bpy.types.Scene.bodymove_map_left_upper_arm = (
        StringProperty(default="")
    )

    bpy.types.Scene.bodymove_map_right_upper_arm = (
        StringProperty(default="")
    )

    bpy.types.Scene.bodymove_map_left_thigh = (
        StringProperty(default="")
    )

    bpy.types.Scene.bodymove_map_right_thigh = (
        StringProperty(default="")
    )

    bpy.types.Scene.bodymove_map_left_shin = (
        StringProperty(default="")
    )

    bpy.types.Scene.bodymove_map_right_shin = (
        StringProperty(default="")
    )

    # Iterative Physics Solver
    bpy.types.Scene.bodymove_physics_enabled = (
        BoolProperty(
            name="Auto Physics",
            default=False,
        )
    )

    bpy.types.Scene.bodymove_show_physics_advanced = (
        BoolProperty(
            name="Physics Advanced",
            default=False,
        )
    )

    bpy.types.Scene.bodymove_physics_strength = (
        FloatProperty(
            name="Solver Strength",
            default=0.35,
            min=0.01,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_physics_iterations = (
        bpy.props.IntProperty(
            name="Iterations",
            default=3,
            min=1,
            max=12,
        )
    )

    bpy.types.Scene.bodymove_physics_deadzone = (
        FloatProperty(
            name="COM Tolerance",
            default=0.012,
            min=0.001,
            soft_max=0.10,
            precision=3,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_physics_pelvis_translation = (
        FloatProperty(
            name="Pelvis Translation",
            default=0.18,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_physics_max_pelvis_step = (
        FloatProperty(
            name="Pelvis Max Step",
            default=0.025,
            min=0.001,
            soft_max=0.20,
            precision=3,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_physics_max_angle_deg = (
        FloatProperty(
            name="Max Rotation",
            default=3.0,
            min=0.1,
            soft_max=15.0,
            precision=1,
        )
    )

    bpy.types.Scene.bodymove_physics_spine_gain = (
        FloatProperty(
            default=2.4,
            min=0.0,
            soft_max=10.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_physics_chest_gain = (
        FloatProperty(
            default=1.8,
            min=0.0,
            soft_max=10.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_physics_head_gain = (
        FloatProperty(
            default=-0.7,
            min=-5.0,
            max=5.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_physics_leg_gain = (
        FloatProperty(
            default=1.1,
            min=0.0,
            soft_max=5.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_physics_arm_gain = (
        FloatProperty(
            default=0.55,
            min=0.0,
            soft_max=5.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_physics_foot_lock = (
        BoolProperty(
            name="Foot Lock",
            default=True,
        )
    )

    bpy.types.Scene.bodymove_physics_status = (
        StringProperty(
            default="OFF",
        )
    )

    bpy.types.Scene.bodymove_physics_error = (
        FloatProperty(
            default=0.0,
        )
    )

    # V0.4 RIG MAPPING
    bpy.types.Scene.bodymove_rig_profile = (
        StringProperty(
            name="Rig Profile",
            default="UNMAPPED",
        )
    )

    bpy.types.Scene.bodymove_map_report = StringProperty(
        name="Mapping Report",
        default="Run Auto Map Rig to inspect this armature",
    )

    bpy.types.Scene.bodymove_map_root = (
        StringProperty(
            name="Root Control",
            default="",
        )
    )

    bpy.types.Scene.bodymove_map_pelvis = (
        StringProperty(
            name="Pelvis Control",
            default="",
        )
    )

    bpy.types.Scene.bodymove_map_spine = (
        StringProperty(
            name="Spine Control",
            default="",
        )
    )

    bpy.types.Scene.bodymove_map_chest = (
        StringProperty(
            name="Chest Control",
            default="",
        )
    )

    bpy.types.Scene.bodymove_map_head = (
        StringProperty(
            name="Head Control",
            default="",
        )
    )

    bpy.types.Scene.bodymove_map_left_foot = (
        StringProperty(
            name="Left Foot Control",
            default="",
        )
    )

    bpy.types.Scene.bodymove_map_right_foot = (
        StringProperty(
            name="Right Foot Control",
            default="",
        )
    )

    # V0.4 AUTO INFLUENCE SOLVER
    bpy.types.Scene.bodymove_auto_influence = (
        BoolProperty(
            name="Auto Influence",
            default=False,
        )
    )

    bpy.types.Scene.bodymove_solver_mode = (
        EnumProperty(
            name="Solver Mode",
            items=(
                (
                    "UPPER_BODY",
                    "Upper Body",
                    "Safer: spine, chest and head react",
                ),
                (
                    "FULL_BODY",
                    "Full Body",
                    "Pelvis plus upper body react",
                ),
                (
                    "FULL_BODY_LOCK",
                    "Full Body + Foot Lock",
                    "Experimental: full body reaction with active foot counter-lock",
                ),
            ),
            default="FULL_BODY",
        )
    )

    bpy.types.Scene.bodymove_solver_strength = (
        FloatProperty(
            name="Reaction Strength",
            default=0.18,
            min=0.0,
            max=1.0,
            precision=3,
        )
    )

    bpy.types.Scene.bodymove_solver_deadzone = (
        FloatProperty(
            name="COM Dead Zone",
            default=0.015,
            min=0.001,
            soft_max=0.20,
            precision=3,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_solver_max_step = (
        FloatProperty(
            name="Max Solver Step",
            default=0.025,
            min=0.001,
            soft_max=0.20,
            precision=3,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_solver_only_unbalanced = (
        BoolProperty(
            name="Only if Unbalanced",
            default=False,
        )
    )

    bpy.types.Scene.bodymove_solver_foot_lock = (
        BoolProperty(
            name="Foot Lock",
            default=True,
        )
    )

    bpy.types.Scene.bodymove_solver_pelvis_weight = (
        FloatProperty(
            name="Pelvis Response",
            default=0.55,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_solver_spine_weight = (
        FloatProperty(
            name="Spine Response",
            default=0.25,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_solver_chest_weight = (
        FloatProperty(
            name="Chest Response",
            default=0.15,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_solver_head_weight = (
        FloatProperty(
            name="Head Response",
            default=0.05,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_solver_last_step = (
        FloatProperty(
            default=0.0,
        )
    )

    bpy.types.Scene.bodymove_solver_last_error = (
        FloatProperty(
            default=0.0,
        )
    )

    bpy.types.Scene.bodymove_solver_stagnation = (
        bpy.props.IntProperty(
            default=0,
        )
    )

    bpy.types.Scene.bodymove_solver_stagnation_limit = (
        bpy.props.IntProperty(
            name="Stagnation Limit",
            default=12,
            min=3,
            max=100,
        )
    )

    bpy.types.Scene.bodymove_solver_status = (
        StringProperty(
            default="OFF",
        )
    )

    bpy.types.Scene.bodymove_live_analysis = (
        BoolProperty(
            name="Live Analysis",
            default=True,
        )
    )

    bpy.types.Scene.bodymove_contact_tolerance = (
        FloatProperty(
            name="Contact Tolerance",
            default=0.035,
            min=0.001,
            soft_max=0.25,
            precision=3,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_surface_offset = (
        FloatProperty(
            default=0.01,
            min=0.0,
            soft_max=0.1,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_mass_category = (
        EnumProperty(
            items=MASS_CATEGORIES,
            default="HEAD",
        )
    )

    bpy.types.Scene.bodymove_custom_mass_name = (
        StringProperty(
            default="Custom Point",
        )
    )

    bpy.types.Scene.bodymove_mass_value = (
        FloatProperty(
            default=5.0,
            min=0.01,
            soft_max=200.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_mass_bone_name = (
        StringProperty(
            name="Mass Bone",
            default="",
        )
    )

    bpy.types.Scene.bodymove_mass_point_size = (
        FloatProperty(
            default=0.06,
            min=0.005,
            soft_max=0.5,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_support_role = (
        EnumProperty(
            items=SUPPORT_ROLES,
            default="LEFT_FOOT",
        )
    )

    bpy.types.Scene.bodymove_custom_support_name = (
        StringProperty(
            default="Custom Support",
        )
    )

    bpy.types.Scene.bodymove_support_bone_name = (
        StringProperty(
            name="Support Bone",
            default="",
        )
    )

    bpy.types.Scene.bodymove_support_radius = (
        FloatProperty(
            default=0.12,
            min=0.01,
            soft_max=1.0,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_com_size = (
        FloatProperty(
            default=0.14,
            min=0.01,
            soft_max=0.5,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_projection_size = (
        FloatProperty(
            default=0.11,
            min=0.01,
            soft_max=0.5,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_assist_primary_bone = (
        StringProperty(
            name="Primary Control Bone",
            default="",
        )
    )

    bpy.types.Scene.bodymove_assist_secondary_bone = (
        StringProperty(
            name="Secondary Control Bone",
            default="",
        )
    )

    bpy.types.Scene.bodymove_assist_primary_weight = (
        FloatProperty(
            name="Primary Influence",
            default=0.70,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_assist_secondary_weight = (
        FloatProperty(
            name="Secondary Influence",
            default=0.30,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_assist_strength = (
        FloatProperty(
            name="Assist Strength",
            default=0.75,
            min=0.0,
            max=1.0,
            precision=2,
        )
    )

    bpy.types.Scene.bodymove_assist_max_correction = (
        FloatProperty(
            name="Max Correction",
            default=0.20,
            min=0.001,
            soft_max=1.0,
            precision=3,
            unit="LENGTH",
        )
    )

    bpy.types.Scene.bodymove_assist_only_unbalanced = (
        BoolProperty(
            name="Only if Unbalanced",
            default=True,
        )
    )

    bpy.types.Scene.bodymove_last_assist_x = (
        FloatProperty(default=0.0)
    )

    bpy.types.Scene.bodymove_last_assist_y = (
        FloatProperty(default=0.0)
    )

    bpy.types.Scene.bodymove_last_assist_distance = (
        FloatProperty(default=0.0)
    )

    bpy.types.Scene.bodymove_detail_name = (
        StringProperty(
            default="",
        )
    )

    bpy.types.Scene.bodymove_analysis_valid = (
        BoolProperty(
            default=False,
        )
    )

    bpy.types.Scene.bodymove_total_mass = (
        FloatProperty(
            default=0.0,
        )
    )

    bpy.types.Scene.bodymove_com_x = (
        FloatProperty(default=0.0)
    )
    bpy.types.Scene.bodymove_com_y = (
        FloatProperty(default=0.0)
    )
    bpy.types.Scene.bodymove_com_z = (
        FloatProperty(default=0.0)
    )

    bpy.types.Scene.bodymove_base_com_x = (
        FloatProperty(default=0.0)
    )
    bpy.types.Scene.bodymove_base_com_y = (
        FloatProperty(default=0.0)
    )
    bpy.types.Scene.bodymove_base_com_z = (
        FloatProperty(default=0.0)
    )

    bpy.types.Scene.bodymove_com_shift_live = (
        FloatProperty(
            default=0.0,
        )
    )

    bpy.types.Scene.bodymove_mass_point_count = (
        bpy.props.IntProperty(
            default=0,
        )
    )

    bpy.types.Scene.bodymove_support_count = (
        bpy.props.IntProperty(
            default=0,
        )
    )

    bpy.types.Scene.bodymove_active_support_count = (
        bpy.props.IntProperty(
            default=0,
        )
    )

    bpy.types.Scene.bodymove_balance_status = (
        StringProperty(
            default="-",
        )
    )

    bpy.types.Scene.bodymove_support_mode = (
        StringProperty(
            default="-",
        )
    )

    bpy.types.Scene.bodymove_distance_outside = (
        FloatProperty(
            default=0.0,
        )
    )

    bpy.types.Scene.bodymove_heaviest_name = (
        StringProperty(
            default="",
        )
    )

    bpy.types.Scene.bodymove_heaviest_mass = (
        FloatProperty(
            default=0.0,
        )
    )

    if bodymove_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(
            bodymove_depsgraph_update
        )

    if bodymove_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(
            bodymove_frame_change
        )

    reactive.register()
    balance_preview.register()


def unregister():
    balance_preview.unregister()
    reactive.unregister()
    if bodymove_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(
            bodymove_depsgraph_update
        )

    if bodymove_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(
            bodymove_frame_change
        )

    props = (
        "bodymove_walk_speed",
        "bodymove_walk_stride",
        "bodymove_walk_cycles",
        "bodymove_walk_forward_axis",
        "bodymove_walk_base_action",
        "bodymove_walk_base_slot",
        "bodymove_walk_action_name",
        "bodymove_segment_dynamics",
        "bodymove_reaction_time",
        "bodymove_reaction_frequency",
        "bodymove_reaction_damping",
        "bodymove_contact_fallbacks",
        "bodymove_original_slot",
        "bodymove_source_object",
        "bodymove_armature_object",
        "bodymove_ground_object",
        "bodymove_initialized",
        "bodymove_collection_name",
        "bodymove_body_weight",
        "bodymove_autophysics_use_scene_range",
        "bodymove_autophysics_start",
        "bodymove_autophysics_end",
        "bodymove_interval_physics_strength",
        "bodymove_pose_preservation",
        "bodymove_smooth_trajectory",
        "bodymove_smooth_rotation",
        "bodymove_compensation_motion",
        "bodymove_motion_separation",
        "bodymove_secondary_motion",
        "bodymove_autophysics_substeps",
        "bodymove_smoothing_radius",
        "bodymove_edge_blend_frames",
        "bodymove_autophysics_max_frames",
        "bodymove_show_autophysics_advanced",
        "bodymove_show_live_pose_assist",
        "bodymove_preview_ready",
        "bodymove_preview_status",
        "bodymove_ghost_collection_name",
        "bodymove_ghost_armature_name",
        "bodymove_ghost_mesh_name",
        "bodymove_original_action_name",
        "bodymove_last_applied_action_name",
        "bodymove_virtual_ground_z",
        "bodymove_show_advanced_setup",
        "bodymove_use_support_stability",
        "bodymove_support_motion_tolerance",
        "bodymove_target_margin",
        "bodymove_target_com_x",
        "bodymove_target_com_y",
        "bodymove_target_com_z",
        "bodymove_map_left_upper_arm",
        "bodymove_map_right_upper_arm",
        "bodymove_map_left_thigh",
        "bodymove_map_right_thigh",
        "bodymove_map_left_shin",
        "bodymove_map_right_shin",
        "bodymove_physics_enabled",
        "bodymove_show_physics_advanced",
        "bodymove_physics_strength",
        "bodymove_physics_iterations",
        "bodymove_physics_deadzone",
        "bodymove_physics_pelvis_translation",
        "bodymove_physics_max_pelvis_step",
        "bodymove_physics_max_angle_deg",
        "bodymove_physics_spine_gain",
        "bodymove_physics_chest_gain",
        "bodymove_physics_head_gain",
        "bodymove_physics_leg_gain",
        "bodymove_physics_arm_gain",
        "bodymove_physics_foot_lock",
        "bodymove_physics_status",
        "bodymove_physics_error",
        "bodymove_rig_profile",
        "bodymove_map_root",
        "bodymove_map_pelvis",
        "bodymove_map_spine",
        "bodymove_map_chest",
        "bodymove_map_head",
        "bodymove_map_left_foot",
        "bodymove_map_right_foot",
        "bodymove_auto_influence",
        "bodymove_solver_mode",
        "bodymove_solver_strength",
        "bodymove_solver_deadzone",
        "bodymove_solver_max_step",
        "bodymove_solver_only_unbalanced",
        "bodymove_solver_foot_lock",
        "bodymove_solver_pelvis_weight",
        "bodymove_solver_spine_weight",
        "bodymove_solver_chest_weight",
        "bodymove_solver_head_weight",
        "bodymove_solver_last_step",
        "bodymove_solver_last_error",
        "bodymove_solver_stagnation",
        "bodymove_solver_stagnation_limit",
        "bodymove_solver_status",
        "bodymove_live_analysis",
        "bodymove_contact_tolerance",
        "bodymove_surface_offset",
        "bodymove_mass_category",
        "bodymove_custom_mass_name",
        "bodymove_mass_value",
        "bodymove_mass_bone_name",
        "bodymove_mass_point_size",
        "bodymove_support_role",
        "bodymove_custom_support_name",
        "bodymove_support_bone_name",
        "bodymove_support_radius",
        "bodymove_com_size",
        "bodymove_projection_size",
        "bodymove_assist_primary_bone",
        "bodymove_assist_secondary_bone",
        "bodymove_assist_primary_weight",
        "bodymove_assist_secondary_weight",
        "bodymove_assist_strength",
        "bodymove_assist_max_correction",
        "bodymove_assist_only_unbalanced",
        "bodymove_last_assist_x",
        "bodymove_last_assist_y",
        "bodymove_last_assist_distance",
        "bodymove_detail_name",
        "bodymove_analysis_valid",
        "bodymove_total_mass",
        "bodymove_com_x",
        "bodymove_com_y",
        "bodymove_com_z",
        "bodymove_base_com_x",
        "bodymove_base_com_y",
        "bodymove_base_com_z",
        "bodymove_com_shift_live",
        "bodymove_mass_point_count",
        "bodymove_support_count",
        "bodymove_active_support_count",
        "bodymove_balance_status",
        "bodymove_support_mode",
        "bodymove_distance_outside",
        "bodymove_heaviest_name",
        "bodymove_heaviest_mass",
    )

    for prop in props:
        if hasattr(
            bpy.types.Scene,
            prop,
        ):
            delattr(
                bpy.types.Scene,
                prop,
            )

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
