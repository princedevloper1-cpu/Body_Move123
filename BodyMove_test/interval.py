"""Blender integration for segment dynamics and interval Action transfer."""

import math
import bpy
from mathutils import Matrix, Vector
from bpy_extras.anim_utils import action_ensure_channelbag_for_slot


def capture_segments(scene, api, depsgraph):
    rig = scene.bodymove_armature_object.evaluated_get(depsgraph)
    pelvis = rig.pose.bones.get(scene.bodymove_map_pelvis)
    reference = rig.matrix_world @ pelvis.matrix if pelvis else rig.matrix_world
    origin = reference.translation
    rotation = reference.to_quaternion().normalized()
    inverse_rotation = rotation.conjugated()
    segments = {}
    for point in api.helpers(scene, api.MASS_POINT):
        bone = rig.pose.bones.get(str(point.get(api.BONE, '')))
        mass = float(point.get(api.MASS, 0))
        if bone is None or mass <= 0:
            continue
        world = rig.matrix_world @ bone.matrix
        position = point.evaluated_get(depsgraph).matrix_world.translation
        length = (rig.matrix_world.to_3x3() @ (bone.tail - bone.head)).length
        segments[point.name] = {
            'bone': bone.name, 'mass': mass, 'length': max(length, 0.001),
            'position': inverse_rotation @ (position - origin),
            'rotation': (inverse_rotation @ world.to_quaternion()).normalized(),
        }
    return segments, rotation


def reaction_receivers(scene, armature):
    fields = (('spine', 0.35), ('chest', 0.3), ('head', 0.15),
              ('left_upper_arm', 0.3), ('right_upper_arm', 0.3))
    receivers = {}
    for field, weight in fields:
        name = getattr(scene, 'bodymove_map_' + field)
        bone = armature.pose.bones.get(name)
        if bone is None or name in receivers:
            continue
        receivers[name] = {'weight': weight,
                           'exclude': {name}}
    return receivers


def apply_rotations(armature, rotations):
    # Parent-first evaluation lets a child's response build on its moved parent.
    ordered = sorted((armature.pose.bones[n] for n in rotations),
                     key=lambda bone: len(bone.parent_recursive))
    for bone in ordered:
        vector = rotations[bone.name]
        if vector.length < 1e-9:
            continue
        axis = armature.matrix_world.inverted().to_3x3() @ vector.normalized()
        pivot = bone.head.copy()
        transform = (Matrix.Translation(pivot)
                     @ Matrix.Rotation(vector.length, 4, axis.normalized())
                     @ Matrix.Translation(-pivot))
        bone.matrix = transform @ bone.matrix
        bpy.context.view_layer.update()


def capture_feet(scene, armature, roles, api):
    return {role: (armature.matrix_world @ armature.pose.bones[name].matrix).copy()
            for role in roles
            if (name := api.role_to_foot_bone(scene, role)) in armature.pose.bones}


def restore_feet(scene, armature, wanted, api, tolerance):
    """Rotate an FK leg chain without translating/stretching the foot bone.

    The lock preserves the authored foot matrix at this frame, not a guessed
    stationary ground anchor. Unreachable targets are reported to the caller.
    """
    inverse = armature.matrix_world.inverted()
    for role, matrix in wanted.items():
        foot = armature.pose.bones.get(api.role_to_foot_bone(scene, role))
        if foot is None:
            continue
        target = (inverse @ matrix).translation
        # Never rotate a torso/root as if it were a leg joint.
        side = 'left' if role == 'LEFT_FOOT' else 'right'
        names = [getattr(scene, 'bodymove_map_' + side + '_shin'),
                 getattr(scene, 'bodymove_map_' + side + '_thigh')]
        ancestors = {bone.name for bone in foot.parent_recursive}
        chain = [armature.pose.bones[n] for n in names if n in ancestors]
        for _ in range(24):
            if (armature.matrix_world @ foot.head - matrix.translation).length <= tolerance:
                break
            for joint in chain:
                pivot = joint.head.copy()
                current = foot.head - pivot
                desired = target - pivot
                if min(current.length, desired.length) < 1e-9:
                    continue
                rotation = current.rotation_difference(desired)
                joint.matrix = (Matrix.Translation(pivot) @ rotation.to_matrix().to_4x4()
                                @ Matrix.Translation(-pivot) @ joint.matrix)
                bpy.context.view_layer.update()
        # Preserve foot orientation without changing its solved translation.
        location = foot.matrix.translation.copy()
        restored = inverse @ matrix
        restored.translation = location
        foot.matrix = restored
        bpy.context.view_layer.update()
    return max(((armature.matrix_world @ armature.pose.bones[api.role_to_foot_bone(scene, role)].matrix).translation
                - matrix.translation).length for role, matrix in wanted.items()) if wanted else 0.0


def channelbag(animation_data):
    action = animation_data.action if animation_data else None
    slot = animation_data.action_slot if animation_data else None
    if action is None or slot is None:
        return None
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(slot)
            if bag is not None:
                return bag
    return None


def apply_block_reason(armature):
    if any(not c.mute for c in armature.constraints):
        return 'Object constraints: use a baked FK rig for Apply.'
    if any(not c.mute for bone in armature.pose.bones for c in bone.constraints):
        return 'Bone constraints/IK: use a baked FK rig for Apply; preview remains available.'
    animation = armature.animation_data
    if animation:
        if animation.drivers or any(not track.mute for track in animation.nla_tracks):
            return 'Drivers or active NLA tracks: bake to a single FK Action before Apply.'
        if animation.action and len(animation.action.layers) > 1:
            return 'Multi-layer Actions require baking before Apply.'
        bag = channelbag(animation)
        if bag and any(curve.modifiers or curve.sampled_points for curve in bag.fcurves):
            return 'F-Curve modifiers/samples require baking before Apply.'
    return ''


def split_curve(curve, frame):
    """Split a Bezier segment at frame without changing its outside shape."""
    keys = list(curve.keyframe_points)
    for key in keys:
        if abs(key.co.x - frame) < 1e-5:
            return
    left = next((key for key in reversed(keys) if key.co.x < frame), None)
    right = next((key for key in keys if key.co.x > frame), None)
    value = curve.evaluate(frame)
    interpolation = left.interpolation if left else 'CONSTANT'
    handle_left = handle_right = Vector((frame, value))
    if left is not None and right is not None and interpolation == 'BEZIER':
        p0, p1, p2, p3 = left.co.copy(), left.handle_right.copy(), right.handle_left.copy(), right.co.copy()
        # Blender shortens overlapping handles before evaluating a segment.
        span = p3.x - p0.x
        handle_span = abs(p1.x-p0.x) + abs(p3.x-p2.x)
        if handle_span > span:
            factor = span / handle_span
            p1 = p0 + (p1-p0)*factor
            p2 = p3 + (p2-p3)*factor
        low, high = 0.0, 1.0
        for _ in range(48):
            t = (low+high)*0.5
            point = p0*(1-t)**3 + p1*3*(1-t)**2*t + p2*3*(1-t)*t*t + p3*t**3
            if point.x < frame:
                low = t
            else:
                high = t
        a, b, c = p0.lerp(p1, t), p1.lerp(p2, t), p2.lerp(p3, t)
        handle_left, handle_right = a.lerp(b, t), b.lerp(c, t)
        left.handle_right_type = 'FREE'
        right.handle_left_type = 'FREE'
        left.handle_right, right.handle_left = a, c
    # Freeze outside handles: insertion must not recalculate AUTO handles.
    saved = [(key.co.x, key.handle_left.copy(), key.handle_right.copy()) for key in keys]
    for key in keys:
        key.handle_left_type = key.handle_right_type = 'FREE'
    key = curve.keyframe_points.insert(frame, value, options={'FAST'})
    key.interpolation = interpolation
    key.handle_left_type = key.handle_right_type = 'FREE'
    key.handle_left, key.handle_right = handle_left, handle_right
    curve.update()
    for time, hl, hr in saved:
        for key in curve.keyframe_points:
            if abs(key.co.x-time) < 1e-5:
                key.handle_left, key.handle_right = hl, hr
                break


def merge_action_interval(armature, ghost, start, end):
    """Copy original Action; transfer only ghost channels inside the interval.

    Guard frames and Bezier splitting retain outside animation, custom channels
    and other slots. No live source mutation is needed to prepare the Action.
    """
    source_data = armature.animation_data
    original = source_data.action if source_data else None
    ghost_bag = channelbag(ghost.animation_data)
    if ghost_bag is None:
        raise RuntimeError('Ghost Action has no animation channel slot')
    new_action = original.copy() if original else bpy.data.actions.new(armature.name + '_BodyMovePhysics')
    try:
        if original and source_data.action_slot:
            slot = next(s for s in new_action.slots if s.identifier == source_data.action_slot.identifier)
        else:
            slot = new_action.slots.new(id_type='OBJECT', name=armature.name)
        bag = action_ensure_channelbag_for_slot(new_action, slot)
        for source_curve in ghost_bag.fcurves:
            curve = bag.fcurves.find(source_curve.data_path, index=source_curve.array_index)
            if curve is None:
                curve = bag.fcurves.new(source_curve.data_path, index=source_curve.array_index)
                # Previously unanimated channels must return to their original value.
                value = armature.path_resolve(source_curve.data_path)[source_curve.array_index]
                curve.keyframe_points.insert(start-1, value)
                curve.keyframe_points.insert(end+1, value)
            elif any(key.interpolation not in {'BEZIER', 'LINEAR', 'CONSTANT'} for key in curve.keyframe_points):
                raise RuntimeError('Bake easing F-Curves before applying interval physics')
            split_curve(curve, start-1)
            split_curve(curve, end+1)
            outside = [(key.co.x, key.handle_left.copy(), key.handle_right.copy(), key.interpolation)
                       for key in curve.keyframe_points if key.co.x < start or key.co.x > end]
            for key in list(curve.keyframe_points):
                if start <= key.co.x <= end:
                    curve.keyframe_points.remove(key, fast=True)
            for source_key in source_curve.keyframe_points:
                if start <= source_key.co.x <= end:
                    key = curve.keyframe_points.insert(source_key.co.x, source_key.co.y, options={'FAST'})
                    key.interpolation = 'LINEAR'
            curve.update()
            for time, hl, hr, interpolation in outside:
                key = next(k for k in curve.keyframe_points if abs(k.co.x-time) < 1e-5)
                key.handle_left_type = key.handle_right_type = 'FREE'
                key.handle_left, key.handle_right = hl, hr
                key.interpolation = interpolation
            # The guard immediately preceding the replacement connects linearly.
            guard = next(k for k in curve.keyframe_points if abs(k.co.x-(start-1)) < 1e-5)
            guard.interpolation = 'LINEAR'
        return new_action, slot
    except Exception:
        bpy.data.actions.remove(new_action)
        raise
