"""Experimental live targets for Auto-Rig Pro control bones.

This first ARP backend drives the rig's own IK controls, samples COM from
BodyMove mass helpers, and makes a bounded root translation to improve
quasi-static balance. It is a posing aid, not a rigid-body simulator.
"""
import bpy
import importlib
import math
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from . import body_controls

TAG = "_bodymove_live_control"


def api():
    return importlib.import_module(__package__)


class LivePose:
    backend = "ARP"

    def __init__(self, scene):
        self.scene = scene
        self.rig = scene.bodymove_armature_object
        if self.rig is None or self.rig.type != "ARMATURE":
            raise ValueError("Choose the Auto-Rig Pro armature first")
        self.height = max(api().source_world_height(scene.bodymove_source_object), 0.1)
        self.frame = (scene.frame_current, scene.frame_subframe)
        self.rig_world = self.rig.matrix_world.copy()
        self.original = {b.name: b.matrix_basis.copy() for b in self.rig.pose.bones}
        self.controls = {}
        self.body_controls = {}
        self.body_initial = {}
        self.controller_bones = {}
        self.translation_jacobians = {}
        self.display = {}
        self.collection = None
        self.saved_physics = scene.bodymove_physics_enabled
        self._authored_pelvis = False
        self._authored_torso = False
        self.authored_shoulders = set()
        self.user_has_edited_target = False
        self.applied_target_follow = Vector()

        names = {
            "left_foot": scene.bodymove_map_left_foot or "c_foot_ik.l",
            "right_foot": scene.bodymove_map_right_foot or "c_foot_ik.r",
            "left_hand": "c_hand_ik.l",
            "right_hand": "c_hand_ik.r",
            "pelvis": scene.bodymove_map_pelvis or "c_root.x",
            "torso": scene.bodymove_map_chest or "c_spine_02.x",
        }
        missing = [name for name in names.values() if self.rig.pose.bones.get(name) is None]
        if missing:
            raise ValueError("Auto-Rig Pro controls not found: " + ", ".join(missing))
        self.controller_bones = names
        self.balance_bones = {
            key: name for key, name in (("left_shoulder", "c_shoulder.l"), ("right_shoulder", "c_shoulder.r"))
            if self.rig.pose.bones.get(name) is not None
        }
        self.balance_root_name = "c_root_master.x" if self.rig.pose.bones.get("c_root_master.x") else names["pelvis"]
        if self.rig.pose.bones.get(self.balance_root_name) is None:
            raise ValueError("Auto-Rig Pro root control not found")
        self.translation_jacobians["balance_root"] = self._measure_translation_jacobian(
            self.rig.pose.bones[self.balance_root_name]
        )
        self.balance_root_initial_world = self._world_control_position(
            self.rig.pose.bones[self.balance_root_name]
        )
        self.arms = {side: {} for side in ("left", "right")}

        for key, bone_name in names.items():
            bone = self.rig.pose.bones[bone_name]
            if key.endswith("foot") or key.endswith("hand"):
                # A controller with no constraints cannot drive the skinned rig.
                if not bone.constraints:
                    raise ValueError("Expected an Auto-Rig Pro IK control: " + bone_name)
            self.translation_jacobians[key] = self._measure_translation_jacobian(bone)
        self.last_basis = {
            key: self.rig.pose.bones[name].matrix_basis.copy()
            for key, name in {**self.controller_bones, **self.balance_bones}.items()
        }
        self.authored_targets = {}
        self.step_source_side = None
        self.generated_step_key = None
        self.generated_step_base = None
        self.generated_step_target = None
        self.initial_target_matrices = {}
        self.initial_world_controls = {
            key: self._world_control_position(self.rig.pose.bones[name])
            for key, name in self.controller_bones.items()
        }
        self.initial_world_shoulders = {
            key: (self.rig.matrix_world @ self.rig.pose.bones[name].matrix).copy()
            for key, name in self.balance_bones.items()
        }

    def _update(self):
        bpy.context.view_layer.update()

    def _world_control_position(self, bone):
        return (self.rig.matrix_world @ bone.matrix).translation.copy()

    def _measure_translation_jacobian(self, bone):
        original = bone.matrix_basis.copy()
        base = self._world_control_position(bone)
        epsilon = max(self.height * 0.001, 1e-4)
        jac = np.zeros((3, 3), dtype=float)
        try:
            for axis in range(3):
                trial = original.copy()
                trial.translation[axis] += epsilon
                bone.matrix_basis = trial
                self._update()
                jac[:, axis] = (np.asarray(self._world_control_position(bone)) - np.asarray(base)) / epsilon
        finally:
            bone.matrix_basis = original
            self._update()
        if not np.isfinite(jac).all() or np.linalg.matrix_rank(jac, tol=1e-5) < 3:
            raise ValueError("Could not calibrate ARP control motion: " + bone.name)
        return jac

    def _place_controller(self, key, world_position):
        bone = self.rig.pose.bones[self.controller_bones[key]]
        error = np.asarray(world_position, dtype=float) - np.asarray(self._world_control_position(bone))
        delta = np.linalg.lstsq(self.translation_jacobians[key], error, rcond=1e-6)[0]
        basis = bone.matrix_basis.copy()
        basis.translation += Vector(delta.tolist())
        bone.matrix_basis = basis
        self._update()
        residual = np.asarray(world_position, dtype=float) - np.asarray(self._world_control_position(bone))
        return float(np.linalg.norm(residual))

    def _add_target(self, key, matrix, shape):
        obj = bpy.data.objects.new("BM ARP " + key.replace("_", " ").title(), None)
        self.collection.objects.link(obj)
        obj[TAG] = True
        obj.matrix_world = matrix.copy()
        self.initial_target_matrices[key] = matrix.copy()
        obj.empty_display_type = shape
        obj.empty_display_size = self.height * 0.045
        obj.show_in_front = True
        obj.show_name = True
        obj.hide_render = True
        obj.lock_scale = (True, True, True)
        obj[body_controls.INFLUENCE] = 1.0
        if key in {"left_foot", "right_foot"}:
            self.controls[key.split("_")[0]] = obj
        else:
            body_key = key
            self.body_controls[body_key] = obj
            self.body_initial[body_key] = matrix.copy()
        self.display[key] = matrix.copy()

    def create_controls(self):
        self.collection = bpy.data.collections.new("BodyMove Auto-Rig Pro Targets")
        self.collection[TAG] = True
        self.scene.collection.children.link(self.collection)
        for key, bone_name in self.controller_bones.items():
            matrix = self.rig.matrix_world @ self.rig.pose.bones[bone_name].matrix
            self._add_target(key, matrix, "CUBE" if key.endswith("foot") else "SPHERE")
        self.scene.bodymove_physics_enabled = False
        self.scene.bodymove_reactive_enabled = True
        self.scene.bodymove_reactive_status = "Auto-Rig Pro targets ready: move a target with G"
        self.solve(force=True)

    def _edited_targets(self):
        result = {}
        for key, bone_name in self.controller_bones.items():
            obj = self.controls.get(key.split("_")[0]) if key.endswith("foot") else self.body_controls.get(key)
            if obj is None:
                continue
            before = self.display[key]
            if max(abs(v) for row in obj.matrix_world - before for v in row) > 1e-5:
                result[key] = obj
        return result

    def _edited_pose_controls(self):
        edited = {}
        for key, bone_name in {**self.controller_bones, **self.balance_bones}.items():
            bone = self.rig.pose.bones[bone_name]
            if max(abs(v) for row in bone.matrix_basis-self.last_basis[key] for v in row) > 2e-5:
                target = self._world_control_position(bone)
                edited[key] = target
                if key in self.controller_bones:
                    self.authored_targets[key] = target.copy()
                    obj = self.controls.get(key.split("_")[0]) if key.endswith("foot") else self.body_controls.get(key)
                    if obj:
                        matrix = obj.matrix_world.copy(); matrix.translation = target
                        obj.matrix_world = matrix
                        self.display[key] = matrix.copy()
                else:
                    self.authored_shoulders.add(key.split("_")[0])
        return edited

    def _com(self):
        graph = bpy.context.evaluated_depsgraph_get()
        points = list(api().helpers(self.scene, api().MASS_POINT))
        total = sum(max(0.0, float(obj.get(api().MASS, 0.0))) for obj in points)
        if total <= 0:
            raise ValueError("Create BodyMove mass points before enabling Auto-Rig Pro targets")
        value = Vector()
        for obj in points:
            mass = max(0.0, float(obj.get(api().MASS, 0.0)))
            value += obj.evaluated_get(graph).matrix_world.translation * mass
        return value / total

    def _active_supports(self):
        graph = bpy.context.evaluated_depsgraph_get()
        active, points = [], {}
        for side in ("left", "right"):
            support = api().support_by_role(self.scene, side.upper() + "_FOOT")
            if support is None:
                continue
            distance = api().support_ground_distance(self.scene, support, graph) if self.scene.bodymove_ground_object else api().virtual_ground_distance(self.scene, support)
            if distance is not None and distance <= max(self.scene.bodymove_contact_tolerance, 1e-4):
                active.append(side)
                points[side] = support.matrix_world.translation.copy()
        return active, points

    def _restore_generated_step(self):
        key = self.generated_step_key
        if not key:
            return
        side = key.split("_")[0]
        matrix = (self.generated_step_base.copy()
                  if self.generated_step_base is not None
                  else self.initial_target_matrices.get(key))
        if matrix is not None:
            matrix = matrix.copy()
            obj = self.controls.get(side)
            if obj:
                obj.matrix_world = matrix
                self.display[key] = matrix.copy()
            self._place_controller(key, matrix.translation)
        self.generated_step_key = None
        self.generated_step_base = None
        self.generated_step_target = None

    def _apply_reciprocal_step(self, edited, active):
        """Make the opposite ARP foot swing when a grounded foot is dragged."""
        edited_feet = [key for key in edited if key in {"left_foot", "right_foot"}]
        if not self.scene.bodymove_reactive_step_assist:
            if self.generated_step_key in edited_feet:
                self.generated_step_key = None
                self.generated_step_base = None
                self.generated_step_target = None
            else:
                self._restore_generated_step()
            self.step_source_side = None
            return

        if len(edited_feet) > 1:
            if self.generated_step_key in edited_feet:
                self.generated_step_key = None
                self.generated_step_base = None
                self.generated_step_target = None
            else:
                self._restore_generated_step()
            self.step_source_side = None
            return
        if edited_feet:
            key = edited_feet[0]
            side = key.split("_")[0]
            if self.step_source_side != side:
                # The user has taken over the foot that was previously moving
                # automatically. Keep that new input and switch the swing side.
                if self.generated_step_key == key:
                    self.generated_step_key = None
                    self.generated_step_base = None
                    self.generated_step_target = None
                else:
                    self._restore_generated_step()
                if self.step_source_side:
                    self.authored_targets.pop(self.step_source_side + "_foot", None)
                self.step_source_side = side

        side = self.step_source_side
        if side is None:
            return
        source_key = side + "_foot"
        source_target = self.authored_targets.get(source_key)
        if source_target is None:
            return
        source_initial = self.initial_world_controls[source_key]
        delta = Vector(source_target) - source_initial
        delta.z = 0.0
        distance = delta.length
        opposite_side = "right" if side == "left" else "left"
        opposite_key = opposite_side + "_foot"

        # Only create a counter-step while the user-led foot is still a ground
        # contact. If it is lifted, leave the opposite foot planted as support.
        if side not in active or distance < self.height * 0.025:
            self._restore_generated_step()
            return

        if self.generated_step_key != opposite_key:
            self._restore_generated_step()
            self.generated_step_key = opposite_key
            self.generated_step_base = self.controls[opposite_side].matrix_world.copy()

        max_travel = self.height * 0.22
        progress = min(1.0, max(0.0, (distance - self.height * 0.025) /
                                max(max_travel - self.height * 0.025, 1e-6)))
        travel = max_travel * progress
        lift = self.height * 0.075 * math.sin(math.pi * progress)
        target = self.generated_step_base.copy()
        target.translation += -delta.normalized() * travel
        target.translation.z += lift
        self._place_controller(opposite_key, target.translation)
        obj = self.controls[opposite_side]
        obj.matrix_world = target
        self.display[opposite_key] = target.copy()
        self.generated_step_target = target.translation.copy()

    def _apply_target_follow(self):
        """Move the shared body root in response to persistent user targets."""
        if not self.authored_targets or self.scene.bodymove_reactive_strength <= 0:
            return
        weights = {
            "left_foot": 0.18, "right_foot": 0.18,
            "left_hand": 0.30, "right_hand": 0.30,
            "pelvis": 0.50, "torso": 0.30,
        }
        desired = Vector()
        for key, position in self.authored_targets.items():
            initial = self.initial_world_controls.get(key)
            if initial is None:
                continue
            delta = Vector(position) - initial
            if key.endswith("_hand"):
                delta.z *= 0.35
            desired += delta * weights.get(key, 0.0)
        desired *= self.scene.bodymove_reactive_strength
        maximum = self.height * 0.15
        if desired.length > maximum:
            desired *= maximum / desired.length
        remaining = desired - self.applied_target_follow
        max_step = self.height * 0.02
        if remaining.length > max_step:
            remaining *= max_step / remaining.length
        if remaining.length < self.height * 1e-6:
            return
        root = self.rig.pose.bones[self.balance_root_name]
        local_delta = np.linalg.lstsq(
            self.translation_jacobians["balance_root"], np.asarray(remaining, dtype=float), rcond=1e-6
        )[0]
        basis = root.matrix_basis.copy()
        basis.translation += Vector(local_delta.tolist())
        root.matrix_basis = basis
        self._update()
        self.applied_target_follow += remaining

    @staticmethod
    def _balance_delta(com, active, points, radius):
        if not active:
            return Vector()
        p = Vector((com.x, com.y, 0.0))
        a = points[active[0]].copy(); a.z = 0.0
        nearest = a
        if len(active) == 2:
            b = points[active[1]].copy(); b.z = 0.0
            ab = b - a
            nearest = a + ab * max(0.0, min(1.0, (p-a).dot(ab) / max(ab.length_squared, 1e-12)))
        delta = nearest - p
        return delta.normalized() * max(0.0, delta.length-radius) if delta.length > 1e-8 else Vector()

    def _adjust_root_for_balance(self, active, points):
        if (not self.user_has_edited_target or not active or self._authored_pelvis
                or self.scene.bodymove_balance_response <= 0):
            return 0.0
        root = self.rig.pose.bones[self.balance_root_name]
        base = root.matrix_basis.copy()
        support_radius = min(float(api().support_by_role(self.scene, s.upper()+"_FOOT").get(api().RADIUS, 0.1)) for s in active) * 0.3
        error = self._balance_delta(self._com(), active, points, support_radius)
        if error.length <= self.height * 0.001:
            return error.length
        response = self.scene.bodymove_balance_response * self.scene.bodymove_reactive_strength * 0.2
        step = Vector((error.x, error.y, 0.0)) * response
        max_step = self.height * 0.001
        if step.length > max_step:
            step *= max_step / step.length
        current_world = self._world_control_position(root)
        total_world = current_world + step - self.balance_root_initial_world - self.applied_target_follow
        total_world.z = 0.0
        max_total = self.height * 0.04
        if total_world.length > max_total:
            total_world *= max_total / total_world.length
        target_world = self.balance_root_initial_world + self.applied_target_follow + total_world
        target_world.z = current_world.z
        world_delta = np.asarray(target_world - current_world, dtype=float)
        local_delta = np.linalg.lstsq(self.translation_jacobians["balance_root"], world_delta, rcond=1e-6)[0]
        moved = base.copy()
        moved.translation += Vector(local_delta.tolist())
        root.matrix_basis = moved
        self._update()
        # Constraints and bone-local axes can mix directions; reject any
        # numerical step that introduces vertical drift.
        actual_world = self._world_control_position(root)
        if abs(actual_world.z - current_world.z) > self.height * 1e-5:
            root.matrix_basis = base
            self._update()
            return error.length
        return self._balance_delta(self._com(), active, self._active_supports()[1], support_radius).length

    def _adjust_spine_for_balance(self, active, points):
        if not self.user_has_edited_target or not active or self.scene.bodymove_balance_response <= 0:
            return 0.0
        error = self._balance_delta(self._com(), active, points, 0.0)
        if error.length <= self.height * 0.002:
            return error.length
        radius = min(float(api().support_by_role(self.scene, s.upper()+"_FOOT").get(api().RADIUS, 0.1)) for s in active) * 0.3
        best_error = self._balance_delta(self._com(), active, points, radius).length
        step_angle = np.deg2rad(1.5 * self.scene.bodymove_reactive_strength)
        max_angle = np.deg2rad(12.0)
        names = [self.scene.bodymove_map_spine, self.scene.bodymove_map_chest]
        for name in dict.fromkeys(names):
            if not name or (self._authored_torso and name == self.controller_bones["torso"]):
                continue
            bone = self.rig.pose.bones.get(name)
            if bone is None or bone.constraints:
                continue
            original = bone.matrix_basis.copy()
            best = original.copy()
            initial_rotation = self.original[name].to_quaternion()
            for axis_index in (0, 2):
                for sign in (-1.0, 1.0):
                    loc, rotation, scale = original.decompose()
                    axis = Vector((0.0, 0.0, 0.0)); axis[axis_index] = 1.0
                    trial_rotation = rotation @ Quaternion(axis, step_angle * sign)
                    if initial_rotation.rotation_difference(trial_rotation).angle > max_angle:
                        continue
                    bone.matrix_basis = Matrix.LocRotScale(loc, trial_rotation, scale)
                    self._update()
                    residual = self._balance_delta(self._com(), active, points, radius).length
                    if residual < best_error - self.height * 1e-6:
                        best_error = residual
                        best = bone.matrix_basis.copy()
            bone.matrix_basis = best
            self._update()
        return best_error

    def _adjust_arms_for_balance(self, active, points):
        if not self.user_has_edited_target or not active or self.scene.bodymove_balance_response <= 0:
            return 0.0
        radius = min(float(api().support_by_role(self.scene, s.upper()+"_FOOT").get(api().RADIUS, 0.1)) for s in active) * 0.3
        best_error = self._balance_delta(self._com(), active, points, radius).length
        if best_error <= self.height * 0.002:
            return best_error
        step_angle = np.deg2rad(1.25 * self.scene.bodymove_reactive_strength)
        max_angle = np.deg2rad(25.0)
        for side in ("left", "right"):
            if side in self.authored_shoulders or side + "_hand" in self.authored_targets:
                continue
            name = self.balance_bones.get(side+"_shoulder")
            bone = self.rig.pose.bones.get(name) if name else None
            if bone is None or bone.constraints:
                continue
            original = bone.matrix_basis.copy()
            best = original.copy()
            reference = self.original[name].to_quaternion()
            for axis_index in (0, 2):
                for sign in (-1.0, 1.0):
                    loc, rotation, scale = original.decompose()
                    axis = Vector((0.0, 0.0, 0.0)); axis[axis_index] = 1.0
                    trial_rotation = rotation @ Quaternion(axis, step_angle * sign)
                    if reference.rotation_difference(trial_rotation).angle > max_angle:
                        continue
                    bone.matrix_basis = Matrix.LocRotScale(loc, trial_rotation, scale)
                    self._update()
                    residual = self._balance_delta(self._com(), active, points, radius).length
                    if residual < best_error - self.height * 1e-6:
                        best_error = residual
                        best = bone.matrix_basis.copy()
            bone.matrix_basis = best
            self._update()
        return best_error

    def _recover_balance_pose(self, active):
        """Ease automatic balance offsets back without releasing user targets."""
        if set(active) != {"left", "right"}:
            return False
        tolerance = self.height * 0.035
        for side in ("left", "right"):
            key = side + "_foot"
            current = self._world_control_position(self.rig.pose.bones[self.controller_bones[key]])
            if (current - self.initial_world_controls[key]).length > tolerance:
                return False

        # Other authored targets stay pinned in solve(). Let only the automatic
        # balance bones recover; per-bone checks below protect user-authored bones.

        factor = 0.12
        names = [self.balance_root_name, self.scene.bodymove_map_spine, self.scene.bodymove_map_chest]
        names.extend(self.balance_bones.values())
        changed = False
        for name in dict.fromkeys(n for n in names if n):
            if name == self.balance_root_name and self._authored_pelvis:
                continue
            if name == self.balance_root_name and self.applied_target_follow.length > tolerance:
                continue
            if name in {self.scene.bodymove_map_spine, self.scene.bodymove_map_chest} and self._authored_torso:
                continue
            shoulder_side = next((key.split("_")[0] for key, value in self.balance_bones.items() if value == name), None)
            if shoulder_side in self.authored_shoulders:
                continue
            bone = self.rig.pose.bones.get(name)
            target = self.original.get(name)
            if bone is None or target is None:
                continue
            loc, rotation, scale = bone.matrix_basis.decompose()
            target_loc, target_rotation, target_scale = target.decompose()
            loc = loc.lerp(target_loc, factor)
            rotation = rotation.slerp(target_rotation, factor)
            scale = scale.lerp(target_scale, factor)
            bone.matrix_basis = Matrix.LocRotScale(loc, rotation, scale)
            changed = True
        if changed:
            self._update()
        return changed

    def _follow_hands_with_shoulders(self):
        """Let un-authored hand IK targets follow automatic shoulder motion."""
        for side in ("left", "right"):
            hand_key = side + "_hand"
            shoulder_key = side + "_shoulder"
            if hand_key in self.authored_targets:
                continue
            shoulder_name = self.balance_bones.get(shoulder_key)
            initial_shoulder = self.initial_world_shoulders.get(shoulder_key)
            if not shoulder_name or initial_shoulder is None:
                continue
            shoulder = self.rig.pose.bones[shoulder_name]
            current_shoulder = self.rig.matrix_world @ shoulder.matrix
            delta_rotation = current_shoulder.to_quaternion() @ initial_shoulder.to_quaternion().inverted()
            pivot = current_shoulder.translation
            initial_pivot = initial_shoulder.translation
            initial_hand = self.initial_world_controls[hand_key]
            target = pivot + delta_rotation @ (initial_hand - initial_pivot)
            self._place_controller(hand_key, target)
            control = self.body_controls.get(hand_key)
            if control:
                matrix = control.matrix_world.copy()
                matrix.translation = target
                control.matrix_world = matrix
                self.display[hand_key] = matrix.copy()

    def solve(self, force=False):
        if (self.scene.frame_current, self.scene.frame_subframe) != self.frame:
            raise RuntimeError("Frame changed: enable Auto-Rig Pro targets again on this pose")
        if max(abs(v) for row in self.rig.matrix_world-self.rig_world for v in row) > 1e-5:
            raise RuntimeError("Rig object moved: re-enable Auto-Rig Pro targets at its new location")
        self._update()
        edited = self._edited_targets()
        pose_edits = self._edited_pose_controls()
        if edited or pose_edits:
            self.user_has_edited_target = True
        for key, obj in edited.items():
            self.authored_targets[key] = obj.matrix_world.translation.copy()
        step_edits = dict(edited)
        step_edits.update((key, None) for key in pose_edits
                          if key in {"left_foot", "right_foot"})
        self._authored_pelvis = self._authored_pelvis or "pelvis" in edited or "pelvis" in pose_edits
        self._authored_torso = self._authored_torso or "torso" in edited or "torso" in pose_edits
        errors = {}
        for key, obj in edited.items():
            errors[key] = self._place_controller(key, obj.matrix_world.translation)
        active, points = self._active_supports()
        self._apply_reciprocal_step(step_edits, active)
        active, points = self._active_supports()
        self._recover_balance_pose(active)
        self._apply_target_follow()
        active, points = self._active_supports()
        residual = self._adjust_root_for_balance(active, points)
        if active:
            residual = self._adjust_spine_for_balance(active, self._active_supports()[1])
            residual = self._adjust_arms_for_balance(active, self._active_supports()[1])
            self._follow_hands_with_shoulders()
        # Preserve every user-placed control while the shared body follows.
        for key, target in self.authored_targets.items():
            self._place_controller(key, target)
        if self.generated_step_key and self.generated_step_target is not None:
            self._place_controller(self.generated_step_key, self.generated_step_target)
        errors = {
            key: float(np.linalg.norm(
                np.asarray(target, dtype=float) -
                np.asarray(self._world_control_position(self.rig.pose.bones[self.controller_bones[key]]), dtype=float)
            ))
            for key, target in self.authored_targets.items()
        }
        for key, obj in self.controls.items():
            self.display[key + "_foot"] = obj.matrix_world.copy()
        for key, obj in self.body_controls.items():
            self.display[key] = obj.matrix_world.copy()
        self.scene.bodymove_reactive_error = residual
        if not active:
            status = "Auto-Rig Pro | no foot support; COM correction paused"
        else:
            status = "Auto-Rig Pro | support: " + " + ".join(s.title() for s in active)
            if residual > self.height * 0.005:
                status += " | balance incomplete"
        if errors and max(errors.values()) > self.height * 0.005:
            status += " | target limited"
        if self.generated_step_key:
            status += " | counter-step: " + self.generated_step_key.split("_")[0].title()
        self.scene.bodymove_reactive_status = status
        self.last_basis = {
            key: self.rig.pose.bones[name].matrix_basis.copy()
            for key, name in {**self.controller_bones, **self.balance_bones}.items()
        }
        api().compute_analysis(self.scene, bpy.context.evaluated_depsgraph_get())
        self.scene.bodymove_analysis_valid = True
        api().tag_redraw()
        return {"supports": active, "balance_error": residual, "target_errors": errors}

    def close(self, keep=False):
        self.scene.bodymove_reactive_enabled = False
        if not keep:
            for name, matrix in self.original.items():
                bone = self.rig.pose.bones.get(name)
                if bone:
                    bone.matrix_basis = matrix
            bpy.context.view_layer.update()
        for obj in list(self.controls.values()) + list(self.body_controls.values()):
            if obj.name in bpy.data.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        if self.collection and self.collection.name in bpy.data.collections:
            bpy.data.collections.remove(self.collection)
        self.controls.clear(); self.body_controls.clear()
        self.scene.bodymove_physics_enabled = self.saved_physics
        self.scene.bodymove_reactive_status = "Pose kept (not keyframed)" if keep else "Original pose restored"
