"""Interactive FK posing: foot targets, support-aware balance and body response.

Each solve starts from the authored pose, not the previous correction. This
avoids accumulating drift while the user drags a control. Two-bone IK rotates
the leg without stretching it. This is a posing aid, not a gravity simulator.
"""
import importlib
import logging
import math

import bpy
from . import body_controls, contacts, coupled_pose, pose_tasks
from bpy.app.handlers import persistent
from mathutils import Matrix, Quaternion, Vector

TAG = '_bodymove_live_control'
_STATE = None
_BUSY = False


def api():
    return importlib.import_module(__package__)


def busy():
    return _BUSY


def pose_basis(rig):
    return {bone.name: bone.matrix_basis.copy() for bone in rig.pose.bones}


def matrix_difference(a, b):
    return max(abs(value) for row in a-b for value in row)


def update():
    bpy.context.view_layer.update()


def restore_basis(rig, values):
    for name, matrix in values.items():
        bone = rig.pose.bones.get(name)
        if bone:
            bone.matrix_basis = matrix
    update()


def world_head(rig, bone):
    return rig.matrix_world @ bone.head


def aim(rig, bone, current, desired, limit=None):
    if min(current.length, desired.length) < 1e-8 or (limit is not None and limit <= 0):
        return
    rotation = current.rotation_difference(desired)
    if limit is not None and rotation.angle > limit:
        rotation = Quaternion(rotation.axis, limit)
    pivot = world_head(rig, bone)
    transform = Matrix.Translation(pivot) @ rotation.to_matrix().to_4x4() @ Matrix.Translation(-pivot)
    bone.matrix = rig.matrix_world.inverted() @ transform @ rig.matrix_world @ bone.matrix
    update()


class LivePose:
    def __init__(self, scene):
        self.scene = scene
        self.backend = scene.bodymove_pose_backend
        self.rig = scene.bodymove_armature_object
        if self.rig is None or self.rig.type != 'ARMATURE':
            raise ValueError('Choose the character mesh and run Quick Human Setup first')
        self.pelvis = self.rig.pose.bones.get(scene.bodymove_map_pelvis)
        if self.pelvis is None:
            raise ValueError('Map the pelvis bone first')
        if any(not c.mute for bone in self.rig.pose.bones for c in bone.constraints):
            raise ValueError('Live Balance needs an FK rig without active bone constraints')
        if any(not c.mute for c in self.rig.constraints):
            raise ValueError('Bake object constraints before using Live Balance')
        if self.rig.animation_data and self.rig.animation_data.drivers:
            raise ValueError('Bake rig drivers before using Live Balance')
        self.height = max(api().source_world_height(scene.bodymove_source_object), 0.1)
        self.frame = (scene.frame_current, scene.frame_subframe)
        self.original = pose_basis(self.rig)
        self.base = pose_basis(self.rig)
        self.output = pose_basis(self.rig)
        self.rig_world = self.rig.matrix_world.copy()
        self.protected = set()
        self.legs = {}
        self.controls = {}
        self.collection = None
        self.signature = None
        self.last_supports = []
        self.solve_count = 0
        self.saved_physics = scene.bodymove_physics_enabled
        for side in ('left', 'right'):
            names = [getattr(scene, 'bodymove_map_' + side + '_' + part) for part in ('thigh', 'shin', 'foot')]
            bones = [self.rig.pose.bones.get(name) for name in names]
            if any(bone is None for bone in bones):
                raise ValueError('Map both thighs, shins and feet before enabling Live Balance')
            thigh, shin, foot = bones
            if shin.parent != thigh or foot.parent != shin:
                raise ValueError('Live Balance needs a direct thigh > shin > foot FK chain')
            a, b, c = (world_head(self.rig, bone) for bone in bones)
            if min((b-a).length, (c-b).length) < 1e-5:
                raise ValueError('Invalid leg lengths')
            ankle_gap = (self.rig.matrix_world @ shin.tail - c).length
            allowed_ankle_gap = self.height * (
                0.025 if getattr(scene, 'bodymove_rig_profile', '') == 'GENERIC' else 0.01
            )
            if ankle_gap > allowed_ankle_gap:
                raise ValueError(
                    'The foot must begin at the shin tip '
                    f'(gap {ankle_gap:.3f} m; maximum {allowed_ankle_gap:.3f} m)'
                )
            axis = (c-a).normalized()
            bend = (b-a) - axis * (b-a).dot(axis)
            if bend.length < self.height*0.005:
                bend = self.rig.matrix_world.to_quaternion() @ Vector((0, -1, 0))
            role = side.upper() + '_FOOT'
            support = api().support_by_role(scene, role)
            if support is None:
                raise ValueError('Create a support point for each foot first')
            matrix = self.rig.matrix_world @ foot.matrix
            self.legs[side] = dict(bones=bones, lengths=((b-a).length, (c-b).length),
                                   bend=bend.normalized(), initial=matrix.copy(), support=support,
                                   contact_offset=matrix.inverted() @ support.matrix_world.translation)

        body_controls.initialize(self)

    def create_controls(self):
        self.collection = bpy.data.collections.new('BodyMove Live Feet')
        self.collection[TAG] = True
        self.scene.collection.children.link(self.collection)
        for side, leg in self.legs.items():
            control = bpy.data.objects.new('BM Foot ' + side.title() + ' - move me', None)
            self.collection.objects.link(control)
            control[TAG] = True
            control['bodymove_live_side'] = side
            control.matrix_world = leg['initial']
            control.empty_display_type = 'CUBE'
            control.empty_display_size = self.height*0.045
            control.show_in_front = True
            control.show_name = True
            control.hide_render = True
            control.color = (0.15, 0.75, 1, 1) if side == 'left' else (1, 0.4, 0.1, 1)
            self.controls[side] = control
        if self.backend == 'WHOLE_BODY':
            pose_tasks.create(self)
        else:
            body_controls.create(self)
        self.scene.bodymove_physics_enabled = False
        self.scene.bodymove_reactive_enabled = True
        if self.backend != 'WHOLE_BODY':
            self.scene.bodymove_reactive_status = 'Ready - move a foot control with G'
        update()

    def signature_now(self):
        if self.backend == 'WHOLE_BODY':
            return pose_tasks.signature(self)
        settings = (self.scene.bodymove_reactive_strength, self.scene.bodymove_reactive_arms,
                    self.scene.bodymove_reactive_torso, self.scene.bodymove_reactive_hip_rise,
                    self.scene.bodymove_reactive_left_arm, self.scene.bodymove_reactive_right_arm,
                    self.scene.bodymove_reactive_pelvis, self.scene.bodymove_reach_shoulder, self.scene.bodymove_reach_torso, self.scene.bodymove_coupled_pose, self.scene.bodymove_balance_response)
        return tuple(round(value, 6) for control in self.controls.values() for row in control.matrix_world for value in row) + settings + body_controls.signature(self)

    def supports(self):
        scene = self.scene
        active, points = [], {}
        for side, leg in self.legs.items():
            point = self.controls[side].matrix_world @ leg['contact_offset']
            points[side] = point
            if scene.bodymove_ground_object:
                closest = api().ground_closest_world(scene, point, bpy.context.evaluated_depsgraph_get())
                distance = (point-closest).length
            else:
                distance = abs(point.z-scene.bodymove_virtual_ground_z)
            threshold = scene.bodymove_contact_tolerance * (1.25 if side in self.last_supports else 1.0)
            if distance <= threshold:
                active.append(side)
        return active, points

    def solve_leg(self, side):
        leg = self.legs[side]
        thigh, shin, foot = leg['bones']
        a = world_head(self.rig, thigh)
        target = self.controls[side].matrix_world.translation
        displacement = target-a
        l1, l2 = leg['lengths']
        direction = displacement.normalized() if displacement.length > 1e-8 else Vector((0,0,-1))
        distance = max(abs(l1-l2)+1e-7, min(displacement.length, (l1+l2)*0.999999))
        reach = a + direction*distance
        along = (l1*l1-l2*l2+distance*distance)/(2*distance)
        across = math.sqrt(max(0, l1*l1-along*along))
        bend = leg['bend']-direction*leg['bend'].dot(direction)
        if bend.length < 1e-8:
            bend = direction.cross(Vector((1,0,0)))
            if bend.length < 1e-8:
                bend = direction.cross(Vector((0,1,0)))
        knee = a + direction*along + bend.normalized()*across
        aim(self.rig, thigh, world_head(self.rig, shin)-a, knee-a)
        b = world_head(self.rig, shin)
        aim(self.rig, shin, world_head(self.rig, foot)-b, reach-b)
        matrix = self.controls[side].matrix_world.copy()
        matrix.translation = world_head(self.rig, foot)
        foot.matrix = self.rig.matrix_world.inverted() @ matrix
        update()
        return (world_head(self.rig, foot)-target).length

    def com(self):
        graph = bpy.context.evaluated_depsgraph_get()
        weighted, total = Vector(), 0.0
        for point in api().helpers(self.scene, api().MASS_POINT):
            mass = max(0.0, float(point.get(api().MASS, 0)))
            weighted += point.evaluated_get(graph).matrix_world.translation * mass
            total += mass
        if total <= 0:
            raise ValueError('No valid mass points')
        return weighted/total

    def target_error(self, active, points):
        com = self.com()
        p = Vector((com.x, com.y, 0))
        if not active:
            return Vector()
        a = points[active[0]].copy()
        a.z = 0
        q = a
        if len(active) == 2:
            b = points[active[1]].copy()
            b.z = 0
            ab = b-a
            q = a + ab*max(0, min(1, (p-a).dot(ab)/max(ab.length_squared, 1e-12)))
        radius = min(float(self.legs[s]['support'].get(api().RADIUS, 0.1)) for s in active)*0.3
        delta = q-p
        return delta.normalized()*max(0, delta.length-radius) if delta.length > 1e-8 else Vector()

    def translate_pelvis(self, delta, active):
        if self.pelvis.name in (self.protected | self.driven) or self.scene.bodymove_reactive_pelvis <= 0:
            return
        origin = world_head(self.rig, self.pelvis)
        wanted = origin + delta
        # Respect each planted leg's reach. Sideways weight transfer may need
        # lower hips; the assistant must not stretch a straight supporting leg.
        for _ in range(4):
            for side in active:
                leg = self.legs[side]
                offset = world_head(self.rig, leg['bones'][0])-origin
                target = self.controls[side].matrix_world.translation
                displacement = wanted+offset-target
                reach = sum(leg['lengths'])*0.99999
                if displacement.length > reach:
                    wanted = target + displacement.normalized()*reach-offset
        api().translate_pose_bone_world(self.rig, self.pelvis.name, wanted-origin)
        update()

    def rotate_world(self, bone, axis, angle):
        if axis.length < 1e-8 or abs(angle) < 1e-8:
            return
        pivot = world_head(self.rig, bone)
        rotation = Quaternion(axis.normalized(), angle).to_matrix().to_4x4()
        transform = Matrix.Translation(pivot) @ rotation @ Matrix.Translation(-pivot)
        bone.matrix = self.rig.matrix_world.inverted() @ transform @ self.rig.matrix_world @ bone.matrix
        update()

    def balance_arm(self, bone, active, points, influence=1.0):
        # A bounded coordinate search measures the actual mass-point COM. Each
        # shoulder is independent; no mirrored pose or foot-height trigger.
        error = self.target_error(active, points)
        if error.length <= self.height*0.001:
            return
        amount = min(1.0, error.length/(self.height*0.06))
        angle = math.radians(self.scene.bodymove_reactive_arms)*amount*self.response_strength*self.scene.bodymove_balance_response*influence
        if angle < 1e-6:
            return
        original = bone.matrix_basis.copy()
        best = original.copy()
        best_error = error.length
        current = self.rig.matrix_world.to_3x3() @ (bone.tail-bone.head)
        # Prefer upward relief when equally good solutions exist, then allow
        # forward/backward and lateral compensation for the current pose.
        axes = (current.cross(Vector((0,0,1))), current.cross(error), Vector((0,0,1)))
        for axis in axes:
            if axis.length < 1e-8:
                continue
            for sign in (1, -1):
                bone.matrix_basis = original
                update()
                self.rotate_world(bone, axis, angle*sign)
                residual = self.target_error(active, points).length
                if residual < best_error-self.height*1e-6:
                    best_error = residual
                    best = bone.matrix_basis.copy()
        bone.matrix_basis = best
        update()

    def react_upper_body(self, active, points):
        scene = self.scene
        strength = self.response_strength*scene.bodymove_balance_response
        if not active or strength <= 0:
            return
        error = self.target_error(active, points)
        if error.length <= self.height*0.001:
            return
        # Rotate the pelvis a little before translating it. Re-solving both
        # legs distributes the change through the supporting knee and ankle.
        # Accept only feasible changes that improve the measured balance.
        if self.pelvis.name not in (self.protected | self.driven) and scene.bodymove_reactive_hip_rise > 0 and scene.bodymove_reactive_pelvis > 0:
            before = pose_basis(self.rig)
            old_error = error.length
            api().rotate_pose_bone_world_lean(self.rig, self.pelvis.name, error,
                0.8*strength*scene.bodymove_reactive_hip_rise*scene.bodymove_reactive_pelvis, math.radians(8))
            update()
            self.translate_pelvis(Vector(), active)
            residuals = {side: self.solve_leg(side) for side in self.legs}
            if (self.target_error(active, points).length >= old_error or
                    max(residuals[side] for side in active) > self.height*0.001):
                restore_basis(self.rig, before)
        used = {self.pelvis.name}
        for field, weight in (('spine', 0.7), ('chest', 0.5)):
            name = getattr(scene, 'bodymove_map_'+field)
            bone = self.rig.pose.bones.get(name)
            if bone and name not in (self.protected | self.driven) and name not in used:
                error = self.target_error(active, points)
                before = bone.matrix_basis.copy()
                api().rotate_pose_bone_world_lean(self.rig, name, error,
                    weight*strength*scene.bodymove_reactive_torso, math.radians(12))
                update()
                if self.target_error(active, points).length >= error.length:
                    bone.matrix_basis = before
                    update()
                used.add(name)
        for side in ('left', 'right'):
            name = getattr(scene, 'bodymove_map_'+side+'_upper_arm')
            bone = self.rig.pose.bones.get(name)
            if bone and name not in (self.protected | self.driven) and name not in used:
                self.balance_arm(bone, active, points, getattr(scene, 'bodymove_reactive_'+side+'_arm'))
                used.add(name)

    def solve(self, force=False):
        scene, rig = self.scene, self.rig
        if (scene.frame_current, scene.frame_subframe) != self.frame:
            raise RuntimeError('Frame changed: enable Live Balance on the new pose')
        if matrix_difference(rig.matrix_world, self.rig_world) > 1e-5:
            raise RuntimeError('Rig object moved: re-enable Live Balance at its new location')
        if len(self.controls) != 2 or any(control.name not in scene.objects for control in list(self.controls.values())+list(self.body_controls.values())):
            raise RuntimeError('A foot control was removed: Live Balance stopped')
        if self.backend == 'WHOLE_BODY':
            return pose_tasks.solve(self, force=force)
        changed = False
        for bone in rig.pose.bones:
            if matrix_difference(bone.matrix_basis, self.output[bone.name]) > 2e-5:
                # Once the user edits a bone, its visible local pose becomes
                # authored input. Do not remove assistance and make it jump.
                self.base[bone.name] = bone.matrix_basis.copy()
                self.protected.add(bone.name)
                changed = True
        signature = self.signature_now()
        if not force and not changed and signature == self.signature:
            return False
        self.signature = signature
        restore_basis(rig, self.base)
        self.contact_limited = False
        direct_base = pose_basis(rig)
        body_controls.prepare(self)
        active, points = self.supports()
        self.last_supports = active
        for side in self.legs:
            self.solve_leg(side)
        contacts.accept(self, direct_base)
        before_error = self.target_error(active, points).length
        before_reaction = pose_basis(rig)
        self.react_upper_body(active, points)
        body_controls.solve_arms(self)
        contacts.accept(self, before_reaction)
        for _ in range(5):
            error = self.target_error(active, points)
            if error.length < self.height*0.001 or not active:
                break
            delta = error*self.response_strength*scene.bodymove_balance_response*scene.bodymove_reactive_pelvis*0.8
            if delta.length > self.height*0.035:
                delta = delta.normalized()*self.height*0.035
            before_step = pose_basis(rig)
            self.translate_pelvis(delta, active)
            for side in self.legs:
                self.solve_leg(side)
            body_controls.solve_arms(self)
            contacts.accept(self, before_step)
        coupled_pose.solve(self, active, points)
        hand_errors = body_controls.solve_arms(self)
        residuals = {side: self.solve_leg(side) for side in self.legs}
        api().reset_support_tracking(scene)
        api().compute_analysis(scene, bpy.context.evaluated_depsgraph_get())
        scene.bodymove_analysis_valid = True
        scene.bodymove_reactive_error = self.target_error(active, points).length
        scene.bodymove_reactive_contact_error = max((residuals[s] for s in active), default=0.0)
        if not active:
            status = 'No ground support - body compensation paused'
        elif max(residuals.values()) > self.height*0.005:
            status = 'Foot target out of reach - move it closer'
        else:
            status = 'Support: ' + ' + '.join(side.title() for side in active)
            if scene.bodymove_reactive_error > self.height*0.005:
                status += ' - balance incomplete; check response limits'
        if max(hand_errors.values(), default=0) > self.height*0.005:
            status += ' - hand target out of reach'
        if self.contact_limited:
            status += " - motion limited by fixed contacts"
        scene.bodymove_reactive_status = status
        self.output = pose_basis(rig)
        self.solve_count += 1
        api().tag_redraw()
        return {'supports': active, 'before_error': before_error,
                'after_error': scene.bodymove_reactive_error, 'foot_errors': residuals, 'hand_errors': hand_errors}

    def close(self, keep=False):
        self.scene.bodymove_reactive_enabled = False
        if not keep:
            restore_basis(self.rig, self.original)
        for obj in list(self.controls.values())+list(self.body_controls.values()):
            try:
                if obj.name in bpy.data.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
            except ReferenceError:
                pass
        if self.collection and self.collection.name in bpy.data.collections:
            bpy.data.collections.remove(self.collection)
        self.controls.clear()
        self.scene.bodymove_physics_enabled = self.saved_physics
        api().reset_support_tracking(self.scene)
        api().compute_analysis(self.scene, bpy.context.evaluated_depsgraph_get())
        self.scene.bodymove_reactive_status = 'Pose kept (not keyframed)' if keep else 'Original pose restored'


def enable(scene):
    global _STATE, _BUSY
    if _STATE:
        disable(keep=True)
    _BUSY = True
    try:
        on_load(None)  # Explicit activation clears abandoned controls.
        rig = scene.bodymove_armature_object
        if api().detect_rig_profile(rig) == "AUTO_RIG_PRO":
            from . import arp_live
            state = arp_live.LivePose(scene)
        else:
            state = LivePose(scene)
        _STATE = state
        state.create_controls()
        state.solve(force=True)
        return state
    except Exception:
        if _STATE:
            _STATE.close(keep=False)
        _STATE = None
        raise
    finally:
        _BUSY = False


def disable(keep=False):
    global _STATE, _BUSY
    if _STATE is None:
        return
    _BUSY = True
    state, _STATE = _STATE, None
    try:
        state.close(keep=keep)
    finally:
        _BUSY = False


def tick():
    global _BUSY, _STATE
    if _BUSY or _STATE is None:
        return 0.05
    try:
        if bpy.context.scene != _STATE.scene:
            return 0.05
        _BUSY = True
        _STATE.solve()
    except Exception as exc:
        scene = bpy.context.scene
        logging.getLogger(__name__).warning('Live Balance stopped: %s', exc)
        # A modal transform may still own these objects. Do not delete them.
        _STATE = None
        scene.bodymove_reactive_enabled = False
        scene.bodymove_physics_enabled = False
        scene.bodymove_reactive_status = str(exc)
    finally:
        _BUSY = False
    return 0.05


@persistent
def on_load(_):
    global _STATE
    _STATE = None
    for obj in list(bpy.data.objects):
        if obj.get(TAG):
            bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.data.collections):
        if collection.get(TAG) and not collection.objects:
            bpy.data.collections.remove(collection)
    for scene in bpy.data.scenes:
        scene.bodymove_reactive_enabled = False


@persistent
def on_undo_redo(_):
    # Blender undo can replace every Scene/Object RNA wrapper. Never reuse a
    # cached LivePose against that new database or restore its stale matrices.
    global _STATE
    _STATE = None
    for scene in bpy.data.scenes:
        scene.bodymove_reactive_enabled = False
        scene.bodymove_physics_enabled = False
        scene.bodymove_reactive_status = 'Undo/redo: enable Live Balance again'


class BODYMOVE_OT_ReactiveStart(bpy.types.Operator):
    bl_idname = 'bodymove.reactive_start'
    bl_label = 'Enable Live Balance'
    bl_options = {'REGISTER', 'UNDO'}
    backend: bpy.props.EnumProperty(items=(('WHOLE_BODY','Whole Body','General targets'),('LEGACY','Legacy','0.15 solver')), default='WHOLE_BODY')

    def execute(self, context):
        try:
            if not context.scene.bodymove_analysis_valid:
                result = bpy.ops.bodymove.quick_human_setup()
                if result != {'FINISHED'}:
                    return {'CANCELLED'}
            context.scene.bodymove_pose_backend = self.backend
            enable(context.scene)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class BODYMOVE_OT_ReactiveSelect(bpy.types.Operator):
    bl_idname = 'bodymove.reactive_select'
    bl_label = 'Select body control'
    side: bpy.props.StringProperty()

    def execute(self, context):
        if _STATE is None:
            return {'CANCELLED'}
        active = context.view_layer.objects.active
        if active and active.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        obj = _STATE.controls.get(self.side) or _STATE.body_controls.get(self.side)
        if obj is None:
            self.report({'WARNING'}, 'This control needs a mapped direct arm chain')
            return {'CANCELLED'}
        api().select_only(context, obj)
        return {'FINISHED'}


class BODYMOVE_OT_ReactiveFinish(bpy.types.Operator):
    bl_idname = 'bodymove.reactive_finish'
    bl_label = 'Finish Live Balance'
    bl_options = {'REGISTER', 'UNDO'}
    keep: bpy.props.BoolProperty(default=True)

    def execute(self, context):
        disable(keep=self.keep)
        return {'FINISHED'}


CLASSES = pose_tasks.CLASSES + (contacts.BODYMOVE_OT_HandContact, BODYMOVE_OT_ReactiveStart, BODYMOVE_OT_ReactiveSelect, BODYMOVE_OT_ReactiveFinish)
PROPS = ('bodymove_reactive_enabled', 'bodymove_reactive_strength', 'bodymove_reactive_arms',
         'bodymove_reactive_torso', 'bodymove_reactive_hip_rise', 'bodymove_reactive_status',
         'bodymove_reactive_error', 'bodymove_reactive_contact_error',
         'bodymove_reactive_left_arm', 'bodymove_reactive_right_arm', 'bodymove_reactive_pelvis',
         'bodymove_reach_shoulder', 'bodymove_reach_torso', 'bodymove_coupled_pose', 'bodymove_balance_response',
         'bodymove_pose_backend', 'bodymove_whole_freedom', 'bodymove_whole_bone')


def draw(layout, scene):
    box = layout.box()
    box.label(text='Live Balance - body controls', icon='POSE_HLT')
    if not scene.bodymove_reactive_enabled:
        box.operator('bodymove.reactive_start', text='Enable Whole Body (beta)', icon='PLAY').backend = 'WHOLE_BODY'
        box.operator('bodymove.reactive_start', text='Legacy Live Balance').backend = 'LEGACY'
    elif _STATE is not None and _STATE.backend == 'WHOLE_BODY':
        pose_tasks.draw(box, _STATE)
        return
    elif _STATE is not None and _STATE.backend == 'ARP':
        row = box.row(align=True)
        row.operator('bodymove.reactive_select', text='Left Foot').side = 'left'
        row.operator('bodymove.reactive_select', text='Right Foot').side = 'right'
        for pair in (('left_hand', 'right_hand'), ('pelvis', 'torso')):
            row = box.row(align=True)
            for key in pair:
                row.operator('bodymove.reactive_select', text=key.replace('_', ' ').title()).side = key
        box.label(text='Move a BM target or an ARP IK bone in Pose Mode with G')
        box.prop(scene, 'bodymove_balance_response', text='COM Balance Assistance', slider=True)
        box.prop(scene, 'bodymove_reactive_strength', text='Body Response', slider=True)
        row = box.row(align=True)
        row.operator('bodymove.reactive_finish', text='Keep Pose', icon='CHECKMARK').keep = True
        row.operator('bodymove.reactive_finish', text='Cancel', icon='LOOP_BACK').keep = False
    else:
        row = box.row(align=True)
        row.operator('bodymove.reactive_select', text='Left Foot').side = 'left'
        row.operator('bodymove.reactive_select', text='Right Foot').side = 'right'
        for pair in (('left_hand','right_hand'),('left_elbow','right_elbow'),('pelvis','torso')):
            row = box.row(align=True)
            for key in pair:
                cell = row.row()
                cell.enabled = _STATE is not None and key in _STATE.body_controls
                cell.operator('bodymove.reactive_select', text=key.replace('_',' ').title()).side = key
        selected = bpy.context.view_layer.objects.active
        if selected is not None and selected.get(TAG) and body_controls.INFLUENCE in selected:
            box.label(text=selected.name)
            box.prop(selected, '["bodymove_influence"]', text='Influence on Body', slider=True)
            if _STATE and any(selected == _STATE.body_controls.get(side+'_hand') for side in _STATE.arms):
                box.prop(selected, '["bodymove_target_priority"]', text='Hand Target Priority')
        box.label(text='G: move target / R: rotate hand, pelvis, torso')
        box.label(text='Body assistance')
        box.prop(scene, 'bodymove_balance_response', text='Balance Assistance', slider=True)
        box.prop(scene, 'bodymove_coupled_pose', text='Whole Body Reach + Balance')
        contacts.draw(box, _STATE)
        box.prop(scene, 'bodymove_reach_shoulder', text='Shoulder Reach')
        box.prop(scene, 'bodymove_reach_torso', text='Torso Reach')
        box.prop(scene, 'bodymove_reactive_strength', text='Body Response')
        box.prop(scene, 'bodymove_reactive_left_arm', text='Left Arm Response')
        box.prop(scene, 'bodymove_reactive_right_arm', text='Right Arm Response')
        box.prop(scene, 'bodymove_reactive_pelvis', text='Pelvis Response')
        box.label(text='0 = no local correction (not a world pin)')
        box.prop(scene, 'bodymove_reactive_arms', text='Arm Response Limit')
        box.prop(scene, 'bodymove_reactive_torso', text='Torso Response')
        box.prop(scene, 'bodymove_reactive_hip_rise', text='Pelvis Rotation')
        row = box.row(align=True)
        row.operator('bodymove.reactive_finish', text='Keep Pose', icon='CHECKMARK').keep = True
        row.operator('bodymove.reactive_finish', text='Cancel', icon='LOOP_BACK').keep = False
    box.label(text=scene.bodymove_reactive_status)


def register():
    # Programmatic legacy callers retain 0.15 behavior; the main UI starts Whole Body.
    bpy.types.Scene.bodymove_pose_backend = bpy.props.EnumProperty(items=(('LEGACY','Legacy','0.15 solver'),('WHOLE_BODY','Whole Body','General targets')), default='LEGACY')
    bpy.types.Scene.bodymove_whole_freedom = bpy.props.FloatProperty(default=1, min=0, max=1, description='Freedom of all solver joints; zero freezes the current pose')
    bpy.types.Scene.bodymove_whole_bone = bpy.props.StringProperty()
    bpy.types.Scene.bodymove_balance_response = bpy.props.FloatProperty(default=1,min=0,max=1,description='COM balance assistance, independent of hand reach')
    bpy.types.Scene.bodymove_coupled_pose = bpy.props.BoolProperty(default=True, description='Jointly refine hand reach and COM balance while preserving planted contacts')
    for part in ('shoulder', 'torso'):
        setattr(bpy.types.Scene, 'bodymove_reach_'+part, bpy.props.FloatProperty(default=1, min=0, max=1, description='Bounded assistance when an edited hand approaches full arm reach'))
    for part in ('left_arm', 'right_arm', 'pelvis'):
        setattr(bpy.types.Scene, 'bodymove_reactive_'+part, bpy.props.FloatProperty(
            default=1, min=0, max=1, description='Local assistance weight. Zero preserves the authored local pose; parent motion still applies.'))
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.bodymove_reactive_enabled = bpy.props.BoolProperty(default=False, options={'SKIP_SAVE'})
    bpy.types.Scene.bodymove_reactive_strength = bpy.props.FloatProperty(default=0.8, min=0, max=1)
    bpy.types.Scene.bodymove_reactive_arms = bpy.props.FloatProperty(default=40, min=0, max=75, description='Maximum shoulder correction in degrees; applied only when it improves balance')
    bpy.types.Scene.bodymove_reactive_torso = bpy.props.FloatProperty(default=1, min=0, max=3)
    # Keep the RNA identifier so existing files retain their setting.
    bpy.types.Scene.bodymove_reactive_hip_rise = bpy.props.FloatProperty(default=1, min=0, max=2, description='Pelvis lean contribution; feet stay constrained to their targets')
    bpy.types.Scene.bodymove_reactive_status = bpy.props.StringProperty(default='Enable, then move a foot control')
    bpy.types.Scene.bodymove_reactive_error = bpy.props.FloatProperty(default=0)
    bpy.types.Scene.bodymove_reactive_contact_error = bpy.props.FloatProperty(default=0)
    if not bpy.app.timers.is_registered(tick):
        bpy.app.timers.register(tick, first_interval=0.05, persistent=True)
    if on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load)
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if on_undo_redo not in handlers:
            handlers.append(on_undo_redo)


def unregister():
    try:
        disable(keep=True)
    except ReferenceError:
        on_load(None)
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if on_undo_redo in handlers:
            handlers.remove(on_undo_redo)
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    if on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load)
    for prop in PROPS:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
