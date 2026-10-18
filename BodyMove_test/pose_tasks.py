"""Viewport adapter for general whole-body targets. Numerical solve is bpy-free."""
import time
import bpy
import numpy as np
from mathutils import Matrix

MODE = 'bodymove_pose_mode'
BONE = 'bodymove_pose_bone'
PRIORITY = 'bodymove_target_priority'
ORIENTATION = 'bodymove_pose_orientation'
MOBILITY = 'bodymove_joint_mobility'


def controls(state):
    return dict(state.controls, **state.body_controls)


def actual_matrix(state, name):
    return state.rig.matrix_world @ state.rig.pose.bones[name].matrix


def configure(state, key, obj, name, mode='FOLLOW'):
    obj[BONE] = name
    obj[MODE] = mode
    obj[PRIORITY] = 1.0
    obj[ORIENTATION] = False
    obj.id_properties_ui(PRIORITY).update(min=0.1, max=10.0,
        description='Importance of this target, independent of all other targets; pins take priority')
    obj.lock_scale = (True,)*3
    obj.show_name = False
    obj.lock_location = (mode == 'PIN',)*3
    obj.lock_rotation = (mode == 'PIN',)*3
    obj.lock_rotations_4d = True
    obj.lock_rotation_w = mode == 'PIN'
    state.whole_display[key] = obj.matrix_world.copy()
    if mode == 'PIN':
        state.whole_pins[key] = obj.matrix_world.copy()


def add_control(state, key, bone_name):
    from . import reactive as r
    if bone_name not in state.rig.pose.bones:
        raise ValueError('Choose a bone belonging to this character')
    if key in controls(state):
        return controls(state)[key]
    obj = bpy.data.objects.new('BM '+key.replace('_', ' ').replace('bone:', ''), None)
    state.collection.objects.link(obj)
    obj[r.TAG] = True
    obj.matrix_world = actual_matrix(state, bone_name)
    obj.empty_display_type = 'SPHERE'
    obj.empty_display_size = state.height*0.025
    obj.show_in_front = True
    obj.hide_render = True
    state.body_controls[key] = obj
    configure(state, key, obj, bone_name)
    return obj


def create(state):
    from . import reactive as r
    from .skeleton_model import SkeletonModel
    state.whole_display = {}
    state.whole_pins = {}
    state.whole_result = None
    state.whole_model = SkeletonModel.from_blender(state.rig, state.pelvis.name,
        r.api().mass_point_bone_offsets(state.scene, state.rig))
    state.whole_rotations, state.whole_translation = state.whole_model.zero_pose()
    for bone in state.rig.pose.bones:
        if MOBILITY not in bone:
            bone[MOBILITY] = 1.0
            bone.id_properties_ui(MOBILITY).update(min=0.0, max=1.0,
                description='Freedom of this local joint; zero is a local lock, not a world pin')
    for side, leg in state.legs.items():
        configure(state, side, state.controls[side], leg['bones'][2].name, 'PIN')
    mapping = r.api().generic_human_bone_map(state.rig)
    names = {'pelvis': state.pelvis.name,
             'torso': state.scene.bodymove_map_chest,
             'head': state.scene.bodymove_map_head,
             'neck': mapping.get('NECK', '')}
    for side in ('left', 'right'):
        names[side+'_knee'] = state.legs[side]['bones'][1].name
        for part, category in (('shoulder', 'UPPER_ARM'), ('elbow', 'FOREARM'), ('hand', 'HAND')):
            names[side+'_'+part] = mapping.get(side.upper()+'_'+category, '')
    for key, name in names.items():
        if name and name in state.rig.pose.bones:
            add_control(state, key, name)
    state.collection.name = 'BodyMove Whole Body Controls'
    state.scene.bodymove_reactive_status = 'Whole Body: drag any sphere; set a pinned foot to Target to move it'


def set_mode(state, key, mode):
    if mode not in {'FOLLOW', 'TARGET', 'PIN'}:
        raise ValueError('Unknown target mode')
    obj = controls(state)[key]
    obj.matrix_world = actual_matrix(state, obj[BONE])
    obj[MODE] = mode
    obj.lock_location = (mode == 'PIN',)*3
    obj.lock_rotation = (mode == 'PIN',)*3
    obj.lock_rotations_4d = True
    obj.lock_rotation_w = mode == 'PIN'
    if mode == 'PIN':
        state.whole_pins[key] = obj.matrix_world.copy()
    else:
        state.whole_pins.pop(key, None)
    if mode == 'FOLLOW':
        obj[ORIENTATION] = False
    state.whole_display[key] = obj.matrix_world.copy()
    state.signature = None
    bpy.context.view_layer.update()


def signature(state):
    values = [state.scene.bodymove_whole_freedom, state.scene.bodymove_balance_response]
    for key, obj in controls(state).items():
        values.extend((key, obj[MODE], float(obj[PRIORITY]), bool(obj[ORIENTATION])))
        values.extend(round(float(x), 6) for row in obj.matrix_world for x in row)
    values.extend(float(b.get(MOBILITY, 1.0)) for b in state.rig.pose.bones)
    return tuple(values)


def _rotation(matrix):
    return np.array(matrix.to_quaternion().to_matrix(), dtype=float)


def solve(state, force=False):
    from . import reactive as r
    from .whole_body_solver import Target, solve as numerical_solve
    started = time.perf_counter()
    # Direct Pose Mode edits become regular tasks, including orientation.
    edited = []
    for bone in state.rig.pose.bones:
        if r.matrix_difference(bone.matrix_basis, state.output[bone.name]) > 2e-5:
            edited.append((bone.name, actual_matrix(state, bone.name).copy()))
    for name, desired in edited:
        existing = next((k for k, o in controls(state).items() if o[BONE] == name), None)
        key = existing or 'bone:'+name
        obj = controls(state).get(key) or add_control(state, key, name)
        if obj[MODE] != 'PIN':
            obj[MODE] = 'TARGET'
            obj[ORIENTATION] = True
            obj.matrix_world = desired
    for key, obj in controls(state).items():
        before = state.whole_display[key]
        if obj[MODE] == 'PIN':
            obj.matrix_world = state.whole_pins[key]
        elif r.matrix_difference(obj.matrix_world, before) > 2e-5:
            obj[MODE] = 'TARGET'
            if before.to_quaternion().rotation_difference(obj.matrix_world.to_quaternion()).angle > 1e-4:
                obj[ORIENTATION] = True
    current_signature = signature(state)
    if not force and not edited and state.signature == current_signature:
        return False
    model = state.whole_model
    target_keys, tasks = [], []
    for key, obj in controls(state).items():
        if obj[MODE] == 'FOLLOW':
            continue
        matrix = state.whole_pins[key] if obj[MODE] == 'PIN' else obj.matrix_world
        orientation = _rotation(matrix) if obj[MODE] == 'PIN' or obj[ORIENTATION] else None
        tasks.append(Target(bone=model.index[obj[BONE]],
            position=np.array(matrix.translation, dtype=float), orientation=orientation,
            weight=float(obj[PRIORITY]), fixed=obj[MODE] == 'PIN'))
        target_keys.append(key)
    active, points = state.supports()
    active = [s for s in active if state.controls[s][MODE] == 'PIN']
    state.last_supports = active
    radius = min((float(state.legs[s]['support'].get(r.api().RADIUS, 0.1))*0.3 for s in active), default=0.0)
    mobility = np.ones(3+3*model.n)
    for name, i in model.index.items():
        mobility[3+3*i:6+3*i] = float(state.rig.pose.bones[name].get(MOBILITY, 1.0))
    mobility[:3] = float(state.rig.pose.bones[state.pelvis.name].get(MOBILITY, 1.0))
    mobility *= state.scene.bodymove_whole_freedom
    before_error = state.target_error(active, points).length if active else 0.0
    result = numerical_solve(model, tasks, [np.array(points[s], dtype=float) for s in active], radius,
        height=state.height, rotations=state.whole_rotations, translation=state.whole_translation,
        mobility=mobility, balance_weight=state.scene.bodymove_balance_response, max_iterations=30 if force else 10)
    if not (np.isfinite(result['rotations']).all() and np.isfinite(result['translation']).all()):
        raise RuntimeError('Whole Body rejected a non-finite pose')
    model.apply_to_blender(state.rig, result['rotations'], result['translation'])
    state.whole_rotations = result['rotations'].copy()
    state.whole_translation = result['translation'].copy()
    # Never overwrite a requested target with the achieved pose.
    for key, obj in controls(state).items():
        if obj[MODE] == 'FOLLOW':
            obj.matrix_world = actual_matrix(state, obj[BONE])
        state.whole_display[key] = obj.matrix_world.copy()
    r.update()
    errors = {key: float(value) for key, value in zip(target_keys, result['target_errors'])}
    r.api().reset_support_tracking(state.scene)
    r.api().compute_analysis(state.scene, bpy.context.evaluated_depsgraph_get())
    state.scene.bodymove_analysis_valid = True
    state.scene.bodymove_reactive_error = state.target_error(active, points).length if active else 0.0
    state.scene.bodymove_reactive_contact_error = float(result['contact_error'])
    status = 'Whole Body | '+('Support: '+' + '.join(s.title() for s in active) if active else 'No ground support')
    if result['contact_error'] > state.height*0.0005:
        status += ' | pin limited'
    if state.scene.bodymove_reactive_error > state.height*0.001:
        status += ' | balance incomplete'
    if max(errors.values(), default=0) > state.height*0.005:
        status += ' | target limited'
    state.scene.bodymove_reactive_status = status
    state.output = r.pose_basis(state.rig)
    state.signature = signature(state)
    state.solve_count += 1
    result['total_seconds'] = time.perf_counter()-started
    result['target_keys'] = target_keys
    state.whole_result = result
    r.api().tag_redraw()
    feet = {s: (r.world_head(state.rig, state.legs[s]['bones'][2])-o.matrix_world.translation).length for s, o in state.controls.items()}
    hands = {s: errors.get(s+'_hand', 0.0) for s in state.arms}
    return dict(supports=active, before_error=before_error, after_error=state.scene.bodymove_reactive_error,
                foot_errors=feet, hand_errors=hands, target_errors=errors)


class BODYMOVE_OT_WholeMode(bpy.types.Operator):
    bl_idname = 'bodymove.whole_mode'
    bl_label = 'Set body target mode'
    key: bpy.props.StringProperty()
    mode: bpy.props.EnumProperty(items=tuple((v, v.title(), '') for v in ('FOLLOW', 'TARGET', 'PIN')))

    def execute(self, context):
        from . import reactive as r
        state = r._STATE
        if state is None or state.backend != 'WHOLE_BODY' or self.key not in controls(state):
            return {'CANCELLED'}
        r._BUSY = True
        try:
            set_mode(state, self.key, self.mode)
            solve(state, force=True)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        finally:
            r._BUSY = False
        return {'FINISHED'}


class BODYMOVE_OT_WholeAddBone(bpy.types.Operator):
    bl_idname = 'bodymove.whole_add_bone'
    bl_label = 'Add bone control'

    def execute(self, context):
        from . import reactive as r
        state = r._STATE
        if state is None or state.backend != 'WHOLE_BODY':
            return {'CANCELLED'}
        name = context.scene.bodymove_whole_bone
        try:
            obj = next((o for o in controls(state).values() if o[BONE] == name), None)
            if obj is None:
                obj = add_control(state, 'bone:'+name, name)
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            r.api().select_only(context, obj)
            state.signature = None
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


CLASSES = (BODYMOVE_OT_WholeMode, BODYMOVE_OT_WholeAddBone)


def draw(box, state):
    from . import reactive as r
    for pair in (('left', 'right'), ('left_hand', 'right_hand'), ('left_elbow', 'right_elbow'),
                 ('left_shoulder', 'right_shoulder'), ('left_knee', 'right_knee'),
                 ('pelvis', 'torso'), ('neck', 'head')):
        row = box.row(align=True)
        for key in pair:
            cell = row.row()
            cell.enabled = key in controls(state)
            label = key.replace('_', ' ').title() if key not in ('left', 'right') else key.title()+' Foot'
            cell.operator('bodymove.reactive_select', text=label).side = key
    obj = bpy.context.view_layer.objects.active
    key = next((k for k, o in controls(state).items() if o == obj), None)
    if key is not None:
        box.label(text=key.replace('_', ' ').replace('bone:', '').title())
        row = box.row(align=True)
        for mode in ('FOLLOW', 'TARGET', 'PIN'):
            op = row.operator('bodymove.whole_mode', text=mode.title(), depress=obj[MODE] == mode)
            op.key, op.mode = key, mode
        if obj[MODE] == 'TARGET':
            box.prop(obj, '["'+PRIORITY+'"]', text='Target Priority')
            box.prop(obj, '["'+ORIENTATION+'"]', text='Match Rotation')
        if obj[MODE] == 'PIN':
            box.label(text='Choose Target to move this pinned part')
        box.prop(state.rig.pose.bones[obj[BONE]], '["'+MOBILITY+'"]', text='Joint Freedom', slider=True)
    box.prop(state.scene, 'bodymove_whole_freedom', text='Body Freedom', slider=True)
    box.prop(state.scene, 'bodymove_balance_response', text='Balance Assistance', slider=True)
    row = box.row(align=True)
    row.prop_search(state.scene, 'bodymove_whole_bone', state.rig.data, 'bones', text='Bone')
    row.operator('bodymove.whole_add_bone', text='', icon='ADD')
    box.label(text='G: move / R: rotate | Follow points adapt')
    row = box.row(align=True)
    row.operator('bodymove.reactive_finish', text='Keep Pose', icon='CHECKMARK').keep = True
    row.operator('bodymove.reactive_finish', text='Cancel', icon='LOOP_BACK').keep = False
    box.label(text=state.scene.bodymove_reactive_status)
