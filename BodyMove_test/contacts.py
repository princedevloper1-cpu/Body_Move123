"""Explicit hand contact states and contact-first pose acceptance."""
import bpy

MODE = 'bodymove_contact_mode'


def initialize(state):
    state.hand_pins = {}
    state.contact_limited = False


def hand_active(state, side):
    key = side+'_hand'
    mode = state.body_controls[key].get(MODE, 'AUTO')
    if mode == 'FREE':
        return False
    return mode in ('FIXED', 'TARGET') or key in state.moved_body or side+'_elbow' in state.moved_body


def target(state, side):
    return state.hand_pins.get(side, state.body_controls[side+'_hand'].matrix_world)


def set_mode(state, side, mode):
    from . import reactive as r
    obj = state.body_controls[side+'_hand']
    actual = state.rig.matrix_world @ state.arms[side]['bones'][2].matrix
    obj.matrix_world = actual
    obj[MODE] = mode
    if mode == 'FREE':
        for bone in state.arms[side]['bones']:
            state.base[bone.name] = bone.matrix_basis.copy()
    if mode == 'FIXED':
        state.hand_pins[side] = actual.copy()
    else:
        state.hand_pins.pop(side, None)
    obj.lock_location = (mode == 'FIXED',)*3
    obj.lock_rotation = (mode == 'FIXED',)*3
    obj.lock_rotations_4d = True
    obj.lock_rotation_w = mode == 'FIXED'
    r.update()


def signature(state):
    return tuple(state.body_controls[s+'_hand'].get(MODE, 'AUTO') for s in state.arms)


def resolve(state):
    from . import body_controls
    hands = body_controls.solve_arms(state)
    feet = {side:state.solve_leg(side) for side in state.legs}
    # Foot targets and fixed hands outrank balance and direct torso requests.
    errors = list(feet.values())+[hands.get(s,0) for s in state.hand_pins]
    return max(errors,default=0)


def accept(state, before):
    """Backtrack a pose proposal if it worsens hard contacts; keep best feasible step."""
    from . import reactive as r
    if not state.hand_pins:
        return
    proposed = r.pose_basis(state.rig)
    proposed_error = resolve(state)
    tolerance = state.height*0.0005
    if proposed_error <= tolerance:
        return
    r.restore_basis(state.rig,before)
    baseline_error = resolve(state)
    best = r.pose_basis(state.rig)
    best_error = baseline_error
    for alpha in (0.5,0.25,0.125,0.0625):
        blended = {name:before[name].lerp(matrix,alpha) for name,matrix in proposed.items()}
        r.restore_basis(state.rig,blended)
        error = resolve(state)
        if error <= max(tolerance,baseline_error+1e-6):
            state.contact_limited = True
            return
        if error < best_error:
            best_error = error
            best = r.pose_basis(state.rig)
    if proposed_error < best_error:
        best = proposed
    r.restore_basis(state.rig,best)
    resolve(state)
    state.contact_limited = True


class BODYMOVE_OT_HandContact(bpy.types.Operator):
    bl_idname = 'bodymove.hand_contact'
    bl_label = 'Set hand contact'
    side: bpy.props.EnumProperty(items=(('left','Left',''),('right','Right','')))
    mode: bpy.props.EnumProperty(items=(('FREE','Free','Follow the body'),('TARGET','Target','Follow the movable target'),('FIXED','Fixed','Keep the current hand transform in world space')))

    def execute(self,context):
        from . import reactive as r
        state = r._STATE
        if state is None or state.backend == 'WHOLE_BODY' or self.side not in state.arms:
            return {'CANCELLED'}
        r._BUSY = True
        try:
            set_mode(state,self.side,self.mode)
            state.solve(force=True)
        finally:
            r._BUSY = False
        return {'FINISHED'}


def draw(box,state):
    if state is None:
        return
    for side in state.arms:
        row = box.row(align=True)
        row.label(text=side.title()+' Hand')
        current = state.body_controls[side+'_hand'].get(MODE,'AUTO')
        if current == 'AUTO':
            row.label(text='Auto')
        for mode,label in (('FREE','Free'),('TARGET','Target'),('FIXED','Fixed')):
            op = row.operator('bodymove.hand_contact',text=label,depress=current==mode)
            op.side,op.mode = side,mode
    box.label(text='Fixed hands and feet take priority over balance')
