"""Direct viewport targets. Only edited targets constrain the pose."""
import math
import bpy
from . import contacts, influence
from mathutils import Vector, Matrix

INFLUENCE = 'bodymove_influence'


def initialize(state):
    from . import reactive as r
    contacts.initialize(state)
    state.body_controls = {}
    state.body_initial = {}
    state.arms = {}
    state.driven = set()
    state.response_strength = state.scene.bodymove_reactive_strength
    state.torso = state.rig.pose.bones.get(state.scene.bodymove_map_chest)
    mapping = r.api().generic_human_bone_map(state.rig)
    for side in ('left', 'right'):
        bones = [state.rig.pose.bones.get(getattr(state.scene, 'bodymove_map_'+side+'_'+part, '') or mapping.get((side+'_'+part).upper(), ''))
                 for part in ('upper_arm', 'forearm', 'hand')]
        if any(b is None for b in bones):
            continue
        upper, lower, hand = bones
        if lower.parent != upper or hand.parent != lower:
            continue
        a,b,c = [r.world_head(state.rig, bone) for bone in bones]
        if min((b-a).length, (c-b).length) < 1e-5:
            continue
        axis = (c-a).normalized()
        bend = (b-a)-axis*(b-a).dot(axis)
        if bend.length < 1e-5:
            bend = state.rig.matrix_world.to_quaternion() @ Vector((0,-1,0))
        state.arms[side] = dict(bones=bones, lengths=((b-a).length,(c-b).length),
                                pole=b+ bend.normalized()*state.height*0.25)


def create(state):
    from . import reactive as r
    def add(key, matrix, shape):
        obj=bpy.data.objects.new('BM '+key.replace('_',' ').title(),None)
        state.collection.objects.link(obj)
        obj[r.TAG]=True
        obj.matrix_world=matrix
        obj.empty_display_type=shape
        obj.empty_display_size=state.height*0.055
        obj.show_in_front=True
        obj.show_name=True
        obj.hide_render=True
        obj.lock_scale=(True,True,True)
        obj[INFLUENCE]=1.0
        obj['bodymove_target_priority']=1.0
        obj.id_properties_ui('bodymove_target_priority').update(min=0.1,max=10.0,description='Soft hand target importance in whole-body search; fixed contacts retain priority')
        obj.id_properties_ui(INFLUENCE).update(min=0.0,max=1.0,description='Influence of this moved target on body compensation')
        state.body_controls[key]=obj
        state.body_initial[key]=matrix.copy()
    add('pelvis',state.rig.matrix_world @ state.pelvis.matrix,'CIRCLE')
    if state.torso:
        add('torso',state.rig.matrix_world @ state.torso.matrix,'CIRCLE')
    for side,arm in state.arms.items():
        add(side+'_hand',state.rig.matrix_world @ arm['bones'][2].matrix,'SPHERE')
        add(side+'_elbow',Matrix.Translation(arm['pole']),'CONE')
    for obj in state.controls.values():
        obj[INFLUENCE]=1.0
        obj.id_properties_ui(INFLUENCE).update(min=0.0,max=1.0)
        obj.lock_scale=(True,True,True)


def signature(state):
    return tuple(round(v,6) for obj in state.body_controls.values() for row in obj.matrix_world for v in row) + tuple(
        float(obj.get(INFLUENCE,1)) for obj in list(state.controls.values())+list(state.body_controls.values())) + contacts.signature(state) + tuple(influence.target_priority(o) for o in state.body_controls.values())


def moved(state,key):
    from . import reactive as r
    return r.matrix_difference(state.body_controls[key].matrix_world,state.body_initial[key])>1e-5


def prepare(state):
    from . import reactive as r
    state.driven=set()
    state.moved_body={key for key in state.body_controls if moved(state,key)}
    state.moved_body = {k for k in state.moved_body if not k.endswith('_hand') or state.body_controls[k].get(contacts.MODE,'AUTO') not in ('FREE','FIXED')}
    sources=[state.body_controls[k] for k in state.moved_body]
    sources += [obj for side,obj in state.controls.items()
                if r.matrix_difference(obj.matrix_world,state.legs[side]['initial'])>1e-5]
    # Multiple edited targets share a bounded, displacement-weighted gain.
    weights=[]
    for obj in sources:
        key=next((k for k,o in state.body_controls.items() if o==obj),None)
        initial=state.body_initial[key] if key else state.legs[next(k for k,o in state.controls.items() if o==obj)]['initial']
        distance=(obj.matrix_world.translation-initial.translation).length/state.height
        angle=initial.to_quaternion().rotation_difference(obj.matrix_world.to_quaternion()).angle
        weights.append(max(distance+angle,1e-6))
    gain=sum(w*influence.response(o.get(INFLUENCE,1)) for w,o in zip(weights,sources))/sum(weights) if weights else 1
    state.response_strength=state.scene.bodymove_reactive_strength*gain
    if 'pelvis' in state.moved_body:
        target=state.body_controls['pelvis'].matrix_world
        state.pelvis.matrix=state.rig.matrix_world.inverted() @ target
        state.driven.add(state.pelvis.name)
        r.update()
    if 'torso' in state.moved_body:
        bone=state.torso
        target=state.body_controls['torso'].matrix_world
        spine=state.rig.pose.bones.get(state.scene.bodymove_map_spine)
        if spine and spine!=bone:
            pivot=r.world_head(state.rig,spine)
            r.aim(state.rig,spine,r.world_head(state.rig,bone)-pivot,target.translation-pivot)
            state.driven.add(spine.name)
        # Preserve the torso joint position: G bends the spine; R orients chest.
        matrix=target.copy()
        matrix.translation=r.world_head(state.rig,bone)
        bone.matrix=state.rig.matrix_world.inverted() @ matrix
        state.driven.add(bone.name)
        r.update()
    for side,arm in state.arms.items():
        if contacts.hand_active(state,side):
            state.driven.update(b.name for b in arm['bones'])
    state.direct_driven=set(state.driven)
    from . import reach
    reach.assist(state)
    solve_arms(state)


def solve_arms(state):
    from . import reactive as r
    errors={}
    for side,arm in state.arms.items():
        if not contacts.hand_active(state,side):
            continue
        upper,lower,hand=arm['bones']
        target=contacts.target(state,side)
        a=r.world_head(state.rig,upper)
        delta=target.translation-a
        direction=delta.normalized() if delta.length>1e-8 else Vector((1,0,0))
        l1,l2=arm['lengths']
        distance=max(abs(l1-l2)+1e-7,min(delta.length,(l1+l2)*0.999999))
        along=(l1*l1-l2*l2+distance*distance)/(2*distance)
        pole=state.body_controls[side+'_elbow'].matrix_world.translation-a
        bend=pole-direction*pole.dot(direction)
        if bend.length<1e-7:
            bend=direction.cross(Vector((0,0,1)))
            if bend.length<1e-7: bend=direction.cross(Vector((0,1,0)))
        elbow=a+direction*along+bend.normalized()*math.sqrt(max(0,l1*l1-along*along))
        r.aim(state.rig,upper,r.world_head(state.rig,lower)-a,elbow-a)
        b=r.world_head(state.rig,lower)
        r.aim(state.rig,lower,r.world_head(state.rig,hand)-b,a+direction*distance-b)
        matrix=target.copy()
        matrix.translation=r.world_head(state.rig,hand)
        hand.matrix=state.rig.matrix_world.inverted() @ matrix
        r.update()
        errors[side]=(r.world_head(state.rig,hand)-target.translation).length
    return errors
