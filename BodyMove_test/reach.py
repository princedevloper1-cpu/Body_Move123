"""Bounded shoulder/torso reach assistance, evaluated once per authored solve."""
import math
from . import influence
from mathutils import Vector


def assist(state):
    from . import reactive as r
    active = []
    for side, arm in state.arms.items():
        key = side+'_hand'
        if key not in state.moved_body:
            continue
        obj = state.body_controls[key]
        gain = influence.response(obj.get('bodymove_influence',1))
        gain *= state.scene.bodymove_reactive_strength
        if gain > 0:
            active.append((arm, obj, gain))
    if not active:
        return

    def cost():
        total = 0.0
        for arm, obj, gain in active:
            distance = (obj.matrix_world.translation-r.world_head(state.rig, arm['bones'][0])).length
            # Leave a small elbow bend; no assistance inside the comfortable reach.
            gap = max(0, distance-sum(arm['lengths'])*0.98)
            total += gain*gap*gap
        return total

    # Smooth activation near full extension instead of a sudden fixed angle.
    activation = 0.0
    for arm, obj, gain in active:
        length = sum(arm['lengths'])
        distance = (obj.matrix_world.translation-r.world_head(state.rig,arm['bones'][0])).length
        x = max(0, min(1, (distance-length*0.98)/(length*0.15)))
        activation = max(activation, x*x*(3-2*x))
    candidates = []
    seen = set()
    for arm, obj, gain in active:
        upper = arm['bones'][0]
        parent = upper.parent
        # Do not mistake an arbitrary parent/spine bone for a clavicle.
        if parent and any(t in parent.name.lower() for t in ('shoulder','clavicle','clav')):
            if parent.name not in seen:
                candidates.append((parent, math.radians(20)*gain*state.scene.bodymove_reach_shoulder))
                seen.add(parent.name)
    for field in ('chest', 'spine'):
        bone = state.rig.pose.bones.get(getattr(state.scene,'bodymove_map_'+field))
        if bone and bone.name not in seen:
            gain = max(item[2] for item in active)
            candidates.append((bone, math.radians(8)*gain*state.scene.bodymove_reach_torso))
            seen.add(bone.name)
    for bone, limit in candidates:
        if bone.name in state.protected or bone.name in state.driven or limit <= 0:
            continue
        limit *= activation
        used = 0.0
        for _ in range(4):
            old = cost()
            if old < (state.height*1e-5)**2:
                break
            pivot = r.world_head(state.rig, bone)
            axis = Vector()
            for arm, obj, gain in active:
                if bone not in arm['bones'][0].parent_recursive:
                    continue
                shoulder = r.world_head(state.rig, arm['bones'][0])
                delta = obj.matrix_world.translation-shoulder
                gap = max(0,delta.length-sum(arm['lengths'])*0.98)
                if delta.length > 1e-8:
                    axis += (shoulder-pivot).cross(delta.normalized())*gap*gain
            if axis.length < 1e-10:
                break
            before = bone.matrix_basis.copy()
            step = min(limit-used, limit/4)
            accepted = False
            for fraction in (1, 0.5, 0.25):
                bone.matrix_basis = before
                r.update()
                state.rotate_world(bone,axis,step*fraction)
                if cost() < old-(state.height*1e-7)**2:
                    used += step*fraction
                    accepted = True
                    break
            if not accepted:
                bone.matrix_basis = before
                r.update()
                break
        if used:
            # Keep the accepted reach correction from being undone by balance.
            state.driven.add(bone.name)
