"""Bounded joint search for hand targets, COM balance and planted contacts.

This is a quasi-static posing solver, not a time-based rigid-body simulation.
Each candidate is evaluated after arm/leg IK; infeasible contacts are rejected.
"""
import math
from . import influence
from mathutils import Vector


def solve(state, active, points):
    from . import reactive as r, body_controls as body, contacts
    hands=[side for side in state.arms if contacts.hand_active(state,side)]
    gain=state.response_strength
    if not hands or gain<=0 or not state.scene.bodymove_coupled_pose:
        return
    authored=getattr(state,'direct_driven',state.driven)
    h=state.height
    origin=r.world_head(state.rig,state.pelvis).copy()

    def evaluate():
        hand_errors=body.solve_arms(state)
        feet={side:state.solve_leg(side) for side in state.legs}
        hard=max([feet[s] for s in active]+[hand_errors.get(s,0) for s in state.hand_pins]+[0])
        balance=state.target_error(active,points)
        cost=2*state.scene.bodymove_balance_response*(balance.length/h)**2
        for side in hands:
            weight=influence.target_priority(state.body_controls[side+'_hand'])
            cost+=4*weight*(hand_errors.get(side,0)/h)**2
        cost+=0.015*((r.world_head(state.rig,state.pelvis)-origin).length/h)**2
        return cost,hard,balance

    cost,hard,balance=evaluate()
    state.coupled_before=cost
    limit_hard=max(h*0.0005,hard+1e-6)
    evaluations=1
    for iteration in range(3):
        baseline=r.pose_basis(state.rig)
        best=baseline
        best_cost=cost
        pull=Vector()
        for side in hands:
            target=contacts.target(state,side).translation
            hand=r.world_head(state.rig,state.arms[side]['bones'][2])
            weight=influence.target_priority(state.body_controls[side+'_hand'])
            pull+=(target-hand)*weight
        proposals=[]
        if state.pelvis.name not in state.protected|authored and state.scene.bodymove_reactive_pelvis>0:
            for delta in (pull,balance*state.scene.bodymove_balance_response,Vector((0,0,-h*0.01))):
                if delta.length>1e-7:
                    step=delta.normalized()*min(delta.length,h*0.025)*gain*state.scene.bodymove_reactive_pelvis
                    proposals.append(('move',None,step,0))
        used=set()
        for field in ('chest','spine'):
            bone=state.rig.pose.bones.get(getattr(state.scene,'bodymove_map_'+field))
            if not bone or bone.name in used or bone.name in state.protected|authored:
                continue
            used.add(bone.name)
            axis=Vector()
            pivot=r.world_head(state.rig,bone)
            for side in hands:
                arm=state.arms[side]['bones'][0]
                if bone not in arm.parent_recursive:continue
                endpoint=r.world_head(state.rig,state.arms[side]['bones'][2])
                delta=contacts.target(state,side).translation-endpoint
                axis+=(endpoint-pivot).cross(delta)
            if axis.length>1e-8 and state.scene.bodymove_reach_torso>0:
                proposals.append(('rotate',bone,axis,math.radians(3)*gain*state.scene.bodymove_reach_torso))
        for kind,bone,vector,angle in proposals:
            r.restore_basis(state.rig,baseline)
            if kind=='move':
                # Reach projection respects planted legs without overriding an authored pelvis.
                state.translate_pelvis(vector,active)
            else:
                state.rotate_world(bone,vector,angle)
                total=state.base[bone.name].to_quaternion().rotation_difference(bone.matrix_basis.to_quaternion()).angle
                if total>math.radians(20):continue
            candidate,contact,_=evaluate(); evaluations+=1
            if contact<=limit_hard and candidate<best_cost-1e-10:
                best_cost=candidate
                best=r.pose_basis(state.rig)
        r.restore_basis(state.rig,best)
        cost,_,balance=evaluate(); evaluations+=1
        if best is baseline:
            break
    state.coupled_after=cost
    state.coupled_evaluations=evaluations
