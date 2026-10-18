"""Damped whole-body IK with hard pose pins and a soft quasi-static COM task.

All task attachments use the same kinematics. Candidate pin projection and
acceptance happen in memory; no Blender dependency graph updates are needed.
Correction bounds are numerical safeguards relative to the snapshot, not a
calibrated anatomical model. No stepping, collision, or time dynamics here.
"""
from dataclasses import dataclass
import time
import numpy as np
from .skeleton_model import rotvec, rotvecs


@dataclass
class Target:
    bone: int
    position: np.ndarray
    orientation: object = None
    weight: float = 1.0
    fixed: bool = False
    offset: object = None


def balance_delta(com, support_points, radius):
    if not support_points:
        return np.zeros(3)
    p = np.asarray(com[:2])
    a = np.asarray(support_points[0])[:2]
    nearest = a
    if len(support_points) > 1:
        b = np.asarray(support_points[1])[:2]
        ab = b-a
        nearest = a+ab*np.clip((p-a)@ab/max(ab@ab, 1e-12), 0, 1)
    delta = nearest-p
    length = np.linalg.norm(delta)
    result = np.zeros(3)
    if length > radius:
        result[:2] = delta*(length-radius)/max(length, 1e-12)
    return result


def _inverse_step(j, error, damping=1e-7):
    if not len(error):
        return np.zeros(j.shape[1])
    u, s, vt = np.linalg.svd(j, full_matrices=False)
    inv = np.where(s > 1e-6, s/(s*s+damping), 0)
    return vt.T@(inv*(u.T@error))


def solve(model, targets, support_points, support_radius, *, height,
          rotations=None, translation=None, mobility=None, balance_weight=1.0,
          max_iterations=30, max_correction_radians=2.1):
    started = time.perf_counter()
    h = max(float(height), 1e-4)
    dof = 3+3*model.n
    rotations = model.reference_rotations.copy() if rotations is None else rotations.copy()
    translation = np.zeros(3) if translation is None else translation.copy()
    mobility = np.ones(dof) if mobility is None else np.clip(np.asarray(mobility), 0, 1)
    scale = np.sqrt(mobility)
    scale[:3] *= h
    fixed = [t for t in targets if t.fixed]
    soft = [t for t in targets if not t.fixed and t.weight > 0]
    tolerance = 0.0005
    orientation_tolerance = np.deg2rad(0.5)
    # Only kinematically relevant columns, preserving all free COM contributors.
    relevant = set()
    for t in targets:
        relevant.update(model.ancestors[t.bone])
    for i, _, _ in model.masses:
        relevant.update(model.ancestors[i])
    mask = np.zeros(dof, dtype=bool); mask[:3] = True
    for i in relevant:
        mask[3+3*i:6+3*i] = True
    columns = np.flatnonzero(mask & (scale > 0))
    s = scale[columns]
    ref = model.reference_rotations
    posture_weight = 0.0003/max(float(np.mean(mobility[columns])) if len(columns) else 1.0, 0.05)**2

    def target_system(pose, items):
        rows, errors = [], []
        for t in items:
            w = 1.0 if t.fixed else np.sqrt(max(t.weight, 0))
            rows.append(w*model.jacobian(pose, t.bone, t.offset)/h)
            errors.append(w*(np.asarray(t.position)-model.point(pose, t.bone, t.offset))/h)
            if t.orientation is not None:
                rows.append(w*0.35*model.orientation_jacobian(pose, t.bone))
                errors.append(w*0.35*rotvec(np.asarray(t.orientation)@pose.rotations[t.bone].T))
        return (np.vstack(rows), np.concatenate(errors)) if rows else (np.zeros((0, dof)), np.zeros(0))

    def hard_metrics(pose):
        pos = max((np.linalg.norm(model.point(pose, t.bone, t.offset)-t.position)/h for t in fixed), default=0.0)
        ang = max((np.linalg.norm(rotvec(t.orientation@pose.rotations[t.bone].T)) for t in fixed if t.orientation is not None), default=0.0)
        return pos, ang

    def posture(r, tr):
        return np.concatenate((tr/h, rotvecs(ref.transpose(0,2,1)@r).reshape(-1)))

    def evaluate(r, tr):
        pose = model.fk(r, tr)
        errors=[]
        for t in soft:
            w=np.sqrt(t.weight)
            errors.extend(w*(t.position-model.point(pose,t.bone,t.offset))/h)
            if t.orientation is not None:
                errors.extend(w*0.35*rotvec(t.orientation@pose.rotations[t.bone].T))
        es=np.array(errors)
        delta = balance_delta(model.com(pose), support_points, support_radius)/h
        # COM is independently weighted; infeasibility remains explicitly reported.
        departure=posture(r,tr)
        cost = es@es+8*balance_weight*(delta@delta)+posture_weight*(departure@departure)
        return float(cost), pose, delta

    def bounded_step(r, tr, reduced, factor=1.0):
        delta = np.zeros(dof); delta[columns] = reduced*s*factor
        angle = float(np.max(np.linalg.norm(delta[3:].reshape(-1, 3),axis=1)))
        distance = np.linalg.norm(delta[:3])/h
        ratio = max(1.0, angle/0.12, distance/0.035)
        candidate, ct = model.step(r, tr, delta/ratio)
        if np.max(np.linalg.norm(rotvecs(ref.transpose(0,2,1)@candidate),axis=1)) > max_correction_radians:
            return None
        return candidate, ct

    def project(r, tr):
        for _ in range(5):
            pose = model.fk(r, tr)
            p, a = hard_metrics(pose)
            if p < tolerance*0.15 and a < orientation_tolerance*0.15:
                break
            j, e = target_system(pose, fixed)
            step = _inverse_step(j[:, columns]*s, e)
            trial = bounded_step(r, tr, step)
            if trial is None:
                break
            r, tr = trial
        return r, tr

    cost, pose, _ = evaluate(rotations, translation)
    start_hard = hard_metrics(pose)
    hard_limit = (max(tolerance, start_hard[0]+1e-8), max(orientation_tolerance, start_hard[1]+1e-8))
    iterations = 0
    for iteration in range(max_iterations if len(columns) else 0):
        iterations = iteration+1
        # Stop once the actual geometric tolerances are met; no extra pose drift.
        pose = model.fk(rotations, translation)
        positions=[np.linalg.norm(model.point(pose,t.bone,t.offset)-t.position)/h for t in soft]
        orientations=[np.linalg.norm(rotvec(t.orientation@pose.rotations[t.bone].T)) for t in soft if t.orientation is not None]
        hard_now=hard_metrics(pose)
        if (max(positions,default=0)<0.001 and max(orientations,default=0)<np.deg2rad(0.5)
                and hard_now[0]<tolerance and hard_now[1]<orientation_tolerance
                and (not balance_weight or np.linalg.norm(balance_delta(model.com(pose),support_points,support_radius))/h<0.0005)):
            break
        jh, eh = target_system(pose, fixed)
        a = jh[:, columns]*s
        if len(eh):
            u, singular, vt = np.linalg.svd(a, full_matrices=False)
            keep = singular > 1e-6
            basis = vt[keep]
            null = np.eye(len(columns))-basis.T@basis
            hard_step = _inverse_step(a, eh)
        else:
            null = np.eye(len(columns)); hard_step = np.zeros(len(columns))
        js, es = target_system(pose, soft)
        delta = balance_delta(model.com(pose), support_points, support_radius)
        if support_points and balance_weight > 0:
            weight = np.sqrt(8*balance_weight)
            jc = model.com_jacobian(pose).copy(); jc[2] = 0
            js = np.vstack((js, weight*jc/h))
            es = np.concatenate((es, weight*delta/h))
        soft_j = js[:, columns]*s
        # Local posture regularizer also resolves redundant DOFs consistently.
        reg_scale = s.copy(); reg_scale[columns < 3] /= h
        reg = np.diag(reg_scale)*np.sqrt(posture_weight)
        reg_error = -posture(rotations, translation)[columns]*np.sqrt(posture_weight)
        j = np.vstack((soft_j, reg))
        error = np.concatenate((es, reg_error))-j@hard_step
        projected = j@null
        normal = projected.T@projected+np.eye(len(columns))*0.0005
        reduced = hard_step+null@np.linalg.solve(normal, projected.T@error)
        if np.linalg.norm(reduced) < 1e-7:
            break
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125):
            trial = bounded_step(rotations, translation, reduced, fraction)
            if trial is None:
                continue
            cr, ct = project(*trial)
            candidate, cp, _ = evaluate(cr, ct)
            hard = hard_metrics(cp)
            if hard[0] <= hard_limit[0] and hard[1] <= hard_limit[1] and candidate < cost-1e-10:
                rotations, translation, cost = cr, ct, candidate
                accepted = True
                # A restored initially infeasible pin may never get worse again.
                hard_limit = (max(tolerance, min(hard_limit[0], hard[0]+1e-8)),
                              max(orientation_tolerance, min(hard_limit[1], hard[1]+1e-8)))
                break
        if not accepted:
            break
    pose = model.fk(rotations, translation)
    hard = hard_metrics(pose)
    errors = [float(np.linalg.norm(model.point(pose, t.bone, t.offset)-t.position)) for t in targets]
    orientation_errors = [float(np.linalg.norm(rotvec(t.orientation@pose.rotations[t.bone].T))) if t.orientation is not None else 0.0 for t in targets]
    balance = float(np.linalg.norm(balance_delta(model.com(pose), support_points, support_radius)))
    return dict(rotations=rotations, translation=translation, target_errors=errors,
                orientation_errors=orientation_errors, contact_error=hard[0]*h,
                contact_orientation_error=hard[1], balance_error=balance, cost=cost,
                converged=(max(errors, default=0) <= h*0.005 and hard[0] <= tolerance
                           and hard[1] <= orientation_tolerance and (not balance_weight or balance <= h*0.001)),
                iterations=iterations, elapsed=time.perf_counter()-started)
