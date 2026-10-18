"""Segment reaction model for baked FK poses (not a rigid-body simulator).

All sampling is in a pelvis-attached reference frame, in Blender length units
and seconds. Uniform world translation/rotation must not excite the model.
"""

import math
from mathutils import Vector


def limited(vector, limit):
    return vector.normalized() * limit if vector.length > limit else vector.copy()


def derivative(values, times):
    if len(values) < 2:
        return [Vector((0, 0, 0)) for _ in values]
    result = []
    for i in range(len(values)):
        a, b = max(0, i - 1), min(len(values) - 1, i + 1)
        dt = times[b] - times[a]
        result.append((values[b] - values[a]) / dt if dt > 0 else Vector((0, 0, 0)))
    return result


def angular_velocity(rotations, times):
    result = []
    for i in range(len(rotations)):
        a, b = max(0, i - 1), min(len(rotations) - 1, i + 1)
        dt = times[b] - times[a]
        delta = rotations[b] @ rotations[a].conjugated()
        delta.normalize()
        if delta.w < 0:
            delta.negate()
        axis, angle = delta.to_axis_angle()
        result.append(axis * (angle / dt) if dt > 0 else Vector((0, 0, 0)))
    return result


def edge_weight(index, count, blend_frames):
    if blend_frames <= 0:
        return 1.0
    t = min(1.0, index / blend_frames, (count - 1 - index) / blend_frames)
    return t * t * (3.0 - 2.0 * t)


def spring_step(position, velocity, target, dt, frequency, damping, limit):
    """Substepped damped spring; units do not depend on frame rate."""
    omega = math.tau * frequency
    steps = max(1, math.ceil(dt * omega * 8))
    h = dt / steps
    for _ in range(steps):
        velocity += ((target - position) * omega**2 - velocity * (2 * damping * omega)) * h
        position += velocity * h
        if position.length > limit:
            position = limited(position, limit)
            outward = velocity.dot(position.normalized())
            if outward > 0:
                velocity -= position.normalized() * outward
    return position, velocity


def solve_segment_reactions(samples, receivers, *, fps, height, strength,
                            preservation, compensation, secondary, frequency,
                            damping, response_time, max_angle, edge_frames):
    """Return per-frame world rotation vectors for explicitly mapped bones.

    A receiver responds to other segments' relative angular momentum, excluding
    its own segment. Child segments still drive a torso receiver. A moving receiver gets less correction, preserving authored
    primary motion. Rod inertia is an approximation, not measured anatomy.
    """
    output = [{} for _ in samples]
    if not samples or strength <= 0 or preservation >= 1:
        return output
    times = [sample['frame'] / fps for sample in samples]
    names = set.intersection(*(set(s['segments']) for s in samples))
    if not names:
        return output
    total_mass = sum(samples[0]['segments'][name]['mass'] for name in names)
    if total_mass <= 0:
        return output
    height = max(height, 0.01)
    velocities, spins = {}, {}
    for name in names:
        velocities[name] = derivative([s['segments'][name]['position'] for s in samples], times)
        spins[name] = angular_velocity([s['segments'][name]['rotation'] for s in samples], times)
    states = {name: (Vector((0, 0, 0)), Vector((0, 0, 0))) for name in receivers}
    gain = strength * (1.0 - preservation)
    for i, sample in enumerate(samples):
        parts = sample['segments']
        center = sum((parts[n]['position'] * parts[n]['mass'] for n in names), Vector()) / total_mass
        center_velocity = sum((velocities[n][i] * parts[n]['mass'] for n in names), Vector()) / total_mass
        momentum = {}
        inertia = Vector((0, 0, 0))
        for name in names:
            part = parts[name]
            r = part['position'] - center
            mass = part['mass']
            rod = mass * part['length']**2 / 12.0
            momentum[name] = r.cross((velocities[name][i] - center_velocity) * mass) + spins[name][i] * rod
            inertia += Vector((mass * (r.y*r.y + r.z*r.z) + rod,
                               mass * (r.x*r.x + r.z*r.z) + rod,
                               mass * (r.x*r.x + r.y*r.y) + rod))
        floor = total_mass * height**2 * 0.005
        dt = (times[i] - times[i-1]) if i else 1.0 / fps
        for bone, receiver in receivers.items():
            other = sum((momentum[n] for n in names if parts[n]['bone'] not in receiver['exclude']), Vector())
            motion = sum((velocities[n][i].length / height + spins[n][i].length
                          for n in names if parts[n]['bone'] == bone), 0.0)
            mobility = receiver['weight'] / (1.0 + response_time * motion)
            target = Vector(tuple(-other[k] / max(inertia[k], floor) for k in range(3)))
            target = limited(target * response_time * mobility, max_angle)
            position, velocity = spring_step(*states[bone], target, dt, frequency, damping, max_angle)
            states[bone] = position, velocity
            # Direct counter-motion plus delayed follow-through relative to it.
            correction = target * compensation + (position - target) * secondary
            correction = limited(correction * gain, max_angle)
            correction *= edge_weight(i, len(samples), edge_frames)
            output[i][bone] = sample['reference_rotation'] @ correction
    return output
