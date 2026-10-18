"""Rigid forward kinematics in world units, with analytic point/COM Jacobians.

The Blender adapter snapshots regular FK rigs with positive uniform scale.
Local translations and scales are frozen; only rotations and pelvis translation
are solved. No bpy calls occur while evaluating candidate poses.
"""
from dataclasses import dataclass
import numpy as np


def exp(vector):
    v = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(v)
    k = np.array(((0, -v[2], v[1]), (v[2], 0, -v[0]), (-v[1], v[0], 0)))
    if angle < 1e-8:
        return np.eye(3)+k+0.5*(k@k)
    return np.eye(3)+(np.sin(angle)/angle)*k+((1-np.cos(angle))/angle**2)*(k@k)


def rotvec(matrix):
    m = np.asarray(matrix, dtype=float)
    angle = np.arccos(np.clip((np.trace(m)-1)*0.5, -1, 1))
    vee = np.array((m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1]))
    if angle < 1e-7:
        return vee*0.5
    if np.pi-angle < 1e-5:
        values, vectors = np.linalg.eigh((m+np.eye(3))*0.5)
        axis = vectors[:, np.argmax(values)]
        if axis@vee < 0:
            axis = -axis
        return axis*angle
    return vee*(angle/(2*np.sin(angle)))


def rotvecs(m):
    angles = np.arccos(np.clip((np.trace(m, axis1=-2, axis2=-1)-1)*0.5, -1, 1))
    vee = np.stack((m[:,2,1]-m[:,1,2], m[:,0,2]-m[:,2,0], m[:,1,0]-m[:,0,1]), axis=1)
    factor = np.full(len(m), 0.5)
    regular = (angles > 1e-7) & (angles < np.pi-1e-5)
    factor[regular] = angles[regular]/(2*np.sin(angles[regular]))
    result = vee*factor[:,None]
    for i in np.flatnonzero(angles >= np.pi-1e-5):
        result[i] = rotvec(m[i])
    return result


def exps(v):
    angles = np.linalg.norm(v, axis=1)
    k = np.zeros((len(v),3,3))
    k[:,0,1]=-v[:,2]; k[:,0,2]=v[:,1]
    k[:,1,0]=v[:,2]; k[:,1,2]=-v[:,0]
    k[:,2,0]=-v[:,1]; k[:,2,1]=v[:,0]
    a=np.ones(len(v)); b=np.full(len(v),0.5)
    mask=angles>1e-8
    a[mask]=np.sin(angles[mask])/angles[mask]
    b[mask]=(1-np.cos(angles[mask]))/angles[mask]**2
    return np.eye(3)[None,:,:]+a[:,None,None]*k+b[:,None,None]*(k@k)


rotation_exp, rotation_log = exp, rotvec


@dataclass
class Pose:
    positions: np.ndarray
    rotations: np.ndarray


class SkeletonModel:
    def __init__(self, names, parents, offsets, reference_rotations, root_index, masses,
                 scales=None):
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(names)}
        self.n = len(names)
        self.parents = np.asarray(parents, dtype=int)
        self.offsets = np.asarray(offsets, dtype=float)
        self.reference_rotations = np.asarray(reference_rotations, dtype=float)
        self.root_index = root_index
        self.scales = np.ones(self.n) if scales is None else np.asarray(scales, dtype=float)
        self.masses = [(int(i), float(m), np.asarray(p, dtype=float)) for i, m, p in masses if m > 0]
        self.total_mass = sum(m for _, m, _ in self.masses)
        if self.total_mass <= 0:
            raise ValueError('Whole Body needs positive mass points attached to bones')
        self.ancestors = []
        for i, parent in enumerate(self.parents):
            if parent >= i or parent < -1:
                raise ValueError('Skeleton must be in parent-first order')
            self.ancestors.append(([i]+self.ancestors[parent]) if parent >= 0 else [i])
        self.root_descendants = np.array([root_index in a for a in self.ancestors])
        depths=[]
        for i,parent in enumerate(self.parents):
            depths.append(0 if parent<0 else depths[parent]+1)
        self.levels=[np.flatnonzero(np.array(depths)==d) for d in range(max(depths)+1)]
        self.mass_indices = np.array([i for i,_,_ in self.masses])
        self.mass_values = np.array([m for _,m,_ in self.masses])
        self.mass_offsets = np.array([p*self.scales[i] for i,_,p in self.masses])
        self.mass_affected = np.array([[a in self.ancestors[i] for i in self.mass_indices] for a in range(self.n)])
        self.subtree_mass = self.mass_affected@self.mass_values

    @classmethod
    def from_blender(cls, rig, pelvis_name, mass_offsets):
        import bpy
        bpy.context.view_layer.update()
        bones = sorted(rig.pose.bones, key=lambda b: len(b.parent_recursive))
        index = {b.name: i for i, b in enumerate(bones)}
        if pelvis_name not in index:
            raise ValueError('Map the pelvis before enabling Whole Body')
        parents, positions, rotations, scales = [], [], [], []
        def rigid(matrix, label):
            a = np.array(matrix, dtype=float)[:3, :3]
            lengths = np.linalg.norm(a, axis=0)
            if min(lengths) < 1e-8 or np.linalg.det(a) <= 0:
                raise ValueError('Whole Body requires positive scale: '+label)
            scale = float(lengths.mean())
            r = a/scale
            if np.max(np.abs(r.T@r-np.eye(3))) > 3e-4:
                raise ValueError('Whole Body needs uniform scale without shear: '+label)
            return r, scale
        rigid(rig.matrix_world, rig.name)
        for b in bones:
            if not b.bone.use_inherit_rotation or b.bone.inherit_scale != 'FULL':
                raise ValueError('Whole Body currently needs regular FK inheritance: '+b.name)
            matrix = rig.matrix_world @ b.matrix
            rot, scale = rigid(matrix, b.name)
            parents.append(index[b.parent.name] if b.parent else -1)
            positions.append(np.array(matrix.translation, dtype=float))
            rotations.append(rot)
            scales.append(scale)
        local, offsets = [], []
        for i, parent in enumerate(parents):
            if parent < 0:
                local.append(rotations[i]); offsets.append(positions[i])
            else:
                local.append(rotations[parent].T@rotations[i])
                offsets.append(rotations[parent].T@(positions[i]-positions[parent]))
        masses = [(index[item['bone']], item['mass'], item['local']) for item in mass_offsets
                  if item['bone'] in index and item['mass'] > 0]
        model = cls([b.name for b in bones], parents, offsets, local, index[pelvis_name], masses, scales)
        pose = model.fk(*model.zero_pose())
        if np.max(np.abs(pose.positions-np.array(positions))) > 1e-5:
            raise ValueError('Whole Body could not reproduce the FK snapshot')
        return model

    def zero_pose(self):
        return self.reference_rotations.copy(), np.zeros(3)

    def fk(self, rotations, translation):
        p = np.empty((self.n, 3)); r = np.empty((self.n, 3, 3))
        for level in self.levels:
            parents=self.parents[level]
            if parents[0]<0:
                p[level]=self.offsets[level]; r[level]=rotations[level]
            else:
                p[level]=p[parents]+np.einsum('nij,nj->ni',r[parents],self.offsets[level])
                r[level]=r[parents]@rotations[level]
            if self.root_index in level:
                p[self.root_index]+=translation
        return Pose(p, r)

    def point(self, pose, index, local_offset=None):
        if local_offset is None:
            return pose.positions[index].copy()
        return pose.positions[index]+pose.rotations[index]@(np.asarray(local_offset)*self.scales[index])

    def jacobian(self, pose, index, local_offset=None):
        j = np.zeros((3, 3+3*self.n))
        if self.root_descendants[index]:
            j[:, :3] = np.eye(3)
        point = self.point(pose, index, local_offset)
        for ancestor in self.ancestors[index]:
            # Pelvis translation is world-space and does not rotate with ancestors
            # above the pelvis. Snapshot assumes those ancestors remain ordinary FK.
            delta = point-pose.positions[ancestor]
            if ancestor != self.root_index and self.root_index in self.ancestors[index] and ancestor in self.ancestors[self.root_index][1:]:
                # Translation added at the pelvis is not affected by these rotations.
                # Recover its world displacement from this FK pose.
                parent = self.parents[self.root_index]
                expected = (pose.positions[parent]+pose.rotations[parent]@self.offsets[self.root_index]) if parent >= 0 else self.offsets[self.root_index]
                delta -= pose.positions[self.root_index]-expected
            x,y,z=delta
            skew = np.array(((0,z,-y),(-z,0,x),(y,-x,0)))
            j[:, 3+3*ancestor:6+3*ancestor] = skew@pose.rotations[ancestor]
        return j

    def orientation_jacobian(self, pose, index):
        j = np.zeros((3, 3+3*self.n))
        for ancestor in self.ancestors[index]:
            j[:, 3+3*ancestor:6+3*ancestor] = pose.rotations[ancestor]
        return j

    def mass_positions(self, pose):
        return pose.positions[self.mass_indices]+np.einsum('nij,nj->ni',pose.rotations[self.mass_indices],self.mass_offsets)

    def com(self, pose):
        return self.mass_values@self.mass_positions(pose)/self.total_mass

    def com_jacobian(self, pose):
        weighted=self.mass_values[:,None]*self.mass_positions(pose)
        lever=self.mass_affected@weighted-self.subtree_mass[:,None]*pose.positions
        root=self.root_index; parent=self.parents[root]
        expected=pose.positions[parent]+pose.rotations[parent]@self.offsets[root] if parent>=0 else self.offsets[root]
        tr=pose.positions[root]-expected
        for ancestor in self.ancestors[root][1:]:
            lever[ancestor]-=tr*self.subtree_mass[root]
        skew=np.zeros((self.n,3,3)); x,y,z=lever.T
        skew[:,0,1]=z; skew[:,0,2]=-y; skew[:,1,0]=-z
        skew[:,1,2]=x; skew[:,2,0]=y; skew[:,2,1]=-x
        j=np.zeros((3,3+3*self.n))
        j[:,:3]=np.eye(3)*self.subtree_mass[root]/self.total_mass
        j[:,3:]=(skew@pose.rotations).transpose(1,0,2).reshape(3,-1)/self.total_mass
        return j

    def step(self, rotations, translation, delta):
        return rotations@exps(delta[3:].reshape(-1,3)), translation+delta[:3]

    def apply_to_blender(self, rig, rotations, translation):
        import bpy
        from mathutils import Matrix
        pose = self.fk(rotations, translation)
        desired = []
        inv = rig.matrix_world.inverted()
        for i, name in enumerate(self.names):
            world = np.eye(4)
            world[:3, :3] = pose.rotations[i]*self.scales[i]
            world[:3, 3] = pose.positions[i]
            desired.append(inv@Matrix(world.tolist()))
        for i, name in enumerate(self.names):
            b = rig.pose.bones[name]
            kwargs = {}
            if self.parents[i] >= 0:
                kwargs = dict(parent_matrix=desired[self.parents[i]], parent_matrix_local=b.parent.bone.matrix_local)
            b.matrix_basis = b.bone.convert_local_to_pose(desired[i], b.bone.matrix_local, invert=True, **kwargs)
        bpy.context.view_layer.update()
