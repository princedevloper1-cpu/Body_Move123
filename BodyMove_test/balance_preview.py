"""Read-only animation diagnostics using a flat-ground linear pendulum model."""
import math
import csv
import io
import importlib
import bpy
from mathutils import Vector

TAG = '_bodymove_balance_preview'


def estimate(positions, fps, floor, gravity=9.81):
    if len(positions)<3 or fps<=0 or gravity<=0:
        raise ValueError('At least three frames, positive FPS and gravity are required')
    rows=[]
    for i,p in enumerate(positions):
        if i==0:
            velocity=(-3*p+4*positions[1]-positions[2])*(fps/2)
            acceleration=(positions[2]-2*positions[1]+p)*fps*fps
        elif i==len(positions)-1:
            velocity=(3*p-4*positions[i-1]+positions[i-2])*(fps/2)
            acceleration=(p-2*positions[i-1]+positions[i-2])*fps*fps
        else:
            velocity=(positions[i+1]-positions[i-1])*(fps/2)
            acceleration=(positions[i+1]-2*p+positions[i-1])*fps*fps
        height=p.z-floor
        cp=None
        if height>0.01:
            cp=Vector((p.x,p.y,floor))+Vector((velocity.x,velocity.y,0))*math.sqrt(height/gravity)
        rows.append(dict(com=p,velocity=velocity,acceleration=acceleration,capture=cp))
    return rows


def margin(point, supports):
    if point is None or not supports:
        return None
    # Conservative equal-radius capsule, matching the existing foot-point model.
    radius=min(r for p,r in supports)
    a=supports[0][0].copy(); a.z=point.z
    b=supports[-1][0].copy(); b.z=point.z
    d=b-a
    q=a+d*max(0,min(1,(point-a).dot(d)/max(d.length_squared,1e-12)))
    return radius-(point-q).length


def clear():
    for obj in list(bpy.data.objects):
        if obj.get(TAG):
            data=obj.data
            action=obj.animation_data.action if obj.animation_data else None
            data_action=data.animation_data.action if data and data.animation_data else None
            action_names={a.name for a in (action,data_action) if a is not None}
            bpy.data.objects.remove(obj,do_unlink=True)
            if data and data.users==0 and isinstance(data,bpy.types.Curve): bpy.data.curves.remove(data)
            for name in action_names:
                a=bpy.data.actions.get(name)
                if a is not None and a.users==0: bpy.data.actions.remove(a)
    for report in list(bpy.data.texts):
        if report.get(TAG): bpy.data.texts.remove(report)
    for collection in list(bpy.data.collections):
        if collection.get(TAG) and not collection.objects: bpy.data.collections.remove(collection)


def sample(scene):
    api=importlib.import_module(__package__)
    if scene.bodymove_reactive_enabled:
        raise ValueError('Finish Live Balance with Keep Pose or Cancel before sampling animation')
    if not scene.bodymove_analysis_valid or not scene.bodymove_armature_object:
        raise ValueError('Run Quick Human Setup first')
    start,end=(scene.frame_start,scene.frame_end) if scene.bodymove_autophysics_use_scene_range else (scene.bodymove_autophysics_start,scene.bodymove_autophysics_end)
    if end-start<2 or end-start>2000:
        raise ValueError('Choose an interval of 3 to 2001 frames')
    scale=scene.unit_settings.scale_length
    floor=scene.bodymove_virtual_ground_z
    ground=scene.bodymove_ground_object
    if ground:
        z=[(ground.matrix_world @ Vector(v)).z for v in ground.bound_box]
        if max(z)-min(z)>0.001/scale or ground.animation_data:
            raise ValueError('This preview needs a static flat horizontal ground plane')
        floor=sum(z)/len(z)
    masses=list(api.helpers(scene,api.MASS_POINT))
    total=sum(max(0,float(p.get(api.MASS,0))) for p in masses)
    if total<=0: raise ValueError('No positive mass points')
    feet=[api.support_by_role(scene,s+'_FOOT') for s in ('LEFT','RIGHT')]
    feet=[p for p in feet if p is not None]
    if len(feet)!=2: raise ValueError('Both foot support points are required')
    frame,subframe=scene.frame_current,scene.frame_subframe
    rig=scene.bodymove_armature_object
    pose=api.reactive.pose_basis(rig)
    flags=scene.bodymove_live_analysis,scene.bodymove_physics_enabled,api.reactive._BUSY
    positions=[]; support_frames=[]
    try:
        scene.bodymove_live_analysis=False
        scene.bodymove_physics_enabled=False
        api.reactive._BUSY=True
        for f in range(start,end+1):
            scene.frame_set(f)
            graph=bpy.context.evaluated_depsgraph_get()
            p=sum((o.evaluated_get(graph).matrix_world.translation*max(0,float(o.get(api.MASS,0))) for o in masses),Vector())/total
            positions.append(p*scale)
            supports=[]
            for foot in feet:
                pos=foot.evaluated_get(graph).matrix_world.translation.copy()
                if abs(pos.z-floor)<=scene.bodymove_contact_tolerance:
                    pos.z=floor
                    supports.append((pos*scale,float(foot.get(api.RADIUS,0.1))*scale*0.3))
            support_frames.append(supports)
    finally:
        scene.frame_set(frame,subframe=subframe)
        api.reactive.restore_basis(rig,pose)
        scene.bodymove_live_analysis,scene.bodymove_physics_enabled,api.reactive._BUSY=flags
    rows=estimate(positions,scene.render.fps/scene.render.fps_base,floor*scale,scene.bodymove_balance_gravity)
    for f,row,supports in zip(range(start,end+1),rows,support_frames):
        row.update(frame=f,supports=supports,margin=margin(row['capture'],supports))
    return rows,scale,floor


def build(scene,rows,scale,floor):
    clear()
    collection=bpy.data.collections.new('BodyMove Dynamic Diagnostics')
    collection[TAG]=True; scene.collection.children.link(collection)
    def obj(name,shape,color):
        o=bpy.data.objects.new(name,None); collection.objects.link(o); o[TAG]=True
        o.empty_display_type=shape; o.empty_display_size=0.05/scale
        o.show_in_front=True; o.show_name=True; o.color=color
        o.hide_select=True; o.hide_render=True
        return o
    com=obj('DIAG COM','SPHERE',(0,1,0,1))
    cp=obj('DIAG Capture estimate','CIRCLE',(1,0.4,0,1))
    velocity=obj('DIAG COM velocity - 0.25 s','SINGLE_ARROW',(0,0.6,1,1))
    velocity.empty_display_size=1
    curve=bpy.data.curves.new('DIAG Support capsule','CURVE'); curve.dimensions='3D'
    spline=curve.splines.new('POLY'); spline.points.add(31); spline.use_cyclic_u=True
    support=bpy.data.objects.new('DIAG Support approximation',curve)
    collection.objects.link(support); support[TAG]=True; support.hide_select=True; support.hide_render=True
    for row in rows:
        f=row['frame']; com.location=row['com']/scale; com.keyframe_insert('location',frame=f)
        cp.location=(row['capture'] if row['capture'] is not None else row['com'])/scale
        cp.hide_viewport=row['capture'] is None or not row['supports']
        cp.keyframe_insert('location',frame=f); cp.keyframe_insert('hide_viewport',frame=f)
        velocity.location=row['com']/scale
        v=row['velocity']*0.25/scale
        velocity.rotation_mode='QUATERNION'
        velocity.rotation_quaternion=Vector((0,0,1)).rotation_difference(v.normalized()) if v.length>1e-8 else Vector((0,0,1)).rotation_difference(Vector((0,0,1)))
        velocity.scale=(1,1,v.length)
        for path in ('location','rotation_quaternion','scale'):velocity.keyframe_insert(path,frame=f)
        supports=row['supports']; radius=min((r for p,r in supports),default=0)
        support.hide_viewport=not supports; support.keyframe_insert('hide_viewport',frame=f)
        for i,point in enumerate(spline.points):
            n=Vector((math.cos(i*math.tau/32),math.sin(i*math.tau/32),0))
            center=max((p for p,r in supports),key=lambda p:p.dot(n)) if supports else Vector((0,0,floor*scale))
            p=(center+n*radius)/scale
            point.co=(*p,1); point.keyframe_insert('co',frame=f)
    # Diagnostics are sampled each frame; avoid Bezier overshoot between samples.
    for owner in (com,cp,velocity,support,curve):
        ad=owner.animation_data
        if ad and ad.action:
            for layer in ad.action.layers:
                for strip in layer.strips:
                    bag=strip.channelbag(ad.action_slot)
                    if bag:
                        for fc in bag.fcurves:
                            for key in fc.keyframe_points:key.interpolation='LINEAR'
    stream=io.StringIO()
    writer=csv.writer(stream)
    writer.writerow(('frame','com_x_m','com_y_m','com_z_m','vx_m_s','vy_m_s','vz_m_s',
                     'ax_m_s2','ay_m_s2','az_m_s2','capture_x_m','capture_y_m','support_margin_m','contact_count','note'))
    for row in rows:
        note='flat-ground constant-height approximation'
        if not row['supports']: note='no foot support; do not interpret as recovery point'
        elif row['capture'] is None: note='invalid COM height'
        elif abs(row['velocity'].z)>0.25 or abs(row['acceleration'].z)>scene.bodymove_balance_gravity*0.2:
            note='vertical motion; constant-height assumption questionable'
        capture=tuple(row['capture'][:2]) if row['capture'] is not None else ('','')
        writer.writerow((row['frame'],*row['com'],*row['velocity'],*row['acceleration'],*capture,
                         row['margin'] if row['margin'] is not None else '',len(row['supports']),note))
    report=bpy.data.texts.new('BodyMove Dynamic Balance.csv')
    report[TAG]=True; report.write(stream.getvalue())
    outside=sum(r['margin'] is not None and r['margin']<0 for r in rows)
    airborne=sum(not r['supports'] for r in rows)
    scene.bodymove_balance_status=f"{len(rows)} frames: {outside} capture estimates outside support; {airborne} without foot contact"
    api=importlib.import_module(__package__)
    rig=scene.bodymove_armature_object
    pose=api.reactive.pose_basis(rig)
    busy=api.reactive._BUSY
    try:
        api.reactive._BUSY=True
        scene.frame_set(scene.frame_current,subframe=scene.frame_subframe)
        api.reactive.restore_basis(rig,pose)
    finally:
        api.reactive._BUSY=busy


class Preview(bpy.types.Operator):
    bl_idname='bodymove.dynamic_balance_preview'
    bl_label='Preview Dynamic Balance'
    def execute(self,context):
        try:
            rows,scale,floor=sample(context.scene)
            build(context.scene,rows,scale,floor)
        except Exception as exc:
            self.report({'ERROR'},str(exc)); return {'CANCELLED'}
        return {'FINISHED'}


class Clear(bpy.types.Operator):
    bl_idname='bodymove.dynamic_balance_clear'
    bl_label='Clear Balance Preview'
    def execute(self,context):
        clear(); context.scene.bodymove_balance_status='Preview cleared'
        return {'FINISHED'}


def draw(layout,scene):
    box=layout.box(); box.label(text='Dynamic Balance Diagnostics (flat ground)')
    box.label(text='Uses the AutoPhysics frame interval')
    box.prop(scene,'bodymove_balance_gravity',text='Gravity (m/s²)')
    box.operator(Preview.bl_idname); box.operator(Clear.bl_idname)
    box.label(text='Estimate only - no automatic recovery step')
    box.label(text='CSV report in Blender Text Editor; rebuild after edits')
    box.label(text=scene.bodymove_balance_status)


def register():
    bpy.types.Scene.bodymove_balance_gravity=bpy.props.FloatProperty(default=9.81,min=0.01)
    bpy.types.Scene.bodymove_balance_status=bpy.props.StringProperty(default='Sample an animation of at least 3 frames')
    for cls in (Preview,Clear):bpy.utils.register_class(cls)


def unregister():
    for cls in (Clear,Preview):bpy.utils.unregister_class(cls)
    del bpy.types.Scene.bodymove_balance_gravity
    del bpy.types.Scene.bodymove_balance_status
