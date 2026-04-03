# ============================================================================
# Armature to Mesh
# Converts an armature's bones into a mesh object.
# Panel lives in: Properties ▸ Object Data (armature selected) OR 3D Viewport ▸ Sidebar (N-panel) ▸ "Armature to Mesh" tab
# ============================================================================

import bpy
import bmesh
import math
from bpy.types  import Operator, Panel, PropertyGroup
from bpy.props  import (EnumProperty, FloatProperty, IntProperty,
                        BoolProperty, PointerProperty)
from mathutils  import Vector, Matrix


# ============================================================================
# Core geometry functions  (no Blender UI dependencies)
# ============================================================================

def bone_axes(head, tail, roll_matrix):
    """Return (y, x, z, length) axes for a bone; None if degenerate."""
    bone_vec = tail - head
    length   = bone_vec.length
    if length < 1e-6:
        return None
    y_axis = bone_vec.normalized()
    x_axis = (roll_matrix @ Vector((1, 0, 0))).normalized()
    z_axis = y_axis.cross(x_axis).normalized()
    x_axis = z_axis.cross(y_axis).normalized()
    return y_axis, x_axis, z_axis, length


# ---- Bone shapes -----------------------------------------------------------

def add_octahedron_to_bmesh(bm, head, tail, roll_matrix, props):
    axes = bone_axes(head, tail, roll_matrix)
    if axes is None:
        return False
    y_axis, x_axis, z_axis, length = axes
    base_center = head + y_axis * (length * props.octa_base_position)
    hw          = length * props.octa_base_width
    positions   = [
        head,
        base_center + x_axis * hw + z_axis * hw,
        base_center - x_axis * hw + z_axis * hw,
        base_center - x_axis * hw - z_axis * hw,
        base_center + x_axis * hw - z_axis * hw,
        tail,
    ]
    bm_verts = [bm.verts.new(p.copy()) for p in positions]
    for fi in [(0,1,2),(0,2,3),(0,3,4),(0,4,1),
               (5,2,1),(5,3,2),(5,4,3),(5,1,4)]:
        try:
            bm.faces.new([bm_verts[i] for i in fi])
        except ValueError:
            pass
    return True


def add_cylinder_to_bmesh(bm, head, tail, roll_matrix, props):
    axes = bone_axes(head, tail, roll_matrix)
    if axes is None:
        return False
    y_axis, x_axis, z_axis, length = axes
    radius = length * props.cyl_radius_factor
    seg    = max(3, props.cyl_segments)
    head_ring, tail_ring = [], []
    for s in range(seg):
        angle  = 2 * math.pi * s / seg
        offset = x_axis * (radius * math.cos(angle)) + z_axis * (radius * math.sin(angle))
        head_ring.append(bm.verts.new((head + offset).copy()))
        tail_ring.append(bm.verts.new((tail + offset).copy()))
    for s in range(seg):
        ns = (s + 1) % seg
        bm.faces.new([head_ring[s], head_ring[ns], tail_ring[ns], tail_ring[s]])
    hc = bm.verts.new(head.copy())
    tc = bm.verts.new(tail.copy())
    for s in range(seg):
        ns = (s + 1) % seg
        bm.faces.new([hc, head_ring[ns], head_ring[s]])
        bm.faces.new([tc, tail_ring[s],  tail_ring[ns]])
    return True


def add_rectangle_to_bmesh(bm, head, tail, roll_matrix, props):
    axes = bone_axes(head, tail, roll_matrix)
    if axes is None:
        return False
    y_axis, x_axis, z_axis, length = axes
    hw = length * props.rect_width_factor * 0.5
    hd = length * props.rect_depth_factor * 0.5

    def ring(center):
        return [
            center + x_axis * hw + z_axis * hd,
            center - x_axis * hw + z_axis * hd,
            center - x_axis * hw - z_axis * hd,
            center + x_axis * hw - z_axis * hd,
        ]

    h_ring = [bm.verts.new(v.copy()) for v in ring(head)]
    t_ring = [bm.verts.new(v.copy()) for v in ring(tail)]
    bm.faces.new([h_ring[0], h_ring[1], h_ring[2], h_ring[3]])
    bm.faces.new([t_ring[0], t_ring[3], t_ring[2], t_ring[1]])
    for s in range(4):
        ns = (s + 1) % 4
        bm.faces.new([h_ring[s], h_ring[ns], t_ring[ns], t_ring[s]])
    return True


SHAPE_BUILDERS = {
    'OCTAHEDRAL': add_octahedron_to_bmesh,
    'CYLINDER':   add_cylinder_to_bmesh,
    'RECTANGLE':  add_rectangle_to_bmesh,
}


# ---- Joint spheres ---------------------------------------------------------

def add_sphere_to_bmesh(bm, center, radius, segments, rings):
    verts = [bm.verts.new(center + Vector((0, 0, radius)))]
    for r in range(1, rings):
        phi = math.pi * r / rings
        z   = radius * math.cos(phi)
        rxy = radius * math.sin(phi)
        for s in range(segments):
            theta = 2 * math.pi * s / segments
            verts.append(bm.verts.new(
                center + Vector((rxy * math.cos(theta), rxy * math.sin(theta), z))
            ))
    verts.append(bm.verts.new(center + Vector((0, 0, -radius))))

    def ri(r, s):
        return 1 + (r - 1) * segments + (s % segments)

    for s in range(segments):
        bm.faces.new([verts[0], verts[ri(1, s)], verts[ri(1, s + 1)]])
    for r in range(1, rings - 1):
        for s in range(segments):
            bm.faces.new([verts[ri(r, s)], verts[ri(r, s+1)],
                          verts[ri(r+1, s+1)], verts[ri(r+1, s)]])
    south = len(verts) - 1
    for s in range(segments):
        bm.faces.new([verts[south], verts[ri(rings-1, s+1)], verts[ri(rings-1, s)]])


def collect_joint_positions(armature_obj, radius_factor, deform_only):
    obj_matrix = armature_obj.matrix_world
    raw = []
    for pbone in armature_obj.pose.bones:
        bone = pbone.bone
        if deform_only and not bone.use_deform:
            continue
        if bone.length < 1e-6:
            continue
        radius = bone.length * radius_factor
        raw.append(((obj_matrix @ bone.head_local).copy(), radius))
        raw.append(((obj_matrix @ bone.tail_local).copy(), radius))
    return raw


# ---- Main builder ----------------------------------------------------------

def bones_to_mesh(armature_obj, props):
    build_bone = SHAPE_BUILDERS[props.bone_shape]
    obj_matrix = armature_obj.matrix_world
    bm         = bmesh.new()

    skipped = 0
    for pbone in armature_obj.pose.bones:
        bone = pbone.bone
        if props.deform_only and not bone.use_deform:
            continue
        head_world = obj_matrix @ bone.head_local
        tail_world = obj_matrix @ bone.tail_local
        roll_mat   = obj_matrix.to_3x3() @ pbone.matrix.to_3x3()
        if not build_bone(bm, head_world, tail_world, roll_mat, props):
            skipped += 1

    if props.add_joint_spheres:
        joints = collect_joint_positions(
            armature_obj,
            props.joint_radius_factor,
            props.deform_only,
        )
        for center, radius in joints:
            add_sphere_to_bmesh(bm, center, radius,
                                props.joint_segments, props.joint_rings)

    bm.normal_update()
    mesh_name = f"{armature_obj.name}_BoneMesh_{props.bone_shape}"
    mesh      = bpy.data.meshes.new(mesh_name)
    bm.to_mesh(mesh)
    bm.free()

    mesh_obj = bpy.data.objects.new(mesh_name, mesh)
    bpy.context.collection.objects.link(mesh_obj)

    if skipped:
        print(f"[Armature to Mesh] Skipped {skipped} degenerate bone(s).")

    return mesh_obj


# ============================================================================
#  Add-on Properties
# ============================================================================

class BonesToMeshProperties(PropertyGroup):

    deform_only: BoolProperty(
        name        = "Deform Bones Only",
        description = "Only convert bones marked as deform bones; non-deform bones such as IK handles and control bones are skipped",
        default     = True,
    )

    bone_shape: EnumProperty(
        name        = "Bone Shape",
        description = "Geometry shape used for each bone",
        items       = [
            ('OCTAHEDRAL', "Octahedral", "Classic Blender double-pyramid bone shape"),
            ('CYLINDER',   "Cylinder",   "Round capped tube along the bone"),
            ('RECTANGLE',  "Rectangle",  "Box / cuboid along the bone"),
        ],
        default = 'OCTAHEDRAL',
    )

    # --- Octahedral ---
    octa_base_position: FloatProperty(
        name        = "Base Position",
        description = "Position of the wide base along the bone (0 = head, 1 = tail)",
        default     = 0.10, min = 0.05, max = 0.95, step = 1,
    )
    octa_base_width: FloatProperty(
        name        = "Base Width",
        description = "Half-width of the octahedron base as a fraction of bone length",
        default     = 0.10, min = 0.01, max = 0.5,  step = 1,
    )

    # --- Cylinder ---
    cyl_radius_factor: FloatProperty(
        name        = "Radius",
        description = "Cylinder radius as a fraction of bone length",
        default     = 0.07, min = 0.01, max = 0.5,  step = 1,
    )
    cyl_segments: IntProperty(
        name        = "Segments",
        description = "Number of sides on the cylinder (higher = rounder)",
        default     = 8, min = 3, max = 64,
    )

    # --- Rectangle ---
    rect_width_factor: FloatProperty(
        name        = "Width",
        description = "Box width as a fraction of bone length",
        default     = 0.12, min = 0.01, max = 0.5,  step = 1,
    )
    rect_depth_factor: FloatProperty(
        name        = "Depth",
        description = "Box depth as a fraction of bone length",
        default     = 0.06, min = 0.01, max = 0.5,  step = 1,
    )

    # --- Joint spheres ---
    add_joint_spheres: BoolProperty(
        name        = "Add Joint Spheres",
        description = "Place a UV sphere at every bone head / tail position",
        default     = True,
    )
    joint_radius_factor: FloatProperty(
        name        = "Joint Radius",
        description = "Sphere radius as a fraction of the bone length",
        default     = 0.05, min = 0.01, max = 0.5,  step = 1,
    )
    joint_segments: IntProperty(
        name        = "Sphere Segments",
        description = "Longitude subdivisions of each joint sphere",
        default     = 8, min = 3, max = 32,
    )
    joint_rings: IntProperty(
        name        = "Sphere Rings",
        description = "Latitude subdivisions of each joint sphere",
        default     = 6, min = 2, max = 32,
    )


# ============================================================================
#  Operator
# ============================================================================

class OBJECT_OT_bones_to_mesh(Operator):
    bl_idname      = "object.bones_to_mesh"
    bl_label       = "Convert to Mesh"
    bl_description = "Convert the active armature's bones to a mesh object"
    bl_options     = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def execute(self, context):
        obj   = context.active_object
        props = context.scene.bones_to_mesh_props

        try:
            result = bones_to_mesh(obj, props)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        bpy.ops.object.select_all(action='DESELECT')
        result.select_set(True)
        context.view_layer.objects.active = result

        self.report({'INFO'}, f"Created '{result.name}'")
        return {'FINISHED'}


# ============================================================================
#  Panels
# ============================================================================

def draw_panel_content(layout, context):
    props = context.scene.bones_to_mesh_props

    # ---- Deform bones filter -----------------------------------------------
    layout.prop(props, "deform_only")

    # ---- Bone shape selector -----------------------------------------------
    box = layout.box()
    box.label(text="Bone Shape", icon='BONE_DATA')
    box.prop(props, "bone_shape", text="")

    # ---- Shape-specific settings -------------------------------------------
    if props.bone_shape == 'OCTAHEDRAL':
        sub = box.column(align=True)
        sub.prop(props, "octa_base_position", slider=True)
        sub.prop(props, "octa_base_width",    slider=True)

    elif props.bone_shape == 'CYLINDER':
        sub = box.column(align=True)
        sub.prop(props, "cyl_radius_factor", slider=True)
        sub.prop(props, "cyl_segments")

    elif props.bone_shape == 'RECTANGLE':
        sub = box.column(align=True)
        sub.prop(props, "rect_width_factor", slider=True)
        sub.prop(props, "rect_depth_factor", slider=True)

    # ---- Joint sphere settings ---------------------------------------------
    box2 = layout.box()
    box2.label(text="Joint Spheres", icon='MESH_UVSPHERE')

    sub = box2.column(align=True)
    sub.prop(props, "add_joint_spheres", text="Enable Joint Spheres")
    if props.add_joint_spheres:
        sub.prop(props, "joint_radius_factor", slider=True)
        sub.prop(props, "joint_segments")
        sub.prop(props, "joint_rings")

    # ---- Convert button ----------------------------------------------------
    layout.separator()
    col = layout.column()
    obj = context.active_object
    if obj is None or obj.type != 'ARMATURE':
        col.label(text="Select an armature first", icon='ERROR')
        col.enabled = False
    col.scale_y = 1.4
    col.operator("object.bones_to_mesh", icon='OUTLINER_OB_MESH')


class VIEW3D_PT_bones_to_mesh(Panel):
    """N-panel tab in the 3D Viewport sidebar."""
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'Armature to Mesh'
    bl_label       = 'Armature to Mesh'
    bl_idname      = 'VIEW3D_PT_bones_to_mesh'

    def draw(self, context):
        draw_panel_content(self.layout, context)


class DATA_PT_bones_to_mesh(Panel):
    """Sub-panel inside Properties > Object Data when an armature is active."""
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = 'data'
    bl_label       = 'Armature to Mesh'
    bl_idname      = 'DATA_PT_bones_to_mesh'
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def draw(self, context):
        draw_panel_content(self.layout, context)


# ============================================================================
#  Registration
# ============================================================================

CLASSES = [
    BonesToMeshProperties,
    OBJECT_OT_bones_to_mesh,
    VIEW3D_PT_bones_to_mesh,
    DATA_PT_bones_to_mesh,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.bones_to_mesh_props = PointerProperty(type=BonesToMeshProperties)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.bones_to_mesh_props


if __name__ == "__main__":
    register()
