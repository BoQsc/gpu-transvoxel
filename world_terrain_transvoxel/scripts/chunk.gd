extends MeshInstance3D
class_name TransvoxelChunk

var chunk_size: int = 32
var lod: int = 0
var rd: RenderingDevice
var neighbor_lods: Array[int] = [0, 0, 0, 0, 0, 0] # +X, -X, +Y, -Y, +Z, -Z

# SSBO RIDs
var density_buffer: RID
var vertex_buffer: RID
var index_buffer: RID
var counter_buffer: RID

# Table RIDs
var table_rids: Array[RID] = []

# Shader RIDs (Shared)
var density_shader: RID
var regular_shader: RID
var transition_shader: RID

# Pipeline RIDs (Shared)
var density_pipeline: RID
var regular_pipeline: RID
var transition_pipeline: RID

func _ready() -> void:
	pass # All initialized via ChunkManager

func generate() -> void:
	var s = chunk_size + 1 # +1 for MC across boundaries
	var num_voxels = s * s * s
	
	_cleanup_buffers()
	
	# 1. Create density buffer
	density_buffer = rd.storage_buffer_create(num_voxels * 4) # float32
	
	# 2. Create vertex/index buffers (Large enough to avoid overflow)
	# 400k vertices * 6 floats * 4 bytes = 9.6MB
	# 800k indices * 4 bytes = 3.2MB
	var max_v = 400000
	var max_i = 800000
	vertex_buffer = rd.storage_buffer_create(max_v * 6 * 4) 
	index_buffer = rd.storage_buffer_create(max_i * 4)
	
	# 3. Create counter buffer (vertex_count, index_count, max_v, max_i)
	var counters = PackedInt32Array([0, 0, max_v, max_i])
	counter_buffer = rd.storage_buffer_create(16, counters.to_byte_array())

	_dispatch_density(s)
	# Barriers are implicit in RenderingDevice for storage buffers
	_dispatch_regular(s)
	
	for i in range(6):
		if neighbor_lods[i] > lod:
			_dispatch_transition(s, i)

	rd.submit()
	rd.sync()
	
	_create_mesh()
	_cleanup_buffers()

func _cleanup_buffers() -> void:
	if density_buffer.is_valid(): rd.free_rid(density_buffer); density_buffer = RID()
	if vertex_buffer.is_valid(): rd.free_rid(vertex_buffer); vertex_buffer = RID()
	if index_buffer.is_valid(): rd.free_rid(index_buffer); index_buffer = RID()
	if counter_buffer.is_valid(): rd.free_rid(counter_buffer); counter_buffer = RID()

func _dispatch_transition(s: int, face: int) -> void:
	if not transition_pipeline.is_valid(): return
	var t_uniforms: Array[RDUniform] = []
	var bindings = [density_buffer, vertex_buffer, index_buffer, counter_buffer]
	bindings.append_array(table_rids.slice(3, 7)) # Transition tables
	
	for i in range(bindings.size()):
		var u = RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = i
		u.add_id(bindings[i])
		t_uniforms.append(u)
		
	var uniform_set = rd.uniform_set_create(t_uniforms, transition_shader, 0)
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, transition_pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	
	# Push: size (int), iso (float), face (int), pad (int) = 16 bytes
	var push = PackedByteArray()
	push.resize(16)
	push.encode_s32(0, s)
	push.encode_float(4, 0.0)
	push.encode_s32(8, face)
	push.encode_s32(12, 0)
	
	rd.compute_list_set_push_constant(compute_list, push, push.size())
	
	var groups = (s / 2 + 7) / 8
	rd.compute_list_dispatch(compute_list, groups, groups, 1)
	rd.compute_list_end()

func _dispatch_density(s: int) -> void:
	if not density_pipeline.is_valid(): return
	var uniform = RDUniform.new()
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform.binding = 0
	uniform.add_id(density_buffer)
	var uniform_set = rd.uniform_set_create([uniform], density_shader, 0)
	
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, density_pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	
	# Push: vec4(pos, scale), size(int), iso(float), pad, pad = 32 bytes
	var push = PackedByteArray()
	push.resize(32)
	push.encode_float(0, position.x)
	push.encode_float(4, position.y)
	push.encode_float(8, position.z)
	push.encode_float(12, 0.1) # scale
	push.encode_s32(16, s)
	push.encode_float(20, 0.0)
	push.encode_s32(24, 0)
	push.encode_s32(28, 0)
	
	rd.compute_list_set_push_constant(compute_list, push, push.size())
	
	var groups = (s + 3) / 4
	rd.compute_list_dispatch(compute_list, groups, groups, groups)
	rd.compute_list_end()

func _dispatch_regular(s: int) -> void:
	if not regular_pipeline.is_valid(): return
	var r_uniforms: Array[RDUniform] = []
	var bindings = [density_buffer, vertex_buffer, index_buffer, counter_buffer]
	bindings.append_array(table_rids.slice(0, 3)) # Regular tables
	
	for i in range(bindings.size()):
		var u = RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = i
		u.add_id(bindings[i])
		r_uniforms.append(u)
		
	var uniform_set = rd.uniform_set_create(r_uniforms, regular_shader, 0)
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, regular_pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	
	# Push: size(int), iso(float), pad, pad = 16 bytes
	var push = PackedByteArray()
	push.resize(16)
	push.encode_s32(0, s)
	push.encode_float(4, 0.0)
	push.encode_s32(8, 0)
	push.encode_s32(12, 0)
	
	rd.compute_list_set_push_constant(compute_list, push, push.size())
	
	var groups = (s + 3) / 4
	rd.compute_list_dispatch(compute_list, groups, groups, groups)
	rd.compute_list_end()

func _create_mesh() -> void:
	# 1. Read counters
	var c_bytes = rd.buffer_get_data(counter_buffer)
	if c_bytes.is_empty(): return
	var counters = c_bytes.to_int32_array()
	var v_count = min(counters[0], counters[2]) # Bound by max_v
	var i_count = min(counters[1], counters[3]) # Bound by max_i
	
	if v_count <= 0 or i_count <= 0:
		return

	# 2. Read vertex data (float32: x, y, z, nx, ny, nz)
	var actual_v_bytes = v_count * 6 * 4
	var v_bytes = rd.buffer_get_data(vertex_buffer, 0, actual_v_bytes)
	var v_floats = v_bytes.to_float32_array()
	
	var vertices_array = PackedVector3Array()
	var normals_array = PackedVector3Array()
	vertices_array.resize(v_count)
	normals_array.resize(v_count)
	
	for i in range(v_count):
		vertices_array[i] = Vector3(v_floats[i * 6 + 0], v_floats[i * 6 + 1], v_floats[i * 6 + 2])
		normals_array[i] = Vector3(v_floats[i * 6 + 3], v_floats[i * 6 + 4], v_floats[i * 6 + 5])

	# 3. Read index data
	var i_bytes = rd.buffer_get_data(index_buffer, 0, i_count * 4)
	var indices_ptr = i_bytes.to_int32_array()

	# 4. Build ArrayMesh
	var arr = []
	arr.resize(Mesh.ARRAY_MAX)
	arr[Mesh.ARRAY_VERTEX] = vertices_array
	arr[Mesh.ARRAY_NORMAL] = normals_array
	arr[Mesh.ARRAY_INDEX] = indices_ptr

	var final_mesh = ArrayMesh.new()
	final_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arr)
	mesh = final_mesh
	
	# Create collision
	_create_collision(vertices_array, indices_ptr)
	
	# Apply basic material
	if not material_override:
		var mat = StandardMaterial3D.new()
		mat.uv1_triplanar = true
		mat.albedo_color = Color(0.4, 0.7, 0.3) # Grass green
		material_override = mat

func _create_collision(verts: PackedVector3Array, idxs: PackedInt32Array) -> void:
	var shape = ConcavePolygonShape3D.new()
	var collision_faces = PackedVector3Array()
	collision_faces.resize(idxs.size())
	
	var v_size = verts.size()
	for i in range(idxs.size()):
		var idx = idxs[i]
		if idx < 0 or idx >= v_size:
			collision_faces[i] = Vector3.ZERO
			continue
		collision_faces[i] = verts[idx]
		
	shape.set_faces(collision_faces)
	
	# Re-use or rebuild static body
	for child in get_children():
		if child is StaticBody3D:
			child.free()
			
	var static_body = StaticBody3D.new()
	var collision_shape = CollisionShape3D.new()
	collision_shape.shape = shape
	static_body.add_child(collision_shape)
	add_child(static_body)
