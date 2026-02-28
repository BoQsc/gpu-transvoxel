extends Node3D
class_name TransvoxelChunkManager

@export var player: Node3D
@export var chunk_size: int = 32
@export var view_distance: int = 4

@export var lod_dist_0: float = 64.0
@export var lod_dist_1: float = 128.0
@export var lod_dist_2: float = 256.0

var chunks: Dictionary = {} # Vector3i -> TransvoxelChunk

# Shared GPU Resources
var rd: RenderingDevice
var density_shader: RID
var regular_shader: RID
var transition_shader: RID

var density_pipeline: RID
var regular_pipeline: RID
var transition_pipeline: RID

var table_rids: Array[RID] = []

func _ready() -> void:
	rd = RenderingServer.create_local_rendering_device()
	if not rd:
		printerr("RenderingDevice not available!")
		set_process(false)
		return
	
	_init_shaders()
	_init_tables()

func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE:
		if rd:
			if density_pipeline.is_valid(): rd.free_rid(density_pipeline)
			if regular_pipeline.is_valid(): rd.free_rid(regular_pipeline)
			if transition_pipeline.is_valid(): rd.free_rid(transition_pipeline)
			
			if density_shader.is_valid(): rd.free_rid(density_shader)
			if regular_shader.is_valid(): rd.free_rid(regular_shader)
			if transition_shader.is_valid(): rd.free_rid(transition_shader)
			
			for table in table_rids:
				if table.is_valid(): rd.free_rid(table)

func _init_shaders() -> void:
	density_shader = _load_shader("res://world_terrain_transvoxel/shaders/density.glsl")
	regular_shader = _load_shader("res://world_terrain_transvoxel/shaders/marching_cubes.glsl")
	transition_shader = _load_shader("res://world_terrain_transvoxel/shaders/transvoxel_transition.glsl")
	
	if density_shader.is_valid(): density_pipeline = rd.compute_pipeline_create(density_shader)
	if regular_shader.is_valid(): regular_pipeline = rd.compute_pipeline_create(regular_shader)
	if transition_shader.is_valid(): transition_pipeline = rd.compute_pipeline_create(transition_shader)

func _load_shader(path: String) -> RID:
	var shader_file: RDShaderFile = load(path)
	if not shader_file: return RID()
	var spirv = shader_file.get_spirv()
	return rd.shader_create_from_spirv(spirv)

func _init_tables() -> void:
	# 1. regularCellClass (Direct)
	table_rids.append(_create_ssbo(TransvoxelData.regularCellClass.to_byte_array()))
	
	# 2. regularCellData (Pad to 16)
	var reg_cell_packed = TransvoxelData.regularCellData
	var reg_cell_padded = PackedInt32Array()
	reg_cell_padded.resize(16 * 16) # 16 classes
	var reg_offsets = []
	reg_offsets.resize(16)
	var p_idx = 0
	for c in range(16):
		reg_offsets[c] = p_idx
		var header = reg_cell_packed[p_idx]
		var tri_count = header & 0x0F
		var size = 1 + tri_count * 3
		for i in range(size):
			reg_cell_padded[c * 16 + i] = reg_cell_packed[p_idx + i]
		p_idx += size
	table_rids.append(_create_ssbo(reg_cell_padded.to_byte_array()))

	# 3. regularVertexData (Pad to 12)
	var reg_vert_packed = TransvoxelData.regularVertexData
	var reg_vert_padded = PackedInt32Array()
	reg_vert_padded.resize(256 * 12)
	p_idx = 0
	for case_idx in range(256):
		var class_idx = TransvoxelData.regularCellClass[case_idx]
		var header = reg_cell_packed[reg_offsets[class_idx]]
		var v_count = (header >> 4) & 0x0F
		for i in range(v_count):
			reg_vert_padded[case_idx * 12 + i] = reg_vert_packed[p_idx + i]
		p_idx += v_count
	table_rids.append(_create_ssbo(reg_vert_padded.to_byte_array()))
	
	# Transition Tables
	var trans_cell_class = TransvoxelData.transitionCellClass
	var max_t_class = 0
	for v in trans_cell_class:
		max_t_class = max(max_t_class, v & 0x7F)
	var num_t_classes = max_t_class + 1
	
	# 4. transitionCellClass (Direct)
	table_rids.append(_create_ssbo(trans_cell_class.to_byte_array()))
	
	# 5. transitionCellData (Pad to 37)
	var t_cell_packed = TransvoxelData.transitionCellData
	var t_cell_padded = PackedInt32Array()
	t_cell_padded.resize(num_t_classes * 37)
	var t_offsets = []
	t_offsets.resize(num_t_classes)
	p_idx = 0
	for c in range(num_t_classes):
		t_offsets[c] = p_idx
		var header = t_cell_packed[p_idx]
		var tri_count = header & 0x0F
		var size = 1 + tri_count * 3
		for i in range(size):
			t_cell_padded[c * 37 + i] = t_cell_packed[p_idx + i]
		p_idx += size
	table_rids.append(_create_ssbo(t_cell_padded.to_byte_array()))
	
	# 6. transitionCornerData (Direct)
	table_rids.append(_create_ssbo(TransvoxelData.transitionCornerData.to_byte_array()))
	
	# 7. transitionVertexData (Pad to 12)
	var t_vert_packed = TransvoxelData.transitionVertexData
	var t_vert_padded = PackedInt32Array()
	t_vert_padded.resize(512 * 12)
	p_idx = 0
	for case_idx in range(512):
		var raw_class = trans_cell_class[case_idx]
		var class_idx = raw_class & 0x7F
		var header = t_cell_packed[t_offsets[class_idx]]
		var v_count = (header >> 4) & 0x0F
		for i in range(v_count):
			t_vert_padded[case_idx * 12 + i] = t_vert_packed[p_idx + i]
		p_idx += v_count
	table_rids.append(_create_ssbo(t_vert_padded.to_byte_array()))

func _create_ssbo(data: PackedByteArray) -> RID:
	return rd.storage_buffer_create(data.size(), data)

func _process(_delta: float) -> void:
	if not player: return
	_update_chunks()

func _update_chunks() -> void:
	var player_pos = player.global_position
	var p_chunk = Vector3i(
		floor(player_pos.x / chunk_size),
		floor(player_pos.y / chunk_size),
		floor(player_pos.z / chunk_size)
	)

	var radius = view_distance
	var active_chunks = {}

	for x in range(-radius, radius + 1):
		for y in range(-2, 1):
			for z in range(-radius, radius + 1):
				var pos = p_chunk + Vector3i(x, y, z)
				var dist = (Vector3(pos) * chunk_size - player_pos).length()
				var lod = 0
				if dist > lod_dist_2: lod = 3
				elif dist > lod_dist_1: lod = 2
				elif dist > lod_dist_0: lod = 1
				
				active_chunks[pos] = lod
				
				if not chunks.has(pos):
					_spawn_chunk(pos, lod)
				elif chunks[pos].lod != lod:
					_rebuild_chunk(pos, lod)

	var to_remove = []
	for pos in chunks:
		if not active_chunks.has(pos):
			to_remove.append(pos)
	for pos in to_remove:
		chunks[pos].queue_free()
		chunks.erase(pos)

	for pos in chunks:
		var chunk = chunks[pos]
		var dirs = [Vector3i(1,0,0), Vector3i(-1,0,0), Vector3i(0,1,0), Vector3i(0,-1,0), Vector3i(0,0,1), Vector3i(0,0,-1)]
		
		var needs_update = false
		for i in range(6):
			var n_pos = pos + dirs[i]
			var n_lod = active_chunks.get(n_pos, chunk.lod)
			if n_lod != chunk.neighbor_lods[i]:
				chunk.neighbor_lods[i] = n_lod
				needs_update = true
		
		if needs_update:
			chunk.generate()

func _spawn_chunk(pos: Vector3i, lod: int) -> void:
	var chunk = TransvoxelChunk.new()
	chunk.position = Vector3(pos) * chunk_size
	chunk.chunk_size = chunk_size
	chunk.lod = lod
	
	# Pass shared resources
	chunk.rd = rd
	chunk.density_shader = density_shader
	chunk.regular_shader = regular_shader
	chunk.transition_shader = transition_shader
	
	chunk.density_pipeline = density_pipeline
	chunk.regular_pipeline = regular_pipeline
	chunk.transition_pipeline = transition_pipeline
	
	chunk.table_rids = table_rids
	
	add_child(chunk)
	chunks[pos] = chunk
	chunk.generate()

func _rebuild_chunk(pos: Vector3i, lod: int) -> void:
	var chunk = chunks[pos]
	chunk.lod = lod
	chunk.generate()
