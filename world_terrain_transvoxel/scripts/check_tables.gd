extends SceneTree

func _init():
	print("--- Transvoxel Table Integrity Check ---")
	
	var tables = {
		"regularCellClass": TransvoxelData.regularCellClass,
		"regularCellData": TransvoxelData.regularCellData,
		"regularVertexData": TransvoxelData.regularVertexData,
		"transitionCellClass": TransvoxelData.transitionCellClass,
		"transitionCellData": TransvoxelData.transitionCellData,
		"transitionVertexData": TransvoxelData.transitionVertexData
	}
	
	for name in tables:
		var arr = tables[name]
		print("%s size: %d" % [name, arr.size()])
		if arr.size() > 0:
			var samples = []
			for i in range(min(5, arr.size())):
				samples.append("0x%X" % arr[i])
			print("  Samples: %s" % ", ".join(samples))

	# Analyze regularVertexData mapping
	var reg_cell_packed = TransvoxelData.regularCellData
	var reg_cell_class = TransvoxelData.regularCellClass
	var total_v = 0
	for i in range(256):
		var class_idx = reg_cell_class[i]
		# Find this class in cell data (first pass for class offsets)
		var p = 0
		for c in range(class_idx):
			var h = reg_cell_packed[p]
			p += 1 + (h & 0x0F) * 3
		var header = reg_cell_packed[p]
		var v_count = (header >> 4) & 0x0F
		total_v += v_count
	
	print("Expected regularVertexData size if per-case: %d" % total_v)
	
	# Check transitions
	var t_cell_class = TransvoxelData.transitionCellClass
	var t_cell_packed = TransvoxelData.transitionCellData
	var total_tv = 0
	for i in range(512):
		var class_idx = t_cell_class[i] & 0x7F
		var p = 0
		for c in range(class_idx):
			var h = t_cell_packed[p]
			p += 1 + (h & 0x0F) * 3
		var header = t_cell_packed[p]
		var v_count = (header >> 4) & 0x0F
		total_tv += v_count
	
	print("Expected transitionVertexData size if per-case: %d" % total_tv)
	
	quit()
