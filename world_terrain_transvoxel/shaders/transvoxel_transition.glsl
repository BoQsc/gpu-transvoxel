#[compute]
#version 450

// transvoxel_transition.glsl - Technical Correctness & Seam Fix

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0, std430) buffer DensityBuffer { float data[]; } density;
layout(set = 0, binding = 1, std430) buffer VertexBuffer { float data[]; } vertices;
layout(set = 0, binding = 2, std430) buffer IndexBuffer { uint data[]; } indices;
layout(set = 0, binding = 3, std430) buffer CounterBuffer { uint vertex_count; uint index_count; uint max_v; uint max_i; } counter;

layout(set = 0, binding = 4, std430) readonly buffer TransitionCellClass { int data[]; } transitionCellClass;
layout(set = 0, binding = 5, std430) readonly buffer TransitionCellData { int data[]; } transitionCellData;
layout(set = 0, binding = 6, std430) readonly buffer TransitionCornerData { int data[]; } transitionCornerData;
layout(set = 0, binding = 7, std430) readonly buffer TransitionVertexData { int data[]; } transitionVertexData;

layout(push_constant) uniform PushConstants {
    int size;
    float iso_level;
    int face; 
    int pad;
} params;

// Map local 2D (u, v) on a face to 3D voxel coordinate
ivec3 get_pos_3d(int u, int v) {
    int s = params.size - 3; 
    switch(params.face) {
        case 0: return ivec3(s, v, u); // +X
        case 1: return ivec3(0, u, v); // -X
        case 2: return ivec3(u, s, v); // +Y
        case 3: return ivec3(v, 0, u); // -Y
        case 4: return ivec3(u, v, s); // +Z
        case 5: return ivec3(u, v, 0); // -Z
    }
    return ivec3(0);
}

// Convert corner ID (0-12) to 3D voxel index with padding compensation
ivec3 corner_to_pos(int c, ivec2 base_uv) {
    ivec3 p3d = ivec3(0);
    // Transvoxel grid mapping: 0-8 are high-res 3x3, 9-12 are low-res corners
    if (c < 9) p3d = get_pos_3d(base_uv.x + (c % 3), base_uv.y + (c / 3));
    else if (c == 9) p3d = get_pos_3d(base_uv.x + 0, base_uv.y + 0);
    else if (c == 10) p3d = get_pos_3d(base_uv.x + 2, base_uv.y + 0);
    else if (c == 11) p3d = get_pos_3d(base_uv.x + 0, base_uv.y + 2);
    else if (c == 12) p3d = get_pos_3d(base_uv.x + 2, base_uv.y + 2);
    
    return p3d + ivec3(1); // Shift by 1 for padded density buffer indexing
}

float get_density(ivec3 p) {
    int s = params.size;
    ivec3 pc = clamp(p, ivec3(0), ivec3(s - 1));
    return density.data[pc.x + pc.y * s + pc.z * s * s];
}

vec3 get_normal(ivec3 p) {
    float dx = get_density(p + ivec3(1,0,0)) - get_density(p - ivec3(1,0,0));
    float dy = get_density(p + ivec3(0,1,0)) - get_density(p - ivec3(0,1,0));
    float dz = get_density(p + ivec3(0,0,1)) - get_density(p - ivec3(0,0,1));
    
    // Smooth pass for organic lighting
    dx += (get_density(p + ivec3(2,0,0)) - get_density(p - ivec3(2,0,0))) * 0.5;
    dy += (get_density(p + ivec3(0,2,0)) - get_density(p - ivec3(0,2,0))) * 0.5;
    dz += (get_density(p + ivec3(0,0,2)) - get_density(p - ivec3(0,0,2))) * 0.5;

    vec3 n = vec3(dx, dy, dz);
    float l = length(n);
    if (l < 0.0001) return vec3(0, 1, 0);
    return -n / l;
}

vec4 interpolate_data(vec3 p1, float d1, vec3 p2, float d2) {
    float diff = d2 - d1;
    if (abs(diff) < 0.0001) return vec4(p1, 0.0);
    float t = clamp((params.iso_level - d1) / diff, 0.0, 1.0);
    return vec4(mix(p1, p2, t), t);
}

void main() {
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    int chunk_size = params.size - 3; 
    int num_cells = chunk_size / 2; 
    
    if (id.x >= num_cells || id.y >= num_cells) return;
    
    ivec2 base_uv = id * 2;

    float d[13];
    int case_index = 0;
    for (int i = 0; i < 9; i++) {
        ivec3 p = corner_to_pos(i, base_uv);
        d[i] = get_density(p);
        if (d[i] > params.iso_level) case_index |= (1 << i);
    }

    // Secondary nodes for the low-res side
    d[9] = d[0]; d[10] = d[2]; d[11] = d[6]; d[12] = d[8];

    if (case_index == 0 || case_index == 511) return;

    int cell_class_raw = transitionCellClass.data[case_index];
    int cell_class = cell_class_raw & 0x7F;
    bool swap_winding = (cell_class_raw & 0x80) != 0;

    int data_start = cell_class * 37;
    int counts = transitionCellData.data[data_start];
    int tri_count = counts & 0x0F;
    int vert_count_cell = (counts >> 4) & 0x0F;

    uint cell_vertex_indices[12];
    int vertex_info_start = case_index * 12;

    vec3 face_normal;
    switch(params.face) {
        case 0: face_normal = vec3(1,0,0); break;
        case 1: face_normal = vec3(-1,0,0); break;
        case 2: face_normal = vec3(0,1,0); break;
        case 3: face_normal = vec3(0,-1,0); break;
        case 4: face_normal = vec3(0,0,1); break;
        case 5: face_normal = vec3(0,0,-1); break;
    }

    for (int i = 0; i < vert_count_cell; i++) {
        int info = transitionVertexData.data[vertex_info_start + i];
        int i1 = (info >> 4) & 0x0F;
        int i2 = info & 0x0F;
        
        ivec3 p1_i = corner_to_pos(i1, base_uv);
        ivec3 p2_i = corner_to_pos(i2, base_uv);
        
        vec4 interp = interpolate_data(vec3(p1_i), d[i1], vec3(p2_i), d[i2]);
        vec3 pos = interp.xyz - 1.0; 
        
        // Z-bias to prevent Z-fighting with regular chunk
        pos += face_normal * 0.005;
        
        float t = interp.w;
        vec3 norm = normalize(mix(get_normal(p1_i), get_normal(p2_i), t));

        uint v_idx = atomicAdd(counter.vertex_count, 1);
        if (v_idx >= counter.max_v) return;

        uint base = v_idx * 6;
        vertices.data[base + 0] = pos.x;
        vertices.data[base + 1] = pos.y;
        vertices.data[base + 2] = pos.z;
        vertices.data[base + 3] = norm.x;
        vertices.data[base + 4] = norm.y;
        vertices.data[base + 5] = norm.z;

        cell_vertex_indices[i] = v_idx;
    }

    for (int i = 0; i < tri_count; i++) {
        uint i_idx = atomicAdd(counter.index_count, 3);
        if (i_idx + 2 >= counter.max_i) return;

        int idx_base = data_start + 1 + i * 3;
        uint v0 = cell_vertex_indices[transitionCellData.data[idx_base + 0]];
        uint v1 = cell_vertex_indices[transitionCellData.data[idx_base + 1]];
        uint v2 = cell_vertex_indices[transitionCellData.data[idx_base + 2]];
        
        if (swap_winding) {
            indices.data[i_idx + 0] = v0;
            indices.data[i_idx + 1] = v1;
            indices.data[i_idx + 2] = v2;
        } else {
            indices.data[i_idx + 0] = v0;
            indices.data[i_idx + 1] = v2;
            indices.data[i_idx + 2] = v1;
        }
    }
}
