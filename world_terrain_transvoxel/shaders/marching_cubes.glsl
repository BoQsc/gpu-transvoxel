#[compute]
#version 450

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(set = 0, binding = 0, std430) buffer DensityBuffer { float data[]; } density;
layout(set = 0, binding = 1, std430) buffer VertexBuffer { float data[]; } vertices;
layout(set = 0, binding = 2, std430) buffer IndexBuffer { uint data[]; } indices;
layout(set = 0, binding = 3, std430) buffer CounterBuffer { uint vertex_count; uint index_count; uint max_v; uint max_i; } counter;

layout(set = 0, binding = 4, std430) readonly buffer RegularCellClass { int data[]; } regularCellClass;
layout(set = 0, binding = 5, std430) readonly buffer RegularCellData { int data[]; } regularCellData;
layout(set = 0, binding = 6, std430) readonly buffer RegularVertexData { int data[]; } regularVertexData;

layout(push_constant) uniform PushConstants {
    int size;
    float iso_level;
    int pad1;
    int pad2;
} params;

const ivec3 corners[8] = {
    ivec3(0, 0, 0), ivec3(1, 0, 0), ivec3(0, 1, 0), ivec3(1, 1, 0),
    ivec3(0, 0, 1), ivec3(1, 0, 1), ivec3(0, 1, 1), ivec3(1, 1, 1)
};

float get_density(ivec3 p) {
    int s = params.size;
    ivec3 pc = clamp(p, ivec3(0), ivec3(s - 1));
    return density.data[pc.x + pc.y * s + pc.z * s * s];
}

vec3 get_normal(ivec3 p) {
    int s = params.size;
    // Centered difference with clamping
    float dx = get_density(p + ivec3(1,0,0)) - get_density(p - ivec3(1,0,0));
    float dy = get_density(p + ivec3(0,1,0)) - get_density(p - ivec3(0,1,0));
    float dz = get_density(p + ivec3(0,0,1)) - get_density(p - ivec3(0,0,1));
    
    // Boundary fix: if sampling the edge, use one-sided difference
    if (p.x == 0) dx = (get_density(ivec3(1, p.y, p.z)) - get_density(p)) * 2.0;
    if (p.x == s-1) dx = (get_density(p) - get_density(ivec3(s-2, p.y, p.z))) * 2.0;
    if (p.y == 0) dy = (get_density(ivec3(p.x, 1, p.z)) - get_density(p)) * 2.0;
    if (p.y == s-1) dy = (get_density(p) - get_density(ivec3(p.x, s-2, p.z))) * 2.0;
    if (p.z == 0) dz = (get_density(ivec3(p.x, p.y, 1)) - get_density(p)) * 2.0;
    if (p.z == s-1) dz = (get_density(p) - get_density(ivec3(p.x, p.y, s-2))) * 2.0;

    vec3 n = vec3(dx, dy, dz);
    float l = length(n);
    if (l < 0.0001) return vec3(0, 1, 0);
    // Invert gradient for noise - y polarity (Positive = Solid)
    // Positive gradient points TOWARDS higher density (Solid), so we invert for outward normal.
    return -n / l;
}

vec3 interpolate(vec3 p1, float d1, vec3 p2, float d2) {
    float diff = d2 - d1;
    if (abs(diff) < 0.0001) return p1;
    float t = (params.iso_level - d1) / diff;
    return mix(p1, p2, clamp(t, 0.01, 0.99)); // Avoid perfect boundary snap artifacts
}

void main() {
    ivec3 id = ivec3(gl_GlobalInvocationID.xyz);
    int s = params.size;
    if (id.x >= s - 1 || id.y >= s - 1 || id.z >= s - 1) return;

    float d[8];
    int case_index = 0;
    for (int i = 0; i < 8; i++) {
        ivec3 p = id + corners[i];
        d[i] = get_density(p);
        if (d[i] > params.iso_level) case_index |= (1 << i);
    }

    if (case_index == 0 || case_index == 255) return;

    int cell_class = regularCellClass.data[case_index];
    int data_start = cell_class * 16;
    int counts = regularCellData.data[data_start];
    int tri_count = counts & 0x0F;
    int vert_count_cell = (counts >> 4) & 0x0F;

    uint cell_vertex_indices[16];
    for(int i=0; i<16; i++) cell_vertex_indices[i] = 0;

    int vertex_info_start = case_index * 12;

    for (int i = 0; i < vert_count_cell; i++) {
        int info = regularVertexData.data[vertex_info_start + i];
        int i1 = (info >> 4) & 0x0F;
        int i2 = info & 0x0F;
        
        ivec3 p1_i = id + corners[i1];
        ivec3 p2_i = id + corners[i2];
        
        vec3 pos = interpolate(vec3(p1_i), d[i1], vec3(p2_i), d[i2]);
        vec3 n1 = get_normal(p1_i);
        vec3 n2 = get_normal(p2_i);
        vec3 norm = normalize(mix(n1, n2, 0.5));

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
        int i0_rel = regularCellData.data[idx_base + 0];
        int i1_rel = regularCellData.data[idx_base + 1];
        int i2_rel = regularCellData.data[idx_base + 2];
        
        // CCW winding for Godot
        indices.data[i_idx + 0] = cell_vertex_indices[i0_rel];
        indices.data[i_idx + 1] = cell_vertex_indices[i2_rel];
        indices.data[i_idx + 2] = cell_vertex_indices[i1_rel];
    }
}
