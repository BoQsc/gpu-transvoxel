#[compute]
#version 450

// density.glsl - Generates the voxel density field

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(set = 0, binding = 0, std430) buffer DensityBuffer {
    float data[];
} density;

layout(push_constant) uniform PushConstants {
    vec4 offset_and_scale; // xyz=pos, w=scale
    int size;
    float iso_level;
    int pad1;
    int pad2;
} params;

float hash(vec3 p) {
    vec3 p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i + vec3(0, 0, 0)), hash(i + vec3(1, 0, 0)), f.x),
                   mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
                   mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

void main() {
    ivec3 id = ivec3(gl_GlobalInvocationID.xyz);
    int s = params.size;
    if (id.x >= s || id.y >= s || id.z >= s) return;

    // Standard Mapping:
    // id=1 is world_pos = offset (chunk origin)
    // id=size-2 is world_pos = offset + chunk_size
    vec3 world_pos = params.offset_and_scale.xyz + (vec3(id) - 1.0);
    vec3 pos = world_pos * params.offset_and_scale.w;

    float noise_sum = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    for (int i = 0; i < 5; i++) {
        noise_sum += noise(pos * freq) * amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    noise_sum *= 15.0;
    
    uint index = uint(id.x) + uint(id.y * s) + uint(id.z * s * s);
    density.data[index] = (noise_sum - 15.0) - world_pos.y;
}
