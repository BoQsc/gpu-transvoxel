#[compute]
#version 450

// density.glsl - Smooth Forest Floor Foundation

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(set = 0, binding = 0, std430) buffer DensityBuffer { float data[]; } density;

layout(push_constant) uniform PushConstants {
    vec4 offset_and_scale;
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

    vec3 world_pos = params.offset_and_scale.xyz + (vec3(id) - 1.0);
    
    // SCALE: Use ultra-low frequency for broad, walkable mounds (0.04)
    vec3 pos = world_pos * 0.04; 

    // 1. Rolling hill base (Very smooth)
    float h = noise(pos) * 1.0;
    h += noise(pos * 2.0) * 0.4;
    
    // 2. Foundation: Solid ground from Y=10 downwards
    float ground_level = 10.0;
    float surface = ground_level + h * 15.0; 
    
    // 3. Simple, robust density: (Surface - Y)
    float d = surface - world_pos.y;
    
    // Safety bedrock (Y < -5 is always solid)
    if (world_pos.y < -5.0) d = max(d, 5.0);
    
    // Sky Damping: Kill floaties above Y=50
    float damping = smoothstep(80.0, 40.0, world_pos.y);
    d *= damping;

    uint index = uint(id.x) + uint(id.y * s) + uint(id.z * s * s);
    density.data[index] = d;
}
