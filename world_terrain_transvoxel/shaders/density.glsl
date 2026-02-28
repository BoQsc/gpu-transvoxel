#[compute]
#version 450

// density.glsl - Advanced organic terrain generation

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

// Ridged noise for sharp mountain peaks
float ridged_noise(vec3 p) {
    return 1.0 - abs(noise(p) * 2.0 - 1.0);
}

void main() {
    ivec3 id = ivec3(gl_GlobalInvocationID.xyz);
    int s = params.size;
    if (id.x >= s || id.y >= s || id.z >= s) return;

    vec3 world_pos = params.offset_and_scale.xyz + (vec3(id) - 1.0);
    vec3 pos = world_pos * params.offset_and_scale.w;

    // 1. Domain Warping: Distort coords with noise for organic flow
    vec3 warp = vec3(
        noise(pos + vec3(0.0)),
        noise(pos + vec3(5.2)),
        noise(pos + vec3(1.3))
    );
    pos += warp * 0.5;

    // 2. Base Terrain Height (FBM)
    float base_noise = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    for (int i = 0; i < 4; i++) {
        base_noise += noise(pos * freq) * amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    
    // 3. Mountains: Ridged Multi-fractal for sharp peaks
    float ridge_noise = 0.0;
    amp = 1.0;
    freq = 0.5;
    for (int i = 0; i < 3; i++) {
        ridge_noise += ridged_noise(pos * freq + warp * 0.3) * amp;
        amp *= 0.5;
        freq *= 2.5; 
    }
    
    // 4. Composition: Mix base hills with sharp ridges
    float final_terrain = mix(base_noise * 12.0, ridge_noise * 25.0, clamp(ridge_noise, 0.0, 1.0));
    
    uint index = uint(id.x) + uint(id.y * s) + uint(id.z * s * s);
    // Standard Polarity: Positive = Solid, Negative = Air
    density.data[index] = (final_terrain - 20.0) - world_pos.y;
}
