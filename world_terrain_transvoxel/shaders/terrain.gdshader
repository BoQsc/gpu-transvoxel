shader_type spatial;
render_mode cull_back, diffuse_burley, specular_schlick_ggx;

// Vibrant Stylized Forest Palette
uniform vec3 grass_color : source_color = vec3(0.4, 0.75, 0.2); // Saturated Lime Green
uniform vec3 rock_color : source_color = vec3(0.35, 0.45, 0.6); // Cool Slate Blue
uniform float slope_threshold : hint_range(0.0, 1.0) = 0.65;
uniform float slope_softness : hint_range(0.0, 1.0) = 0.15;

varying vec3 v_world_pos;
varying vec3 v_normal;
varying float v_dist;

void vertex() {
	v_world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	v_normal = NORMAL;
	v_dist = length(v_world_pos - CAMERA_POSITION_WORLD);
}

// Organic Value Noise for Painted Variety (No Grain)
float hash3(vec3 p) {
	p = fract(p * 0.1031);
	p += dot(p, p.yzx + 33.33);
	return fract((p.x + p.y) * p.z);
}

float v_noise(vec3 p) {
	vec3 i = floor(p);
	vec3 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	return mix(mix(mix(hash3(i + vec3(0,0,0)), hash3(i + vec3(1,0,0)), f.x),
	               mix(hash3(i + vec3(0,1,0)), hash3(i + vec3(1,1,0)), f.x), f.y),
	           mix(mix(hash3(i + vec3(0,0,1)), hash3(i + vec3(1,0,1)), f.x),
	               mix(hash3(i + vec3(0,1,1)), hash3(i + vec3(1,1,1)), f.x), f.y), f.z);
}

void fragment() {
	// 1. Triplanar Weights
	vec3 blending = abs(v_normal);
	blending /= (blending.x + blending.y + blending.z);
	
	// 2. Slope Blending
	float slope = v_normal.y;
	float grass_weight = smoothstep(slope_threshold - slope_softness, slope_threshold + slope_softness, slope);
	
	// 3. Painted Variation
	// We use large noise patches for color variety, matching the "image goal"
	float patch = v_noise(v_world_pos * 0.1);
	float detail = v_noise(v_world_pos * 0.5);
	
	vec3 base_color = mix(rock_color, grass_color, grass_weight);
	
	// Add "Sun-kissed" warmth to patches
	vec3 warm_tint = vec3(0.5, 0.8, 0.2); // Brighter meadow green
	base_color = mix(base_color, warm_tint, patch * 0.2 * grass_weight);
	
	// Soft grit for depth
	base_color *= (0.92 + 0.1 * detail);
	
	// 4. Stylized Lighting & AO
	// Darken crevices and sides to ground the hills
	float ao = mix(0.5, 1.0, smoothstep(-0.2, 0.5, slope));
	ALBEDO = base_color * ao;
	
	// 5. Normal Softening for "Creamy" feel
	float detail_fade = clamp((v_dist - 30.0) / 100.0, 0.0, 1.0);
	vec3 final_normal = normalize(v_normal);
	NORMAL = mix(final_normal, vec3(0, 1, 0), detail_fade * 0.3);
	
	ROUGHNESS = mix(0.9, 1.0, detail_fade);
}
