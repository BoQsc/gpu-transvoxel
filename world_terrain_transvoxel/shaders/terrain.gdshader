shader_type spatial;
render_mode cull_back, diffuse_burley, specular_schlick_ggx;

uniform vec3 grass_color : source_color = vec3(0.2, 0.4, 0.1);
uniform vec3 rock_color : source_color = vec3(0.3, 0.3, 0.35);
uniform float triplanar_sharpness : hint_range(0.0, 10.0) = 4.0;
uniform float slope_threshold : hint_range(0.0, 1.0) = 0.7;
uniform float slope_softness : hint_range(0.0, 1.0) = 0.1;

varying vec3 v_world_pos;
varying vec3 v_normal;
varying float v_dist;

void vertex() {
	v_world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	v_normal = NORMAL;
	v_dist = length(v_world_pos - CAMERA_POSITION_WORLD);
}

// Micro-detail noise for high-res look
float detail_noise(vec3 p) {
	vec3 p2 = p * 12.0;
	float n = fract(sin(dot(p2, vec3(12.989, 78.233, 45.164))) * 43758.5453);
	p2 *= 2.5;
	n += fract(sin(dot(p2, vec3(12.989, 78.233, 45.164))) * 43758.5453) * 0.5;
	return n / 1.5;
}

void fragment() {
	// 1. Calculate triplanar weights
	vec3 blending = abs(v_normal);
	blending /= (blending.x + blending.y + blending.z);
	
	// 2. Slope-based blending (UP vs Sides)
	float slope = v_normal.y;
	float grass_weight = smoothstep(slope_threshold - slope_softness, slope_threshold + slope_softness, slope);
	
	// 3. Detail Fading: Reduce noise contrast at distance to mask LOD popping
	float detail_fade = clamp((v_dist - 40.0) / 100.0, 0.0, 1.0);
	
	// 4. Base procedural "landscape" noise (Mid-freq)
	float noise = fract(sin(dot(v_world_pos.xz, vec2(12.9898, 78.233))) * 43758.5453);
	
	// 5. Micro-detail pass (High-freq) - ONLY for close range
	float m_noise = mix(detail_noise(v_world_pos), 0.5, detail_fade);
	
	vec3 base_color = mix(rock_color, grass_color, grass_weight);
	
	// Mix in the noises. Micro-noise adds the "grit/dirt" feel.
	base_color *= (0.8 + 0.2 * mix(noise, m_noise, 0.7));
	
	// 6. Subtle Ambient Occlusion based on slope (darker in crevices)
	float ao = mix(0.7, 1.0, smoothstep(-0.2, 0.5, slope));
	ALBEDO = base_color * ao;
	
	// 7. Normal Softening: Mix actual normal with a smoothed "Up" normal at extreme distance
	vec3 final_normal = normalize(v_normal);
	NORMAL = mix(final_normal, vec3(0, 1, 0), detail_fade * 0.2);
	
	ROUGHNESS = mix(0.85, 0.95, m_noise * (1.0 - detail_fade));
}
