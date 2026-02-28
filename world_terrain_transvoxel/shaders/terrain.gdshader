shader_type spatial;
render_mode cull_back, diffuse_burley, specular_schlick_ggx;

uniform vec3 grass_color : source_color = vec3(0.2, 0.4, 0.1);
uniform vec3 rock_color : source_color = vec3(0.3, 0.3, 0.35);
uniform float triplanar_sharpness : hint_range(0.0, 10.0) = 4.0;
uniform float slope_threshold : hint_range(0.0, 1.0) = 0.7;
uniform float slope_softness : hint_range(0.0, 1.0) = 0.1;

varying vec3 v_world_pos;
varying vec3 v_normal;

void vertex() {
	v_world_pos = (MODEL_MATRIX * vec4(VERTEX, 1.0)).xyz;
	v_normal = NORMAL;
}

void fragment() {
	// 1. Calculate triplanar weights
	vec3 blending = abs(v_normal);
	blending /= (blending.x + blending.y + blending.z);
	
	// 2. Slope-based blending (UP vs Sides)
	float slope = v_normal.y;
	float grass_weight = smoothstep(slope_threshold - slope_softness, slope_threshold + slope_softness, slope);
	
	// 3. Simple procedural "noise" for texture break-up
	float noise = fract(sin(dot(v_world_pos.xz, vec2(12.9898, 78.233))) * 43758.5453);
	vec3 base_color = mix(rock_color, grass_color, grass_weight);
	
	// 4. Subtle color variation for distance
	float dist = length(v_world_pos - CAMERA_POSITION_WORLD);
	base_color *= (0.9 + 0.1 * noise);
	
	ALBEDO = base_color;
	NORMAL = normalize(v_normal); // Ensure normalization for shadows
	ROUGHNESS = 0.8;
}
