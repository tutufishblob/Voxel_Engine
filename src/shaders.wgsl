
struct Uniforms {
  mvp: mat4x4<f32>
}

struct VertexInput {
  @location(0) position: vec3f,
  @location(1) tex_coords:vec2f,
  @location(2) instance_pos: vec3f,
  
  @builtin(vertex_index) vertex_index: u32,
}

struct VertexOutput {
  @builtin(position) position:vec4f,
  @location(0) tex_coords:vec2f,
}

struct WireframeVertexInput {
  @location(0) position:vec3f,
  @location(1) instance_pos: vec3f,
}

struct CrosshairVertexInput {
  @location(0) position:vec2f,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(0) var my_texture: texture_2d<f32>;
@group(1) @binding(1) var my_sampler: sampler;

// struct Ray {
//   origin: vec3f,
//   direction: vec3f,
// }

// fn sky_color(ray: Ray) -> vec3f {
//   let t = 0.5 * (normalize(ray.direction).y + 1.);
//   return (1. - t) * vec3(1.) + t * vec3(0.3,0.5,1.);
// }

//fn two_dimension_projection(cube: CubeVertices, ray: Ray) -> TransformedCube{
  
//}

// alias TriangleVertices = array<vec2f, 6>;
// var<private> vertices: TriangleVertices = TriangleVertices(
//   vec2f(-0.8, 0.8),
//   vec2f( -0.8, -0.8),
//   vec2f( 0.8,  0.8),
//   vec2f( 0.8,  0.8),
//   vec2f( -0.8,  -0.8),
//   vec2f( 0.8,  -0.8),
// );

// var<private> colors : array<vec3f,6> = array<vec3f,6>(
//   vec3f(1.0, 0.0, 0.0), // Red
//   vec3f(0.0, 1.0, 0.0), // Green
//   vec3f(0.0, 0.0, 1.0), // Blue
//   vec3f(1.0, 1.0, 0.0), // Yellow
//   vec3f(1.0, 0.0, 1.0), // Magenta
//   vec3f(0.0, 1.0, 1.0), // Cyan
// );


@vertex 
fn display_vs(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  let cube_position = in.position + in.instance_pos;
  out.position = uniforms.mvp * vec4f(cube_position, 1.0);
  out.tex_coords = in.tex_coords;
  return out;
}

@fragment 
fn display_fs(@location(0) tex_coords: vec2f) -> @location(0) vec4f {
  return textureSample(my_texture, my_sampler, tex_coords);
}


@vertex
fn wireframe_vs(in: WireframeVertexInput) -> @builtin(position) vec4f {
  // let scaled_position = in.position * 1.01; //unnecessary
  // return uniforms.mvp * vec4f(scaled_position, 1.0);
  let wireframe_position = in.position + in.instance_pos;
  return uniforms.mvp * vec4f(wireframe_position,1.0);
}

@fragment 
fn wireframe_fs() -> @location(0) vec4f {
  return vec4f(0.0, 0.0, 0.0, 1.0);
  //return vec4(1.0, 0.0, 1.0, 1.0); //testing a different color to ensure its working

}

@vertex
fn crosshair_vs(in: CrosshairVertexInput) -> @builtin(position) vec4f {
  let scaled_position = in.position * 0.7; //unnecessary
  // return uniforms.mvp * vec4f(scaled_position, 1.0);
  return vec4f(scaled_position,0.0,1.0);
}

@fragment 
fn crosshair_fs() -> @location(0) vec4f {
  return vec4f(0.5, 0.5, 0.5, 0.5);
  

}