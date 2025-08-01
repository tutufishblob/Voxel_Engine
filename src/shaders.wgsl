
struct Uniforms {
  mvp: mat4x4<f32>
}

struct VertexInput {
  @location(0) position: vec3f,
  @location(1) color: vec3f,
  @builtin(vertex_index) vertex_index: u32,
}

struct VertexOutput {
  @builtin(position) position:vec4f,
  @location(0) color:vec3f,
}

struct WireframeVertexInput {
  @location(0) position:vec3f,
}

struct CrosshairVertexInput {
  @location(0) position:vec2f,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;


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
  out.position = uniforms.mvp * vec4f(in.position, 1.0);
  out.color = in.color;
  return out;
}

@fragment 
fn display_fs(@location(0) color: vec3f) -> @location(0) vec4f {
  return vec4f(color, 1.0);
}


@vertex
fn wireframe_vs(in: WireframeVertexInput) -> @builtin(position) vec4f {
  // let scaled_position = in.position * 1.01; //unnecessary
  // return uniforms.mvp * vec4f(scaled_position, 1.0);
  return uniforms.mvp * vec4f(in.position,1.0);
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