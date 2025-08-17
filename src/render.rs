use core::f32;
use std::collections::HashMap;

use bytemuck::{bytes_of, checked::cast_slice, Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use wgpu::{core::device::queue, util::DeviceExt, Buffer};
use crate::ChunkMetaData;
use crate::{camera::Camera};
use crate::textures::load_texture;
use noise::{core::perlin, NoiseFn, Perlin, Simplex, Worley};
use crate::terrain::{self, TerrainCreator};

pub struct BufferStorage {
    cube_vertex_buffer: Buffer,

    pub voxel_instance_buffer: Buffer,
    voxel_index_buffer: Buffer,
    voxel_index_length: usize,

    wireframe_vertex_buffer: Buffer,

    wireframe_instance_buffer: Buffer,
    wireframe_index_buffer: Buffer,
    //wireframe_index_length: usize,

    crosshair_vertex_buffer: Buffer,
    crosshair_index_buffer: Buffer,
}

pub struct Rasterizer{
    device: wgpu::Device,
    pub queue: wgpu::Queue,

    aspect_ratio: f32,

    pub camera: MvpMakeup,

    pub uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,


    display_pipeline: wgpu::RenderPipeline,
    display_bind_group: wgpu::BindGroup,

    wireframe_bind_group: Option<wgpu::BindGroup>,
    wireframe_pipeline: Option<wgpu::RenderPipeline>,

    crosshair_pipeline: wgpu::RenderPipeline,

    texture: wgpu::Texture,
    sampler: wgpu::Sampler,
    texture_bind_group: wgpu::BindGroup,



    // display_buffers: BufferStorage<'a>,
}



#[derive(Copy,Clone,Pod,Zeroable)]
#[repr(C)]
pub struct Uniforms {
    pub mvp: glam::Mat4
}

pub struct MvpMakeup {//change architecture to seperate the matrix set vectors from the matrices themselves
    pub model: Mat4,
    pub view: Mat4,
    pub projection :Mat4,
}


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: Vec3,
    pub tex_coords: Vec2,
}
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    position: Vec3
}

const MAPWIDTH: usize = 2000;
const MAPLENGTH: usize = 2000;
const LARGEPERLINSCALE: f64 = 0.03; //controls the smoothness 
const MEDIUMPERLINSCALE: f64 = 0.008; //controls the smoothness 
const SMALLPERLINSCALE: f64 = 0.001; //controls the smoothness 
const MAPMAXHEIGHT: f64 = 20.0;

const WIDTH:u32 = 1920;
const HEIGHT:u32 = 1080;

const ASPECTRATIO: f32 = WIDTH as f32 / HEIGHT as f32;


// const CUBE_VERTICES: [Vertex; 24] = [
//     // Front face (red)
//     Vertex { position: Vec3::new(0.0, 0.0, 1.0), color: Vec3::new(1.0, 0.0, 0.0) },
//     Vertex { position: Vec3::new(1.0, 0.0, 1.0), color: Vec3::new(1.0, 0.0, 0.0) },
//     Vertex { position: Vec3::new(1.0, 1.0, 1.0), color: Vec3::new(1.0, 0.0, 0.0) },
//     Vertex { position: Vec3::new(0.0, 1.0, 1.0), color: Vec3::new(1.0, 0.0, 0.0) },

//     // Back face (green)
//     Vertex { position: Vec3::new(0.0, 0.0, 0.0), color: Vec3::new(0.0, 1.0, 0.0) },
//     Vertex { position: Vec3::new(1.0, 0.0, 0.0), color: Vec3::new(0.0, 1.0, 0.0) },
//     Vertex { position: Vec3::new(1.0, 1.0, 0.0), color: Vec3::new(0.0, 1.0, 0.0) },
//     Vertex { position: Vec3::new(0.0, 1.0, 0.0), color: Vec3::new(0.0, 1.0, 0.0) },

//     // Left face (blue)
//     Vertex { position: Vec3::new(0.0, 0.0, 0.0), color: Vec3::new(0.0, 0.0, 1.0) },
//     Vertex { position: Vec3::new(0.0, 0.0, 1.0), color: Vec3::new(0.0, 0.0, 1.0) },
//     Vertex { position: Vec3::new(0.0, 1.0, 1.0), color: Vec3::new(0.0, 0.0, 1.0) },
//     Vertex { position: Vec3::new(0.0, 1.0, 0.0), color: Vec3::new(0.0, 0.0, 1.0) },

//     // Right face (yellow)
//     Vertex { position: Vec3::new(1.0, 0.0, 0.0), color: Vec3::new(1.0, 1.0, 0.0) },
//     Vertex { position: Vec3::new(1.0, 0.0, 1.0), color: Vec3::new(1.0, 1.0, 0.0) },
//     Vertex { position: Vec3::new(1.0, 1.0, 1.0), color: Vec3::new(1.0, 1.0, 0.0) },
//     Vertex { position: Vec3::new(1.0, 1.0, 0.0), color: Vec3::new(1.0, 1.0, 0.0) },

//     // Top face (magenta)
//     Vertex { position: Vec3::new(0.0, 1.0, 1.0), color: Vec3::new(1.0, 0.0, 1.0) },
//     Vertex { position: Vec3::new(1.0, 1.0, 1.0), color: Vec3::new(1.0, 0.0, 1.0) },
//     Vertex { position: Vec3::new(1.0, 1.0, 0.0), color: Vec3::new(1.0, 0.0, 1.0) },
//     Vertex { position: Vec3::new(0.0, 1.0, 0.0), color: Vec3::new(1.0, 0.0, 1.0) },

//     // Bottom face (cyan)
//     Vertex { position: Vec3::new(0.0, 0.0, 1.0), color: Vec3::new(0.0, 1.0, 1.0) },
//     Vertex { position: Vec3::new(1.0, 0.0, 1.0), color: Vec3::new(0.0, 1.0, 1.0) },
//     Vertex { position: Vec3::new(1.0, 0.0, 0.0), color: Vec3::new(0.0, 1.0, 1.0) },
//     Vertex { position: Vec3::new(0.0, 0.0, 0.0), color: Vec3::new(0.0, 1.0, 1.0) },
// ];

const CUBE_VERTICES: &[Vertex] = &[
    // Front face (face_index = 0)
    Vertex { position: Vec3::new(0.0, 0.0, 1.0), tex_coords: Vec2::new(0.0 / 6.0, 1.0) }, // bottom-left
    Vertex { position: Vec3::new(1.0, 0.0, 1.0), tex_coords: Vec2::new(1.0 / 6.0, 1.0) }, // bottom-right
    Vertex { position: Vec3::new(1.0, 1.0, 1.0), tex_coords: Vec2::new(1.0 / 6.0, 0.0) }, // top-right
    Vertex { position: Vec3::new(0.0, 1.0, 1.0), tex_coords: Vec2::new(0.0 / 6.0, 0.0) }, // top-left

    // Back face (face_index = 1)
    Vertex { position: Vec3::new(1.0, 0.0, 0.0), tex_coords: Vec2::new(1.0 / 6.0, 1.0) }, // bottom-left
    Vertex { position: Vec3::new(0.0, 0.0, 0.0), tex_coords: Vec2::new(2.0 / 6.0, 1.0) }, // bottom-right
    Vertex { position: Vec3::new(0.0, 1.0, 0.0), tex_coords: Vec2::new(2.0 / 6.0, 0.0) }, // top-right
    Vertex { position: Vec3::new(1.0, 1.0, 0.0), tex_coords: Vec2::new(1.0 / 6.0, 0.0) }, // top-left

    // Left face (face_index = 2)
    Vertex { position: Vec3::new(0.0, 0.0, 0.0), tex_coords: Vec2::new(2.0 / 6.0, 1.0) },
    Vertex { position: Vec3::new(0.0, 0.0, 1.0), tex_coords: Vec2::new(3.0 / 6.0, 1.0) },
    Vertex { position: Vec3::new(0.0, 1.0, 1.0), tex_coords: Vec2::new(3.0 / 6.0, 0.0) },
    Vertex { position: Vec3::new(0.0, 1.0, 0.0), tex_coords: Vec2::new(2.0 / 6.0, 0.0) },

    // Right face (face_index = 3)
    Vertex { position: Vec3::new(1.0, 0.0, 1.0), tex_coords: Vec2::new(3.0 / 6.0, 1.0) },
    Vertex { position: Vec3::new(1.0, 0.0, 0.0), tex_coords: Vec2::new(4.0 / 6.0, 1.0) },
    Vertex { position: Vec3::new(1.0, 1.0, 0.0), tex_coords: Vec2::new(4.0 / 6.0, 0.0) },
    Vertex { position: Vec3::new(1.0, 1.0, 1.0), tex_coords: Vec2::new(3.0 / 6.0, 0.0) },

    // Top face (face_index = 4)
    Vertex { position: Vec3::new(0.0, 1.0, 1.0), tex_coords: Vec2::new(4.0 / 6.0, 1.0) },
    Vertex { position: Vec3::new(1.0, 1.0, 1.0), tex_coords: Vec2::new(5.0 / 6.0, 1.0) },
    Vertex { position: Vec3::new(1.0, 1.0, 0.0), tex_coords: Vec2::new(5.0 / 6.0, 0.0) },
    Vertex { position: Vec3::new(0.0, 1.0, 0.0), tex_coords: Vec2::new(4.0 / 6.0, 0.0) },

    // Bottom face (face_index = 5)
    Vertex { position: Vec3::new(0.0, 0.0, 0.0), tex_coords: Vec2::new(5.0 / 6.0, 1.0) },
    Vertex { position: Vec3::new(1.0, 0.0, 0.0), tex_coords: Vec2::new(6.0 / 6.0, 1.0) },
    Vertex { position: Vec3::new(1.0, 0.0, 1.0), tex_coords: Vec2::new(6.0 / 6.0, 0.0) },
    Vertex { position: Vec3::new(0.0, 0.0, 1.0), tex_coords: Vec2::new(5.0 / 6.0, 0.0) },
];


const CUBE_INDICES: [u32; 36] = [
    0, 1, 2, 0, 2, 3,        // front (ok)
    4, 5, 6, 4, 6, 7,        // back (flipped winding)
    8, 9, 10, 8, 10, 11,     // left (check winding)
    12, 13, 14, 12, 14, 15,  // right (check winding)
    16, 17, 18, 16, 18, 19,  // top (check winding)
    20, 21, 22, 20, 22, 23,  // bottom (check winding)
];

const WIREFRAME_CORNERS: [Vec3; 8] = [
    Vec3::new(0.0, 0.0, 0.0), // 0
    Vec3::new(1.0, 0.0, 0.0), // 1
    Vec3::new(1.0, 1.0, 0.0), // 2
    Vec3::new(0.0, 1.0, 0.0), // 3
    Vec3::new(0.0, 0.0, 1.0), // 4
    Vec3::new(1.0, 0.0, 1.0), // 5
    Vec3::new(1.0, 1.0, 1.0), // 6
    Vec3::new(0.0, 1.0, 1.0), // 7
];

const WIREFRAME_INDICES: [u32; 24] = [
    // front face
    0, 1, 1, 2, 
    2, 3, 3, 0,

    4, 5, 5, 6, 
    6, 7, 7, 4,

    
    // back face
    
    // sides
    0, 4, 1, 5, 
    2, 6, 3, 7,
];

const CHUNK_COORDS: [Vec3; 16] = [
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(2.0, 0.0, 0.0),
    Vec3::new(3.0, 0.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(1.0, 0.0, 1.0),
    Vec3::new(2.0, 0.0, 1.0),
    Vec3::new(3.0, 0.0, 1.0),
    Vec3::new(0.0, 0.0, 2.0),
    Vec3::new(1.0, 0.0, 2.0),
    Vec3::new(2.0, 0.0, 2.0),
    Vec3::new(3.0, 0.0, 2.0),
    Vec3::new(0.0, 0.0, 3.0),
    Vec3::new(1.0, 0.0, 3.0),
    Vec3::new(2.0, 0.0, 3.0),
    Vec3::new(3.0, 0.0, 3.0),
];

const CROSSHAIR_VERTICES: [Vec2; 8] = [
    Vec2::new(-0.05 / ASPECTRATIO,  0.01), // Top-left
    Vec2::new( 0.05 / ASPECTRATIO,  0.01), // Top-right
    Vec2::new( 0.05 / ASPECTRATIO, -0.01), // Bottom-right
    Vec2::new(-0.05 / ASPECTRATIO, -0.01), // Bottom-left

    // Vertical bar (top to bottom, left then right)
    Vec2::new(-0.01 / ASPECTRATIO,  0.05 ), // Top-left
    Vec2::new( 0.01 / ASPECTRATIO,  0.05 ), // Top-right
    Vec2::new( 0.01 / ASPECTRATIO, -0.05 ), // Bottom-right
    Vec2::new(-0.01 / ASPECTRATIO, -0.05 ), // Bottom-left
];

const CROSSHAIR_INDICDES: [u16; 12] = [
    0, 1, 2,
    2, 3, 0,

    // Second rectangle (vertical bar)
    4, 5, 6,
    6, 7, 4,
];

pub fn generate_and_write_terrain(renderer: &Rasterizer) -> (BufferStorage, Vec<Vec<u16>>){
        let mut terrain_creator = TerrainCreator::new();
        // let limits = renderer.device.limits();
        // println!("Max buffer size: {}", limits.max_buffer_size);

        let mut voxel_vertex_input: Vec<Instance> = vec![];
        let mut wireframe_vertex_input: Vec<Vec3> = vec![];

        //let mut voxel_index_input: Vec<u32> = vec![];
        //let mut wireframe_index_input: Vec<u32> = vec![];

        let mut height_map: Vec<Vec<u16>> = vec![vec![0;MAPWIDTH];MAPLENGTH];




        for x in 0..MAPLENGTH {
            for z in 0..MAPWIDTH {
                let height = terrain_creator.get_height(x as f64,z as f64);

                if x <= 15 && z <= 15{
                    println!("old [{},{},{}] ",x,height,z);
                }
                
                height_map[x][z] = height as u16;


            }
        }


        for (i,rows) in height_map.iter().enumerate(){
            for (j, height) in rows.iter().enumerate(){
                
                let temp_vertex = Instance{
                    position: Vec3::new(i as f32, *height as f32, j as f32),
                };
                voxel_vertex_input.push(temp_vertex);

                wireframe_vertex_input.push(Vec3::new(i as f32, *height as f32, j as f32));
                
                
            }
        }

        

        let cube_vertex_buffer = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Vertex Buffer"),
            contents: bytemuck::cast_slice(&CUBE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let voxel_instance_buffer = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cube Position Vertex Buffer"),
            size: (100000 * std::mem::size_of::<u16>()) as u64,
            mapped_at_creation: false,
            //contents: bytemuck::cast_slice(&voxel_vertex_input),
            //contents: &[],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let voxel_index_buffer = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&CUBE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });


        let wireframe_vertex_buffer = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Wireframe Vertex Buffer"),
            contents: bytemuck::cast_slice(&WIREFRAME_CORNERS),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let wireframe_instance_buffer = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Wireframe Vertex Buffer"),
            contents: bytemuck::cast_slice(&wireframe_vertex_input),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let wireframe_index_buffer = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Wireframe Index Buffer"),
            contents: bytemuck::cast_slice(&WIREFRAME_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });


        let crosshair_vertex_buffer = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Vertex Buffer"),
            contents: bytemuck::cast_slice(&CROSSHAIR_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let crosshair_index_buffer = renderer.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&CROSSHAIR_INDICDES),
            usage: wgpu::BufferUsages::INDEX,
        });

        


        (
            BufferStorage {

                cube_vertex_buffer:cube_vertex_buffer,

                voxel_instance_buffer: voxel_instance_buffer,
                voxel_index_buffer: voxel_index_buffer,
                voxel_index_length: voxel_vertex_input.len(),

                wireframe_vertex_buffer: wireframe_vertex_buffer,
                wireframe_instance_buffer: wireframe_instance_buffer,
                wireframe_index_buffer: wireframe_index_buffer,
                //wireframe_index_length: wireframe_index_input.len(),

                crosshair_vertex_buffer: crosshair_vertex_buffer,
                crosshair_index_buffer: crosshair_index_buffer,
            },
             height_map
        )

    }


impl Rasterizer {

    pub fn render_frame(&self, target: &wgpu::TextureView, display_buffers: &BufferStorage, chunk_pool: &mut HashMap<(i32,i32), ChunkMetaData>) {
        
        

        //#TODO change this so that the camera has its own buffer and is not in the uniform buffer probably
        self.queue.write_buffer(&self.uniform_buffer, 0, bytes_of(&self.uniforms));//updates uniform buffer with new camera angle

        let depth_texture = &self.device.create_texture(&wgpu::TextureDescriptor {//handles z buffer depth testing as to avoid overwriting faces
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render frame"),
            });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("display pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0), // farthest
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            //..Default::default()
            //depth_stencil_attachment: None, //for testing
            ..Default::default()
        });


        
        render_pass.set_pipeline(&self.display_pipeline);
       
        render_pass.set_bind_group(0, &self.display_bind_group, &[]);
        render_pass.set_bind_group(1, &self.texture_bind_group, &[]);

        

        render_pass.set_vertex_buffer(0, display_buffers.cube_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, display_buffers.voxel_instance_buffer.slice(..));
        render_pass.set_index_buffer(display_buffers.voxel_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        for (offsets,chunks) in chunk_pool{
            //eprintln!("{}",chunks.instance_offset);
            render_pass.draw_indexed(0..36 as u32, 0,chunks.instance_offset..chunks.instance_offset + chunks.instance_count);
            //render_pass.draw_indexed(0..36 as u32, 0,0..512);// for testing
        }
       

        match (self.wireframe_pipeline.as_ref(), self.wireframe_bind_group.as_ref()) {
            
            (Some(wireframe_pipeline), Some(wireframe_bind_group)) => {
                //println!("working");
                render_pass.set_pipeline(wireframe_pipeline);
                render_pass.set_bind_group(0, wireframe_bind_group, &[]);

                render_pass.set_vertex_buffer(0, display_buffers.wireframe_vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, display_buffers.wireframe_instance_buffer.slice(..));
                render_pass.set_index_buffer(display_buffers.wireframe_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..24 as u32, 0,0..display_buffers.voxel_index_length as u32);
            }
            _ => {

            }
        }
        
        render_pass.set_pipeline(&self.crosshair_pipeline);

        render_pass.set_vertex_buffer(0, display_buffers.crosshair_vertex_buffer.slice(..));
        render_pass.set_index_buffer(display_buffers.crosshair_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        


        render_pass.draw_indexed(0..12 as u32, 0,0..1);

        // End the render pass by consuming the object.
        drop(render_pass);

        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));
    }



    pub fn new(device: wgpu::Device, queue: wgpu::Queue, width: u32, height: u32, current_view: Camera) -> Rasterizer {
        device.on_uncaptured_error(Box::new(|error| {
            panic!("Aborting due to an error: {}", error);
        }));

        let shader_module = compile_shader_module(&device);
        let (display_pipeline, display_layout) = create_display_pipeline(&device, &shader_module);
        let (wireframe_pipeline,wireframe_layout) = create_wireframe_pipeline(&device, &shader_module);
        let crosshair_pipeline = create_crosshair_pipeline(&device, &shader_module);

        let aspect_ratio = width as f32 / height as f32;

        let fov_y = f32::consts::FRAC_PI_2;

        let camera = MvpMakeup { 
            model:Mat4::IDENTITY,
            view: current_view.view_matrix(),
            projection: Mat4::perspective_rh_gl(
                fov_y,
                aspect_ratio,
                0.1,
                5000.0
            )
        };

        let mvp = camera.projection * camera.view * camera.model;

        let uniforms = Uniforms {
            mvp,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("uniforms"),
            // size: std::mem::size_of ::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            contents: bytes_of(&uniforms),
            //mapped_at_creation: true,
        });
        // uniform_buffer.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::bytes_of(&uniforms));

        // uniform_buffer.unmap();

        let display_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &display_layout,
            entries: &[wgpu::BindGroupEntry{
                binding:0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None 
                }),
            }],
        });

        let wireframe_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &wireframe_layout,
            entries: &[wgpu::BindGroupEntry{
                binding:0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None 
                }),
            }],
        });

        let (texture, sampler, texture_bind_group) = load_texture(&device, &queue,"textures/grass_texture.png");


        Rasterizer {
            device,
            queue,
            aspect_ratio,
            camera,
            uniforms,
            uniform_buffer,
            display_pipeline,
            display_bind_group,
            wireframe_bind_group: Some(wireframe_bind_group),
            wireframe_pipeline: Some(wireframe_pipeline),

            crosshair_pipeline,

            texture:texture,
            sampler:sampler,
            texture_bind_group:texture_bind_group,
        }
    }
}

fn compile_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    use std::borrow::Cow;

    let code = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders.wgsl"));
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(code)),
    })
}

fn create_display_pipeline(
    device: &wgpu::Device,
    shader_module: &wgpu::ShaderModule,
) -> (wgpu::RenderPipeline,wgpu::BindGroupLayout) {

    let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label:None,
        entries: &[
            wgpu::BindGroupLayoutEntry{
                binding:0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None
                },
                count: None,

                },
        ],
    });

    

    let cube_vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }, 
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },              
            ],
            
        };

    let instance_vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },            
            ],
            
        };


    let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Texture Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });



    let voxel_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("display"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
            ..Default::default()
        })),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            ..Default::default()
        },
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: "display_vs",
            buffers: &[cube_vertex_buffer_layout, instance_vertex_buffer_layout],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: "display_fs",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        depth_stencil: Some(wgpu::DepthStencilState {// handles depth compare
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less, // like glDepthFunc(GL_LESS)
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    (voxel_pipeline,uniform_bind_group_layout)
}


fn create_wireframe_pipeline(
    device: &wgpu::Device, 
    shader_module: &wgpu::ShaderModule
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    
    let wireframe_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label:None,
        entries: &[
            wgpu::BindGroupLayoutEntry{
                binding:0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None
                },
                count: None,

                },
        ],
    });

    let wireframe_vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vec3>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }            
            ],
            
        };
    
    let wireframe_instance_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vec3>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }            
            ],
            
        };


    let wireframe_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("wireframe"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&wireframe_bind_group_layout],
            ..Default::default()
        })),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: "wireframe_vs",
            buffers: &[wireframe_vertex_buffer_layout,wireframe_instance_buffer_layout],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: "wireframe_fs",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        depth_stencil: Some(wgpu::DepthStencilState {// handles depth compare
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less, // like glDepthFunc(GL_LESS)
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),

        //depth_stencil:None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    (wireframe_pipeline,wireframe_bind_group_layout)
}

fn create_crosshair_pipeline(
    device: &wgpu::Device, 
    shader_module: &wgpu::ShaderModule
) -> (wgpu::RenderPipeline) {
    
    // let crosshair_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
    //     label:None,
    //     entries: &[
    //         wgpu::BindGroupLayoutEntry{
    //             binding:0,
    //             visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
    //             ty: wgpu::BindingType::Buffer {
    //                 ty: wgpu::BufferBindingType::Uniform,
    //                 has_dynamic_offset: false,
    //                 min_binding_size: None
    //             },
    //             count: None,

    //             },
    //     ],
    // });

    let crosshair_vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vec2>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                }            
            ],
            
        };


    let crosshair_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("wireframe"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("crosshair Pipeline Layout"),
            bind_group_layouts: &[],
            ..Default::default()
        })),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: "crosshair_vs",
            buffers: &[crosshair_vertex_buffer_layout],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: "crosshair_fs",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        depth_stencil: Some(wgpu::DepthStencilState {// handles depth compare
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less, // like glDepthFunc(GL_LESS)
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),

        //depth_stencil:None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    (crosshair_pipeline)
}