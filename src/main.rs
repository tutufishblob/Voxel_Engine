use {
    crate::{camera::Camera, render::{generate_and_write_terrain, Rasterizer}}, anyhow::{Context, Result}, glam::{Mat4,Vec3}, noise::{NoiseFn, Perlin}, std::{collections::HashSet,
        time::{Duration,Instant}}, wgpu::TextureView, winit::{
        dpi::{LogicalPosition, PhysicalPosition, Position},
        event::{self, DeviceEvent, ElementState, Event, KeyEvent, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        keyboard::{KeyCode, PhysicalKey},
        window::{Window, WindowBuilder},
        }
};

mod render;
mod camera;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;


const UPDATE_INTERVAL: Duration = Duration::from_nanos(8_333_333);
const FRAME_INTERVAL: Duration = Duration::from_micros(6944);
const MOVESPEED: f32 = 0.15;
const SENSITIVITY: f64 = 0.1;
const CENTER: LogicalPosition<u32> = LogicalPosition{x: WIDTH/2 as u32, y: HEIGHT/2 as u32};
const GRAVITY: f32 = -9.81;


#[pollster::main]
async fn main() -> Result<()> {
    


    let mut current_view = Camera::new(Vec3::new(0.0, 2.0, 0.0),0.0,90.0);

    let mut pressed_key = HashSet::new();

    //let update_interval = Duration::from_millis(100);
    let mut last_update = Instant::now();
    let mut last_frame = Instant::now();

    let mut previous_cursor_position:Option<LogicalPosition<f64>> = None;

    let event_loop = EventLoop::new()?;
    let window_size = winit::dpi::PhysicalSize::new(WIDTH, HEIGHT);
    let window = WindowBuilder::new()
        .with_inner_size(window_size)
        .with_resizable(true)
        .with_title("Voxel Window".to_string())
        .build(&event_loop)?;
    let (device, queue, surface) = connect_to_gpu(&window).await?;
    let mut renderer = render::Rasterizer::new(device, queue, WIDTH, HEIGHT, current_view);
    let display_buffers = generate_and_write_terrain(&renderer);


    _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
    window.set_cursor_visible(false);
    
    let scale_factor = window.scale_factor();

    event_loop.run(|event, control_handle| {
        control_handle.set_control_flow(ControlFlow::Poll);
        match event {
            Event::AboutToWait =>{
                let now = Instant::now();

                if now - last_update >= UPDATE_INTERVAL {
                    update_position(&mut renderer, &mut current_view, &pressed_key);
                    
                    last_update = now;
                    
                }     
                if now - last_frame >= FRAME_INTERVAL{
                    window.request_redraw();
                    last_frame = now;
                }  
                         
            },
            Event::DeviceEvent {event ,..} => {
                if let DeviceEvent::MouseMotion { delta } = event{
                    update_aim(&mut renderer, &mut current_view, delta);
                }
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => control_handle.exit(),  

                WindowEvent::KeyboardInput {event,.. } => {     
                    if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                        control_handle.exit();
                    }      
                    if !event.repeat {
                        match event.state {
                            ElementState::Pressed => {
                                pressed_key.insert(event.physical_key);
                            },
                            ElementState::Released => {
                                pressed_key.remove(&event.physical_key);
                            }
                        }
                        
                    }
                },
                // WindowEvent::CursorMoved { position,.. } => {
                //         let mut logical_position:LogicalPosition<f64> = position.to_logical(scale_factor);
                //         update_aim(&mut renderer,&window, &mut current_view, &mut logical_position, &mut previous_cursor_position);
                //         _ = window.set_cursor_position(CENTER);
                    
                    
                    
                // }
                WindowEvent::RedrawRequested => {
                    // Wait for the next available frame buffer.
                    let frame: wgpu::SurfaceTexture = surface
                        .get_current_texture()
                        .expect("failed to get current texture");

                    let render_target = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                    

                    renderer.render_frame(&render_target, &display_buffers);

                    frame.present();
                    
                }
                _ => (),
                
            },
            _ => (),
        }
    })?;
    Ok(())
}

async fn connect_to_gpu(window: &Window) -> Result<(wgpu::Device, wgpu::Queue, wgpu::Surface)> {
    use wgpu::TextureFormat::{Bgra8Unorm, Rgba8Unorm};

    // Create an "instance" of wgpu. This is the entry-point to the API.
    let instance = wgpu::Instance::default();

    // Create a drawable "surface" that is associated with the window.
    let surface = instance.create_surface(window)?;

    // Request a GPU that is compatible with the surface. If the system has multiple GPUs then
    // pick the high performance one.
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .context("failed to find a compatible adapter")?;

    // Connect to the GPU. "device" represents the connection to the GPU and allows us to create
    // resources like buffers, textures, and pipelines. "queue" represents the command queue that
    // we use to submit commands to the GPU.
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .context("failed to connect to the GPU")?;

    // Configure the texture memory backs the surface. Our renderer will draw to a surface texture
    // every frame.
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .into_iter()
        .find(|it| matches!(it, Rgba8Unorm | Bgra8Unorm))
        .context("could not find preferred texture format (Rgba8Unorm or Bgra8Unorm)")?;
    let size = window.inner_size();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 3,
    };
    surface.configure(&device, &config);

    Ok((device, queue, surface))
}

pub fn update_position(renderer: &mut Rasterizer, current_view: &mut Camera, pressed_keys: &HashSet<PhysicalKey>) {
   
    //code for making cam spin, used early in dev...
    // {let angle = std::f32::consts::PI / 64.0; // 45 degrees 
    // let y_rotation = glam::Mat3::from_rotation_y(angle);
    
    // current_view.eye = y_rotation * current_view.eye;}
    let mut adjusted_movespeed = MOVESPEED;
    if pressed_keys.contains(&PhysicalKey::Code(KeyCode::ShiftLeft)) {
        adjusted_movespeed *= 2.0;
    }

    if pressed_keys.contains(&PhysicalKey::Code(KeyCode::KeyW)) {
        current_view.position = current_view.position + (adjusted_movespeed * (current_view.direction * Vec3::new(1.0,0.0,1.0)));
    }
    if pressed_keys.contains(&PhysicalKey::Code(KeyCode::KeyS)) {
        current_view.position =  current_view.position + (adjusted_movespeed * current_view.direction * Vec3::new(-1.0,0.0,-1.0));
    }
    if pressed_keys.contains(&PhysicalKey::Code(KeyCode::KeyA)) {
        current_view.position = current_view.position + (adjusted_movespeed * (current_view.direction.cross(Vec3::Y)) * Vec3::new(-1.0,0.0,-1.0));
        
        
    }
    if pressed_keys.contains(&PhysicalKey::Code(KeyCode::KeyD)) {
        current_view.position = current_view.position + (adjusted_movespeed * (current_view.direction.cross(Vec3::Y)) * Vec3::new(1.0,0.0,1.0));
    }


    if pressed_keys.contains(&PhysicalKey::Code(KeyCode::Space)) {
        current_view.position = current_view.position + (MOVESPEED * (Vec3::new(0.0,1.0,0.0)));
    }
    if pressed_keys.contains(&PhysicalKey::Code(KeyCode::ControlLeft)) {
        current_view.position = current_view.position + (MOVESPEED * (Vec3::new(0.0,-1.0,0.0)));
    }

    renderer.camera.view = current_view.view_matrix();

    renderer.uniforms.mvp = renderer.camera.projection * renderer.camera.view * renderer.camera.model;

}

pub fn update_aim(renderer: &mut Rasterizer, current_view: &mut Camera, delta: (f64,f64)) {



        let yaw_delta = delta.0 * SENSITIVITY;
        let pitch_delta = delta.1 * SENSITIVITY;

        current_view.yaw += yaw_delta;
        current_view.pitch -= pitch_delta;

        
        current_view.pitch = current_view.pitch.clamp(-89.0, 89.0);

        current_view.update_aim();
        
    

    
    //code for making cam spin, used early in dev...
    // {let angle = std::f32::consts::PI / 64.0; // 45 degrees 
    // let y_rotation = glam::Mat3::from_rotation_y(angle);
    
    // current_view.eye = y_rotation * current_view.eye;}
    
    




    renderer.camera.view = current_view.view_matrix();

    renderer.uniforms.mvp = renderer.camera.projection * renderer.camera.view * renderer.camera.model;

}