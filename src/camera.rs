use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::types::AABB;
#[derive(Copy,Clone)]
pub struct Camera{ 
    pub position: Vec3,
    pub chunk_relative_position: Vec3,
    pub yaw: f64,
    pub pitch: f64,
    pub direction: Vec3,
    pub horizontal_velocity: Vec2,
    pub vertical_velocity: f32,

    pub current_chunk: Vec2,
}


impl Camera {
    pub fn update_aim(&mut self) {
        let yaw = self.yaw;
        let pitch = self.pitch;

        let dir = Vec3::new(
            yaw.to_radians().cos() as f32 * pitch.to_radians().cos() as f32,
            //0.0,
            pitch.to_radians().sin() as f32,
           // 0.0,
            yaw.to_radians().sin() as f32 * pitch.to_radians().cos() as f32,
        );

        self.direction = dir.normalize();
    }   

    pub fn view_matrix(&self) -> Mat4{
        Mat4::look_at_rh(
            self.position,
            self.position + self.direction,
            Vec3::Y
        )
    }

    pub fn new(position: Vec3, pitch: f64, yaw: f64) -> Self {
        let dir = Vec3::new(
            yaw.to_radians().cos() as f32 * pitch.to_radians().cos()as f32 ,
            pitch.to_radians().sin() as f32,
            yaw.to_radians().sin() as f32 * pitch.to_radians().cos()as f32,
        );

        let horizontal_velocity = Vec2::new(0.0, 0.0);
        let vertical_velocity = 0.0;

        let chunk_relative = Vec3::ZERO;

        let current_chunk = Vec2::new((position.x as i32 /16) as f32,(position.z as i32 /16) as f32);
        Self{
            position,
            yaw,
            pitch,
            direction: dir.normalize(),
            horizontal_velocity:horizontal_velocity,
            vertical_velocity: vertical_velocity,

            current_chunk: current_chunk,

            chunk_relative_position: chunk_relative,

             
        }
    }

    pub fn recompute_chunk(&mut self){

        if self.chunk_relative_position.x > 16.0 {
            self.chunk_relative_position.x -= 16.0;
            self.current_chunk.x += 1.0;
        }
        else if self.chunk_relative_position.x < 0.0 {
            self.chunk_relative_position.x += 16.0;
            self.current_chunk.x -= 1.0;
        }

        if self.chunk_relative_position.z > 16.0 {
            self.chunk_relative_position.z -= 16.0;
            self.current_chunk.y += 1.0;
        }
        else if self.chunk_relative_position.z < 0.0 {
            self.chunk_relative_position.z += 16.0;
            self.current_chunk.y -= 1.0;
        }   
        
    }
}
#[derive(Copy,Clone)]
pub struct Frustrum {
    planes: [Vec4;6]
}

impl Frustrum{
    pub fn new(view: &mut Mat4, proj: &mut Mat4) -> Self{
        let m = *view * *proj;

        let mut planes= [
                (m.row(3) + m.row(0)).normalize(),
                (m.row(3) - m.row(0)).normalize(),
                (m.row(3) + m.row(1)).normalize(), 
                (m.row(3) - m.row(1)).normalize(), 
                (m.row(3) + m.row(2)).normalize(), 
                (m.row(3) - m.row(2)).normalize()
            ];

        for p in &mut planes {
            let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            *p /= len;
        }

        Self { 
            planes: planes
        }
    }

    pub fn update(&mut self, view: &mut Mat4, proj: &mut Mat4){
        let m = *proj * *view;

        let mut planes= [
            (m.row(3) + m.row(0)).normalize(),
            (m.row(3) - m.row(0)).normalize(),
            (m.row(3) + m.row(1)).normalize(), 
            (m.row(3) - m.row(1)).normalize(), 
            (m.row(3) + m.row(2)).normalize(), 
            (m.row(3) - m.row(2)).normalize()
        ];

        for p in &mut planes {
            let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            *p /= len;
        }

        self.planes = planes;
        
    }

    pub fn box_in_fov(&self, aabb: &AABB) -> bool{
        for i in 0..6{
            let mut out = 0;

            if self.planes[i].dot(Vec4::new(aabb.min.x, aabb.min.y, aabb.min.z, 1.0)) < 0.0 { out+=1; }
            if self.planes[i].dot(Vec4::new(aabb.max.x, aabb.min.y, aabb.min.z, 1.0)) < 0.0 { out+=1; }
            if self.planes[i].dot(Vec4::new(aabb.min.x, aabb.max.y, aabb.min.z, 1.0)) < 0.0 { out+=1; }
            if self.planes[i].dot(Vec4::new(aabb.min.x, aabb.min.y, aabb.max.z, 1.0)) < 0.0 { out+=1; }
            if self.planes[i].dot(Vec4::new(aabb.max.x, aabb.max.y, aabb.min.z, 1.0)) < 0.0 { out+=1; }
            if self.planes[i].dot(Vec4::new(aabb.min.x, aabb.max.y, aabb.max.z, 1.0)) < 0.0 { out+=1; }
            if self.planes[i].dot(Vec4::new(aabb.max.x, aabb.min.y, aabb.max.z, 1.0)) < 0.0 { out+=1; }
            if self.planes[i].dot(Vec4::new(aabb.max.x, aabb.max.y, aabb.max.z, 1.0)) < 0.0 { out+=1; }

            if out == 8 { return false; }

            //out = 0; for i in 0..8 { out += if self.planes[i]} //maybe deal with later? might increase performance -> https://iquilezles.org/articles/frustumcorrect/
        }   

        true
    }
}