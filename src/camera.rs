use glam::{Mat4, Vec3,Vec2};
#[derive(Copy,Clone)]
pub struct Camera{ 
    pub position: Vec3,
    pub yaw: f64,
    pub pitch: f64,
    pub direction: Vec3,
    pub horizontal_velocity: Vec2,
    pub vertical_velocity: f32,
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

        Self{
            position,
            yaw,
            pitch,
            direction: dir.normalize(),
            horizontal_velocity:horizontal_velocity,
            vertical_velocity: vertical_velocity,
        }
    }
}