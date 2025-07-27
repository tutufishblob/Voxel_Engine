use glam::Vec3;

const PLAYERSIZE: Vec3 = Vec3 {x: 0.6,y: 1.8,z: 0.6};

pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

pub struct Player {
    bounding_box: AABB,

}

impl AABB {
    pub fn compute_aabb (position: Vec3) -> Self {
        AABB {
            min: Vec3::new( 
                position.x - (0.5 * PLAYERSIZE.x),
                position.y,
                position.z - (0.5 * PLAYERSIZE.z)),
            max: Vec3::new( 
                position.x + (0.5 * PLAYERSIZE.x),
                position.y + (PLAYERSIZE.y),
                position.z + (0.5 * PLAYERSIZE.z)),
        }
    } 

}