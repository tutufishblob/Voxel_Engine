use glam::Vec2;
use glam::Vec3;


use crate::types::AABB;
use crate::MOVESPEED;
use crate::types;


fn is_valid_voxel(x:u16, y:u16, z: u16 , height_map: &Vec<Vec<u16>>) -> bool{
    if height_map[x as usize][z as usize] >= y{
        return true;
    }
    false
}   

pub fn move_player(positon: Vec3, horizontal_velocity: Vec2, vertical_velocity: f32, height_map: &Vec<Vec<u16>>)-> Vec3{

    //let aabb = AABB::compute_aabb(new_pos);

    //Println!("move_player horizontal{}",horizontal_velocity);

    let total_velocity = Vec3::new(horizontal_velocity.x, vertical_velocity, horizontal_velocity.y);

    //Println!("move_player pos{}",positon);
    //Println!("move_player velocity{}",total_velocity);

    let mut new_pos = positon + total_velocity;

    //Println!("pre new pos{}",new_pos);

    let old_pos_voxel = positon.floor();
    let new_pos_voxel = new_pos.floor();    

    //Println!("mid pos{}",new_pos);

    if old_pos_voxel != new_pos_voxel{
        if height_map[new_pos_voxel.x as usize][new_pos_voxel.z as usize] + 1 > (new_pos_voxel.y as u16 - 1)  {
            if new_pos_voxel.x != old_pos_voxel.x {
                new_pos.x = positon.x;
            }
            if new_pos_voxel.z != old_pos_voxel.z {
                new_pos.z = positon.z;
            }
        }
    }

    new_pos.y = new_pos.y.clamp((height_map[new_pos.x as usize][new_pos.z as usize] + 2) as f32, 999.0);


    //Println!("end new pos{}",new_pos);
    new_pos
}



