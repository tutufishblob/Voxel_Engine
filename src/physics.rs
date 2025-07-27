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

pub fn move_player(positon: Vec3, direction:Vec3, velocity: f32, height_map: &Vec<Vec<u16>>)-> Vec3{
    let mut new_pos = positon + (direction * velocity);
    //let aabb = AABB::compute_aabb(new_pos);
    let old_pos_voxel = positon.floor();
    let new_pos_voxel = new_pos.floor();    

    if old_pos_voxel != new_pos_voxel{
        if height_map[new_pos_voxel.x as usize][new_pos_voxel.z as usize] + 1 > (new_pos_voxel.y as u16 - 2)  {
            if new_pos_voxel.x != old_pos_voxel.x {
                new_pos.x = positon.x;
            }
            if new_pos_voxel.z != old_pos_voxel.z {
                new_pos.z = positon.z;
            }
        }
    }

    

    eprintln!("{}",new_pos);
    new_pos
}



