use noise::{core::perlin, Fbm, MultiFractal, NoiseFn, Perlin, Seedable, Simplex, Worley};

use crate::terrain;

enum Biomes {
    Plains, //all other moisture
    Forest, //moisture > .7
    Mountains, //moisture < .3
} 


pub struct TerrainCreator {
    temperature: Perlin,
    moisture: Perlin,
    terrain_generating_function: Fbm<Perlin>,

    moisture_sensitivity: f64,
    temperature_sensitivity:f64,

    current_biome: Biomes,

    moisture_level: f64,
    temperature_level:f64,

    mountain_sensitivity: f64,
    plains_sensitivity: f64,
    forest_sensitivity: f64,

    mountain_height: f64,
    plains_height: f64,
    forest_height: f64,

}


impl TerrainCreator {
    pub fn new() -> Self{
        let moisture = Perlin::new(0);
        let temperature = Perlin::new(0);
        let terrain_func = Fbm::new(1).set_octaves(6).set_persistence(0.6).set_frequency(1.4).set_lacunarity(1.4);

        Self {
            temperature: temperature,
            moisture: moisture,
            terrain_generating_function: terrain_func,

            moisture_sensitivity: 0.004,
            temperature_sensitivity: 0.6,


            current_biome: Biomes::Plains,

            moisture_level: 0.5,
            temperature_level: 0.0,

            mountain_height: 63.0,
            plains_height: 5.0,
            forest_height: 15.0,

            mountain_sensitivity: 0.005,
            forest_sensitivity: 0.003,
            plains_sensitivity: 0.0005
        }
    }

   

    pub fn get_height(&mut self,x:f64,z:f64) -> u16 {
        // self.get_biome(&x,&z);

        // let temp_forest = ((self.terrain_generating_function.get([x *  self.forest_sensitivity,z * self.forest_sensitivity]) + 1.0) *  self.forest_height) as u16;
        // let temp_plains = ((self.terrain_generating_function.get([x *  self.plains_sensitivity,z * self.plains_sensitivity]) + 1.0) *  self.plains_height) as u16;
        // let temp_mountian = ((self.terrain_generating_function.get([x *  self.mountain_sensitivity,z * self.mountain_sensitivity]) + 1.0) *  self.mountain_height) as u16;

        // match self.current_biome {
        //     Biomes::Forest => {
        //         return Self::lerp(temp_forest,temp_plains,  self.moisture_level);
        //     },

        //     Biomes::Mountains => {
        //         return Self::lerp(temp_mountian, temp_forest, 2.0 * (self.moisture_level) );
        //     },

        //     Biomes::Plains => {
        //         return Self::lerp(temp_forest, temp_plains, (self.moisture_level) );
        //     }
                
        // }
        let mut temp_height = (((self.terrain_generating_function.get([x * self.forest_sensitivity,z * self.forest_sensitivity]) + 1.0) * 0.5)  * 400.0) as u16;

        // temp_height = (temp_height as f64).powf(0.9) as u16;
        
        // if temp_height <= 100 {
        //     temp_height /= 8;
        //     temp_height += 90;
        // }
        
        if temp_height <= 100{
            temp_height = (temp_height as f64).powf(0.3) as u16;
            temp_height += 314;
        }
        else if temp_height <= 150{
            temp_height = (temp_height as f64).powf(0.4) as u16;
            temp_height += 312;
        }
        else if temp_height <= 200{
            temp_height = (temp_height as f64).powf(0.6) as u16;
            temp_height += 299;
        }
        else if temp_height <= 300{
            temp_height = (temp_height as f64).powf(0.8) as u16;
            temp_height += 254;
        }
        else if temp_height <= 400{
            temp_height = (temp_height as f64).powf(0.9) as u16;
            temp_height += 180;
        }
        else {
            //temp_height = (temp_height as f64).powf(0.9) as u16;
        }
        

        temp_height as u16
        


    }

    fn lerp(a: u16, b: u16, weight: f64) -> u16 {
        (a as i32  + (weight * (b as i32 - a as i32 ) as f64)  as i32) as u16
    }

    fn get_biome(&mut self,x:&f64,z:&f64) {
        self.moisture_level = (self.moisture.get([*x * self.moisture_sensitivity,*z * self.moisture_sensitivity]) + 1.0) * 0.5;
        //eprintln!("{}",self.moisture_level);
        if self.moisture_level < 0.5 {
            self.current_biome = Biomes::Plains;
        }
        else if self.moisture_level >= 0.5 {
            self.current_biome =  Biomes::Mountains;
        }
        

        
    }
}
// pub fn generate_noise(x:f64, z:f64, base_noise: Perlin, moisture_noise: Simplex, temperature_noise: Worley) -> f64{

    


//     let mut amplitude = 1.0;
//     let mut base_frequency = 0.008;
//     let mut moist_frequency = 0.001;
//     let mut temp_frequency = 0.0005;
//     let mut noise_height = 0.0;
//     let persistence = 0.5;
    

//     //let basic_noise = base_noise.get([x * base_frequency, z * base_frequency]) + 1.0;
//     let moist_noise = moisture_noise.get([x * moist_frequency, z * moist_frequency]) + 1.0;
//     let temp_noise = temperature_noise.get([x * temp_frequency, z * temp_frequency]) + 1.0;

//     let octaves = (5.0) as i32;

//     for _ in 0..octaves {
//         noise_height += amplitude * base_noise.get([x * base_frequency, z * base_frequency]) + 1.0;
//         amplitude *= persistence;
//         base_frequency *= moist_noise;
//     }
//     //eprint!("{}\n",noise_height);
//     lerp(noise_height,moist_noise, temp_noise)
// }

