pub mod activator;
pub mod cross;
pub mod flood;
pub mod loss;
pub mod model;
pub mod utills;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    //flood::flood_8_4_1(0.01, 0.01, "flood-8-4-1_temp")?;
    //cross::cross_2_4_2(0.001, 0.001, "cross")?;

    /* 
    let mut net = utills::io::load("models/cross/6.json")?;
    println!("{:?}", net.forward(vec![0.0902, 0.2690])[0] > 0.5);
    println!("{:?}", net.forward(vec![0.2962, 0.0697])[0] > 0.5);
    */

    let mut net = utills::io::load("models/flood-8-4-1_temp/6.json")?;
    println!("{:?}", net.forward(vec![95f64 ,95f64, 95f64, 95f64, 148f64, 149f64, 150f64, 150f64])[0]);

    Ok(())
}
