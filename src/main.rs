pub mod activator;
pub mod flood;
pub mod cross;
pub mod loss;
pub mod model;
pub mod utills;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    flood::flood_8_4_1(0.01, 0.01, "flood-8-4-1_temp")?;
    Ok(())
}
