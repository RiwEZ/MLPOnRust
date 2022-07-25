pub mod activator;
pub mod cross;
pub mod flood;
pub mod loss;
pub mod model;
pub mod utills;

use std::error::Error;
use utills::data::un_standardization;

fn main() -> Result<(), Box<dyn Error>> {
    //flood::flood_8_4_1(0.01, 0.01, "flood-8-4-1_temp")?;
    //cross::cross_2_4_1(0.01, 0.01, "cross")?;

    /*
    let mut net = utills::io::load("models/cross/6.json")?;
    println!("{:?}", net.forward(vec![0.0902, 0.2690])[0] > 0.5);
    println!("{:?}", net.forward(vec![0.2962, 0.0697])[0] > 0.5);
    */

    /* 
    let dataset = utills::data::flood_dataset()?;
    let mut net = utills::io::load("models/flood-8-4-1_temp/3.json")?;
    
    let mean = dataset.mean();
    let std = dataset.std();
    let st_dt = utills::data::standardization(&dataset, mean, std);

    for data in st_dt.get_datas() {

        let result = un_standardization(net.forward(data.inputs.clone())[0], mean, std);
        let desired = un_standardization(data.labels[0], mean, std);
        println!("desired: {}, result: {:.3}, diff: {:.3}", desired, result, (desired-result).abs());
    }
    */
    
    Ok(())
}
