pub mod activator;
pub mod cross;
pub mod flood;
pub mod loss;
pub mod model;
pub mod utills;

use std::error::Error;
use utills::data::un_standardization;

fn main() -> Result<(), Box<dyn Error>> {
    // training code

    //flood::flood_8_4_1(0.01, 0.01, "flood-8-4-1", true)?; // 1
    //flood::flood_8_4_1(0.01, 0.0, "flood-8-4-1_2", true)?; // 2
    //flood::flood_8_4_1(0.0001, 0.0, "flood-8-4-1_3", true)?; // 3
    //flood::flood_8_4_1(0.01, 0.01, "flood-8-4-1_4", false)?; // 4
    //flood::flood_8_8_1(0.01, 0.01, "flood-8-8-1")?;
    //cross::cross_2_4_1(0.01, 0.01, "cross-2-4-1")?;
    //cross::cross_2_4_1(0.01, 0.0, "cross-2-4-1_2")?;
    //cross::cross_2_4_1(0.0001, 0.01, "cross-2-4-1_3")?;
    //cross::cross_2_8_1(0.01, 0.01, "cross-2-8-1")?;

    Ok(())
}

// temp code
    /*
    let dataset = utills::data::flood_dataset()?;
    let mut net = utills::io::load("models/flood-8-4-1/3.json")?;

    let mean = dataset.mean();
    let std = dataset.std();
    let st_dt = utills::data::standardization(&dataset, mean, std);

    let mut loss_mean = 0.0;
    for data in st_dt.get_datas() {
        let result = un_standardization(net.forward(&data.inputs)[0], mean, std);
        let desired = un_standardization(data.labels[0], mean, std);
        println!("desired: {}, result: {:.3}, diff: {:.3}", desired, result, (desired-result).abs());
        loss_mean += (result-desired).powi(2);
    }
    println!("standardization: {}", (loss_mean/st_dt.len() as f64).sqrt());

    let mut net = utills::io::load("models/flood-8-4-1/3.json")?;
    let min = dataset.min();
    let max = dataset.max();
    let st_dt = utills::data::minmax_norm(&dataset, min, max);

    let mut loss_mean = 0.0;
    for data in st_dt.get_datas() {
        let result = net.forward(&data.inputs)[0] * (max - min) + min;
        let desired = data.labels[0] * (max - min) + min;
        //println!("desired: {}, result: {:.3}, diff: {:.3}", desired, result, (desired-result).abs());
        loss_mean += (result-desired).abs();
    }
    println!("min-max: {}", loss_mean);
    */