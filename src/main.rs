pub mod activator;
pub mod utills;
pub mod model;
pub mod io;
pub mod loss;

use std::error::Error;
use model::{Layer, Net};

fn lr_finder(net: &mut model::Net, loss: &mut loss::MSELoss, dataset: &utills::DataSet) -> Result<(), Box<dyn Error>> {
    let mut lr = 0f64;
    let mut loss_vec: Vec<f64> = vec![];
    let mut x_vec = vec![];

    for i in 0..400 {
        let mut running_loss = 0.0;
        
        // 1 batch?
        for data in dataset.get_shuffled() {
            let result = net.forward(data.inputs.clone());

            loss.criterion(result, data.labels.clone());
            loss.backward(&mut net.layers);

            net.update(lr, 0.0);

            running_loss += loss.item();
        }
        println!("epoch: {}, loss: {}", i, running_loss);
        loss_vec.push(running_loss);
        x_vec.push(lr);
        lr += 0.0000001;
    }

    utills::draw_loss(x_vec, loss_vec,"img/lr_finder3.png".to_string())?;
    Ok(())
}


fn main() -> Result<(), Box<dyn Error>> {
    let dataset = utills::flood_dataset()?;

    let mut loss = loss::MSELoss::new();
    let lr = 0.0005;
    let momentum = 0.001;
    
    let mut j = 0;
    for dt in dataset.cross_valid_set(0.1) {
        if j > 0 {
            break;
        }

        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(8, 6, 1.0, activator::sigmoid()));
        layers.push(Layer::new(6, 4, 1.0, activator::sigmoid()));
        layers.push(Layer::new(4, 1, 1.0, activator::linear()));
        
        let mut net = Net::from_layers(layers);

        let training_set = dt.0;
        let validation_set = dt.1;
        
        // lr finder
        //lr_finder(&mut net, &mut loss, &training_set)?;

        // training
        let mut loss_vec: Vec<f64> = vec![];
        let mut x_vec: Vec<f64> = vec![];
        for i in 0..5000 {
            let mut running_loss = 0.0;
            
            // 1 batch?
            for data in training_set.get_shuffled() {
                let result = net.forward(data.inputs.clone());

                loss.criterion(result, data.labels.clone());
                loss.backward(&mut net.layers);

                net.update(lr, momentum);

                running_loss += loss.item();
            }
            x_vec.push(i as f64);
            loss_vec.push(running_loss);
            println!("epoch: {}, loss: {}", i, running_loss);
        }

        //println!("{:?}", &net);

        utills::draw_loss(x_vec, loss_vec, format!("img/{}.png", j))?;
        io::save(&net.layers, "models/flood1.json".to_string())?;
        j += 1;
    }
    Ok(())
}
