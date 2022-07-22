pub mod activator;
pub mod model;
pub mod utills;
pub mod loss;

use std::error::Error;
use model::{Layer, Net};
use utills::data;
use utills::io;

fn lr_finder(net: &mut model::Net, loss: &mut loss::MSELoss, dataset: &data::DataSet) -> Result<(), Box<dyn Error>> {
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

            net.update(lr, 0.01);

            running_loss += loss.item();
        }
        running_loss /= dataset.get_datas().len() as f64;
        println!("epoch: {}, loss: {}", i, running_loss);
        loss_vec.push(running_loss);
        x_vec.push(lr);
        lr += 0.000001;
    }

    //utills::draw_loss(x_vec, loss_vec,"img/lr_finder3.png".to_string())?;
    Ok(())
}


fn main() -> Result<(), Box<dyn Error>> {
    let dataset = data::flood_dataset()?;
    /* 
    let mut loss = loss::MSELoss::new();
    let lr = 0.00001;
    let momentum = 0.01;
    
    let mut j = 0;
    for dt in dataset.cross_valid_set(0.1) {
        /* 
        if j > 0 {
            break;
        }
        */

        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(8, 4, 1.0, activator::sigmoid()));
        layers.push(Layer::new(4, 1, 1.0, activator::linear()));
        
        let mut net = Net::from_layers(layers);

        let training_set = dt.0;
        let validation_set = dt.1;
        
        // lr finder
        //lr_finder(&mut net, &mut loss, &training_set)?;

        // training
        let mut loss_vec: Vec<f64> = vec![];
        let mut valid_loss_vec: Vec<f64> = vec![];
        let mut x_vec: Vec<f64> = vec![];
        for i in 0..2500 {
            let mut running_loss: f64 = 0.0;

            // cyclic learning rate?
            /* 
            let cycle = (1.0 + (i as f64)/(2.0*step_size)).floor();
            let x = (i as f64/step_size - (2.0*cycle) + 1.0).abs();
            let lr = min_lr + ((max_lr - min_lr) * f64::max(0.0, 1.0-x));
             */
            //println!("cycle: {}, x: {}, lr: {}", cycle, x, lr);
            
            for data in training_set.get_shuffled() {
                let result = net.forward(data.inputs.clone());

                loss.criterion(result, data.labels.clone());
                loss.backward(&mut net.layers);

                net.update(lr, momentum);

                running_loss += loss.item();
            }
            x_vec.push(i as f64);

            running_loss /= training_set.get_datas().len() as f64;
            loss_vec.push(running_loss);
            
            let mut valid_loss: f64 = 0.0;
            for data in validation_set.get_datas() {
                let result = net.forward(data.inputs.clone());

                loss.criterion(result, data.labels.clone());
                valid_loss += loss.item();
            }
            valid_loss /= validation_set.get_datas().len() as f64;
            valid_loss_vec.push(valid_loss);
               
            println!("epoch: {}, loss: {:.6}, valid_loss: {:.6}", i, running_loss, valid_loss);
        }
        
        //println!("{:?}", &net);
        utills::draw_loss(x_vec, loss_vec, valid_loss_vec, format!("img/{}_valid_loss.png", j))?;
        io::save(&net.layers, format!("models/flood/{}.json", j))?;
        

        j += 1;
    }
    */

    let mut net = io::load("models/flood/9.json")?;
    let data = dataset.get_datas().clone();
    let mut loss_vec: Vec<f64> = vec![];

    for i in 0..data.len() {
        let result = data::flood_stdsc_rev(net.forward(data[i].inputs.clone())[0]);
        let label = data::flood_stdsc_rev(data[i].labels.clone()[0]);
        let diff = (label - result).powi(2);
        println!("label: {}, result: {}, diff: {}", label, result, diff);
        loss_vec.push(diff);
    }


    Ok(())
}
