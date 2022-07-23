pub mod activator;
pub mod model;
pub mod utills;
pub mod loss;

use std::error::Error;
use model::{Layer, Net};
use utills::data;
use utills::io;
use utills::graph;
use std::time::{Duration, Instant};

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
    let mut loss = loss::MSELoss::new();
    let lr = 0.01;
    let momentum = 0.01;
    
    let mut j = 0;
    let mut cv_score: Vec<f64> = vec![];
    let start = Instant::now();
    for dt in dataset.cross_valid_set(0.1) {
        // creating a model
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(8, 4, 1.0, activator::sigmoid()));
        layers.push(Layer::new(4, 1, 1.0, activator::linear()));
        let mut net = Net::from_layers(layers);

        // get training set and validation set
        let training_set = data::standardization(&dt.0, dt.0.mean(), dt.0.std());
        let validation_set = data::standardization(&dt.1, dt.0.mean(), dt.0.std());
        
        // training
        let mut loss_vec: Vec<f64> = vec![];
        let mut valid_loss_vec: Vec<f64> = vec![];
        let mut x_vec: Vec<f64> = vec![];
        for i in 0..2000 {
            let mut running_loss: f64 = 0.0;

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
        cv_score.push(valid_loss_vec[valid_loss_vec.len() - 1]);

        //println!("{:?}", &net);
        graph::draw_loss(x_vec, loss_vec, valid_loss_vec, format!("img/flood/{}_valid_loss.png", j))?;
        //io::save(&net.layers, format!("models/flood/{}.json", j))?;
        j += 1;
    }
    let duration: Duration = start.elapsed();
    print!("cv_score: {:?}, time used: {:?}", cv_score, duration);
    graph::draw_cv_scores(cv_score.clone(), "img/flood/cv_score.png".to_string())?;
    Ok(())
}
