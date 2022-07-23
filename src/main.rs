pub mod activator;
pub mod loss;
pub mod model;
pub mod utills;

use model::{Layer, Net};
use std::error::Error;
use std::time::{Duration, Instant};
use utills::data;
use utills::graph;
use utills::io;

fn main() {
}

#[test]
fn flood_8_4_2() -> Result<(), Box<dyn Error>> {
    let dataset = data::flood_dataset()?;
    let mut loss = loss::MSELoss::new();
    let lr = 0.01;
    let momentum = 0.001;
    let epochs = 2000;

    let mut cv_score: Vec<f64> = vec![];
    let mut r2_score: Vec<f64> = vec![];
    let mut loss_g = graph::LossGraph::new();
    let start = Instant::now();
    for (j, dt) in dataset.cross_valid_set(0.1).iter().enumerate() {
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
        for i in 0..epochs {
            let mut running_loss: f64 = 0.0;

            for data in training_set.get_shuffled() {
                let result = net.forward(data.inputs.clone());

                running_loss += loss.criterion(result, data.labels.clone());
                loss.backward(&mut net.layers);

                net.update(lr, momentum);
            }
            x_vec.push(i as f64);
            running_loss /= training_set.len() as f64;
            loss_vec.push(running_loss);

            let mut valid_loss: f64 = 0.0;
            for data in validation_set.get_datas() {
                let result = net.forward(data.inputs.clone());
                valid_loss += loss.criterion(result, data.labels.clone());
            }
            valid_loss /= validation_set.get_datas().len() as f64;
            valid_loss_vec.push(valid_loss);

            if i == epochs - 1 {
                // log score
                let label_mean = validation_set.get_datas().iter().fold(0f64, |mean, val| {
                    mean + val.labels[0] / validation_set.len() as f64
                });

                let mut total_sum_sqr = 0f64;
                let mut sum_sqr = 0f64;

                for data in validation_set.get_datas() {
                    let result = net.forward(data.inputs.clone());
                    sum_sqr += (data.labels[0] - result[0]).powi(2);
                    total_sum_sqr += (data.labels[0] - label_mean).powi(2);
                }

                r2_score.push(1.0 - sum_sqr / total_sum_sqr);
                cv_score.push(valid_loss);
            }

            println!(
                "epoch: {}, loss: {:.6}, valid_loss: {:.6}",
                i, running_loss, valid_loss
            );
        }

        loss_g.add_loss(loss_vec, valid_loss_vec);
        io::save(&net.layers, format!("models/flood-8-4-2/{}.json", j))?;
    }
    let duration: Duration = start.elapsed();
    print!(
        "cv_score: {:?}\n r2_score: {:?}\n time used: {:?}",
        cv_score, r2_score, duration
    );

    loss_g.draw("img/flood-8-4-2/loss.png".to_string())?;
    graph::draw_loss_scores(cv_score.clone(), "img/flood-8-4-2/cv_score.png".to_string())?;
    graph::draw_r2_scores(r2_score, "img/flood-8-4-2/r2_score.png".to_string())?;
    Ok(())
}
