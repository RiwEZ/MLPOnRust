//! Contains training code for variations of flood dataset models.
use super::activator;
use super::loss;
use super::model;
use super::utills;

use model::{Layer, Net};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::time::{Duration, Instant};
use utills::data;
use utills::graph;
use utills::io;

pub fn flood_8_4_1(lr: f64, momentum: f64, folder: &str) -> Result<(), Box<dyn Error>> {
    fn model() -> Net {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(8, 4, 1.0, activator::sigmoid()));
        layers.push(Layer::new(4, 1, 1.0, activator::linear()));
        Net::from_layers(layers)
    }

    flood_fit(&model, lr, momentum, folder)?;
    Ok(())
}

pub fn flood_8_8_1(lr: f64, momentum: f64, folder: &str) -> Result<(), Box<dyn Error>> {
    fn model() -> Net {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(8, 8, 1.0, activator::sigmoid()));
        layers.push(Layer::new(8, 1, 1.0, activator::linear()));
        Net::from_layers(layers)
    }

    flood_fit(&model, lr, momentum, folder)?;
    Ok(())
}

pub fn flood_fit(
    model: &dyn Fn() -> Net,
    lr: f64,
    momentum: f64,
    folder: &str,
) -> Result<(), Box<dyn Error>> {
    let (models, img) = utills::io::check_dir(folder)?;

    let dataset = data::flood_dataset()?;
    let mut loss = loss::Loss::mse();
    let epochs = 1000;

    let mut cv_valid_loss: Vec<f64> = vec![];
    let mut cv_train_loss: Vec<f64> = vec![];

    let mut r2_score: Vec<f64> = vec![];
    let mut loss_g = graph::LossGraph::new();
    let start = Instant::now();
    for (j, dt) in dataset.cross_valid_set(0.1).iter().enumerate() {
        // creating a model
        let mut net = model();

        // get training set and validation set
        let training_set = data::standardization(&dt.0, dt.0.mean(), dt.0.std());
        let validation_set = data::standardization(&dt.1, dt.0.mean(), dt.0.std());
        //let training_set = data::minmax_norm(&dt.0, dt.0.min(), dt.0.max());
        //let validation_set = data::minmax_norm(&dt.1, dt.1.min(), dt.1.max());

        // training
        let mut loss_vec: Vec<f64> = vec![];
        let mut valid_loss_vec: Vec<f64> = vec![];
        for i in 0..epochs {
            let mut running_loss: f64 = 0.0;

            for data in training_set.get_shuffled() {
                let result = net.forward(&data.inputs);

                running_loss += loss.criterion(&result, &data.labels);
                loss.backward(&mut net.layers);

                net.update(lr, momentum);
            }
            running_loss /= training_set.len() as f64;
            loss_vec.push(running_loss);

            let mut valid_loss: f64 = 0.0;
            for data in validation_set.get_datas() {
                let result = net.forward(&data.inputs);
                valid_loss += loss.criterion(&result, &data.labels);
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
                    let result = net.forward(&data.inputs);
                    sum_sqr += (data.labels[0] - result[0]).powi(2);
                    total_sum_sqr += (data.labels[0] - label_mean).powi(2);
                }

                r2_score.push(1.0 - (sum_sqr / total_sum_sqr));
                cv_valid_loss.push(valid_loss);
                cv_train_loss.push(running_loss);
            }

            println!(
                "iteration: {}, epoch: {}, loss: {:.6}, valid_loss: {:.6}",
                j, i, running_loss, valid_loss
            );
        }

        loss_g.add_loss(loss_vec, valid_loss_vec);
        io::save(&net.layers, format!("{}/{}.json", models, j))?;
    }
    let duration: Duration = start.elapsed();

    let mut file = fs::File::create(format!("{}/result.txt", models))?;
    file.write_all(
        format!(
            "cv_score: {:?}\n\nr2_score: {:?}\n\ntime used: {:?}",
            cv_valid_loss, r2_score, duration
        )
        .as_bytes(),
    )?;

    loss_g.draw(format!("img/{}/loss.png", folder))?;

    graph::draw_2hist(
        [cv_valid_loss, cv_train_loss],
        "Validation/Training MSE",
        ("Iterations", "Validataion/Training MSE"),
        format!("{}/cv_l.png", img),
    )?;
    
    graph::draw_histogram(
        r2_score,
        "Cross Validation R2 Scores",
        ("Iterations", "R2 Scores"),
        format!("{}/r2_score.png", img),
    )?;
    Ok(())
}
