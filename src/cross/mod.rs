//! Contains training code for variations of cross.pat dataset models.
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

pub fn cross_2_4_2(lr: f64, momentum: f64, folder: &str) -> Result<(), Box<dyn Error>> {
    fn model() -> Net {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(2, 4, 1.0, activator::sigmoid()));
        layers.push(Layer::new(4, 2, 1.0, activator::sigmoid()));
        Net::from_layers(layers)
    }

    cross_fit(&model, lr, momentum, folder)?;
    Ok(())
}

pub fn cross_fit(
    model: &dyn Fn() -> Net,
    lr: f64,
    momentum: f64,
    folder: &str,
) -> Result<(), Box<dyn Error>> {
    let (models, img) = utills::io::check_dir(folder)?;

    let dataset = data::cross_dataset()?;
    let mut loss = loss::Loss::bce();
    let epochs = 2000;

    let mut cv_score: Vec<f64> = vec![];
    let mut loss_g = graph::LossGraph::new();

    let start = Instant::now();
    for (j, dt) in dataset.cross_valid_set(0.1).iter().enumerate() {
        // creating a model
        let mut net = model();

        // get training set and validation set
        let training_set = &dt.0;
        let validation_set = &dt.1;

        // training
        let mut loss_vec: Vec<f64> = vec![];
        let mut valid_loss_vec: Vec<f64> = vec![];
        for i in 0..epochs {
            let mut running_loss: f64 = 0.0;

            for data in training_set.get_shuffled() {
                let result = net.forward(data.inputs.clone());

                running_loss += loss.criterion(result, data.labels.clone());
                loss.backward(&mut net.layers);

                net.update(lr, momentum);
            }
            running_loss /= training_set.len() as f64;
            loss_vec.push(running_loss);

            let mut valid_loss: f64 = 0.0;
            let mut correct_predict = 0;
            for data in validation_set.get_datas() {
                let result = net.forward(data.inputs.clone());
                
                if result[0] > 0.5 && data.labels.clone()[0] == 1.0 {
                    correct_predict += 1;
                }
                else if result[0] <= 0.5 && data.labels.clone()[1] == 0.0 {
                    correct_predict += 1;
                }

                valid_loss += loss.criterion(result, data.labels.clone());
            }
            valid_loss /= validation_set.get_datas().len() as f64;
            valid_loss_vec.push(valid_loss);

            if i == epochs - 1 {
                cv_score.push(correct_predict as f64/validation_set.len() as f64);
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
    file.write_all(format!("cv_score: {:?}\n\ntime used: {:?}", cv_score, duration).as_bytes())?;

    loss_g.draw(format!("img/{}/loss.png", folder))?;
    graph::draw_loss_scores(cv_score.clone(), format!("{}/cv_score.png", img))?;
    Ok(())
}
