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

fn confusion_count(matrix: &mut [[i32; 2]; 2], result: &Vec<f64>, label: &Vec<f64>) {
    let threshold = 0.5;
    if result[0] > threshold {
        // true positive
        if label[0] == 1.0 {
            matrix[0][0] += 1
        } else {
            // false negative
            matrix[1][0] += 1
        }
    } else if result[0] <= threshold {
        // true negative
        if label[0] == 0.0 {
            matrix[1][1] += 1
        }
        // false positive
        else {
            matrix[0][1] += 1
        }
    }
}

pub fn cross_2_4_1(lr: f64, momentum: f64, folder: &str) -> Result<(), Box<dyn Error>> {
    fn model() -> Net {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(2, 4, 1.0, activator::sigmoid()));
        layers.push(Layer::new(4, 1, 1.0, activator::sigmoid()));
        Net::from_layers(layers)
    }

    cross_fit(&model, lr, momentum, folder)?;
    Ok(())
}

pub fn cross_2_8_1(lr: f64, momentum: f64, folder: &str) -> Result<(), Box<dyn Error>> {
    fn model() -> Net {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(2, 8, 1.0, activator::sigmoid()));
        layers.push(Layer::new(8, 1, 1.0, activator::sigmoid()));
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
    let mut loss = loss::Loss::mse();
    let epochs = 5000;

    let mut valid_acc: Vec<f64> = vec![];
    let mut train_acc: Vec<f64> = vec![];
    let mut loss_g = graph::LossGraph::new();
    let mut matrix_vec: Vec<[[i32; 2]; 2]> = vec![];

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
                let mut matrix = [[0, 0], [0, 0]];
                for data in validation_set.get_datas() {
                    let result = net.forward(&data.inputs);
                    confusion_count(&mut matrix, &result, &data.labels);
                }

                let mut matrix2 = [[0, 0], [0, 0]];
                for data in training_set.get_datas() {
                    let result = net.forward(&data.inputs);
                    confusion_count(&mut matrix2, &result, &data.labels);
                }
                valid_acc.push((matrix[0][0] + matrix[1][1]) as f64 / validation_set.len() as f64);
                train_acc.push((matrix2[0][0] + matrix2[1][1]) as f64 / training_set.len() as f64);
                matrix_vec.push(matrix);
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
    file.write_all(format!("cv_score: {:?}\n\ntime used: {:?}", valid_acc, duration).as_bytes())?;

    loss_g.draw(format!("{}/loss.png", img))?;
    
    graph::draw_2hist(
        [valid_acc, train_acc],
        "Validation/Training Accuracy",
        ("Iterations", "Validataion/Training Accuracy"),
        format!("{}/acc.png", img),
    )?;

    graph::draw_confustion(matrix_vec, format!("{}/confusion_matrix.png", img))?;

    Ok(())
}
