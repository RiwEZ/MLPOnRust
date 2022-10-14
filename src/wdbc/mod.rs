pub mod ga;

use std::error::Error;
use crate::{model::{self, Layer, Net}, activator, utills::data, loss};

pub fn wdbc_30_15_1() {
    fn model() -> Net {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(30, 15, 1.0, activator::sigmoid()));
        layers.push(Layer::new(15, 1, 1.0, activator::sigmoid()));
        Net::from_layers(layers)
    }
    wdbc_ga(&model).unwrap();
}

/// train mlp with genitic algorithm
pub fn wdbc_ga(model: &dyn Fn() -> Net) -> Result<(), Box<dyn Error>> {
    let dataset = data::wdbc_dataset()?;
    let mut loss = loss::Loss::square_err();

    for (j, dt) in dataset.cross_valid_set(0.1).iter().enumerate() {
        let mut net = model();
        let training_set = &dt.0;
        let validation_set = &dt.1;

        // training with GA
        let pop = ga::init_pop(&net, 10);
        let mut ind_fitness: Vec<f64> = vec![];
        for ind in pop {
            ga::assign_ind(&mut net, &ind);
            let mut running_loss = 0.0;
            for data in training_set.get_shuffled() {
                let result = net.forward(&data.inputs);
                running_loss += loss.criterion(&result, &data.labels);
            }
            let mse = running_loss/training_set.len() as f64;
            let fitness = 1.0/mse;
            ind_fitness.push(fitness);
            println!("{}: {}, {}", j, fitness, mse);
        }
        // selection
    }

    Ok(())
}

#[test]
fn temp_entry() {
    wdbc_30_15_1();
}