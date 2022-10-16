pub mod ga;
pub mod selection;

use rand::seq::index::sample;
use std::{error::Error, time::Instant};

use crate::{
    activator, loss,
    model::{self, Layer, Net},
    utills::data::{self, confusion_count},
    wdbc::ga::Individual,
};

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

    let start = Instant::now();
    for (j, dt) in dataset.cross_valid_set(0.1).iter().enumerate() {
        if j > 0 {
            break;
        }

        let mut net = model();
        let training_set = &dt.0;
        let validation_set = &dt.1;
        let max_gen = 200;

        // training with GA
        let mut pop = ga::init_pop(&net, 20);

        for k in 0..max_gen {
            let mut max_fitness = f64::MIN;
            for i in 0..pop.len() {
                ga::assign_ind(&mut net, &pop[i]);
                let mut matrix = [[0, 0], [0, 0]];

                for data in training_set.get_shuffled() {
                    let result = net.forward(&data.inputs);
                    confusion_count(&mut matrix, &result, &data.labels);
                }
                let fitness = (matrix[0][0] + matrix[1][1]) as f64 / training_set.len() as f64;
                pop[i].set_fitness(fitness);
                max_fitness = fitness.max(max_fitness);
                //println!("{}: {}, {}", j, fitness, mse);
            }
            // selection
            let p1 = selection::d_tornament(&pop);
            let mating_result = ga::mating(&p1);
            let mut mut_result = ga::mutate(&mating_result, 6);

            let mut new_pop: Vec<Individual> = vec![];
            new_pop.append(&mut mut_result);
            let pop_need = pop.len() - new_pop.len();
            let indexes: Vec<_> = sample(&mut rand::thread_rng(), p1.len(), pop_need).into_vec();
            for i in indexes {
                new_pop.push(p1[i].clone());
            }
            pop = new_pop;
            println!("[{}] max_fitness: {}", k, max_fitness);
        }
        let best_ind = pop
            .iter()
            .reduce(|best, x| if best.fitness < x.fitness { x } else { best });

        let mut matrix = [[0, 0], [0, 0]];
        for data in validation_set.get_datas() {
            let result = net.forward(&data.inputs);
            confusion_count(&mut matrix, &result, &data.labels);
        }
        let valid_acc = (matrix[0][0] + matrix[1][1]) as f64 / validation_set.len() as f64;
        println!("valid_acc: {}", valid_acc);
    }
    let duration = start.elapsed();
    println!("Time used: {} sec", duration.as_secs());

    Ok(())
}

#[test]
fn temp_entry() {
    wdbc_30_15_1();
}