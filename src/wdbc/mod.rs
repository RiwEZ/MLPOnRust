use std::{error::Error, time::Instant};

use crate::{
    activator,
    ga::{self, Individual},
    loss,
    model::{self, Layer, Net},
    utills::{
        data::{self, confusion_count},
        graph, io,
    },
};

pub fn wdbc_30_15_1() {
    fn model() -> Net {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(30, 15, 1.0, activator::sigmoid()));
        layers.push(Layer::new(15, 1, 1.0, activator::sigmoid()));
        Net::from_layers(layers)
    }
    wdbc_ga(&model, "wdbc-30-15-1").unwrap();
}

/// train mlp with genitic algorithm
pub fn wdbc_ga(model: &dyn Fn() -> Net, folder: &str) -> Result<(), Box<dyn Error>> {
    let dataset = data::wdbc_dataset()?;
    let mut valid_acc: Vec<f64> = vec![];
    let mut train_acc: Vec<f64> = vec![];
    let mut matrix_vec: Vec<[[i32; 2]; 2]> = vec![];
    let threshold = 0.5;

    let start = Instant::now();
    for (j, dt) in dataset.cross_valid_set(0.1).iter().enumerate() {
        let mut net = model();
        let training_set = &dt.0;
        let validation_set = &dt.1;
        let max_gen = 200;
        let mut loss = loss::Loss::square_err();

        // training with GA
        let mut pop = ga::init_pop(&net, 25);
        let mut best_ind = pop[0].clone();

        for k in 0..max_gen {
            let mut max_fitness = f64::MIN;
            let mut local_best_ind = pop[0].clone();

            for p in pop.iter_mut() {
                ga::assign_ind(&mut net, &p);
                let mut matrix = [[0, 0], [0, 0]];
                let mut run_loss = 0.0;
                for data in training_set.get_shuffled() {
                    let result = net.forward(&data.inputs);
                    run_loss += loss.criterion(&result, &data.labels);
                    confusion_count(&mut matrix, &result, &data.labels, threshold);
                }
                let fitness = ((matrix[0][0] + matrix[1][1]) as f64 / training_set.len() as f64)
                    + 0.001 / (run_loss / training_set.len() as f64);
                p.set_fitness(fitness);
                if fitness > max_fitness {
                    max_fitness = fitness;
                    local_best_ind = p.clone();
                }
                // store best individual for all generation
                if best_ind.fitness < fitness {
                    best_ind = p.clone();
                }
            }

            // selection
            let p1 = ga::selection::d_tornament(&pop);
            let mating_result = ga::mating(&p1);
            let mut mut_result = ga::mutate(&mating_result, 20, 0.02);

            let mut new_pop: Vec<Individual> = vec![];
            new_pop.append(&mut mut_result);
            let pop_need = pop.len() - new_pop.len();

            // elitsm
            for _ in 0..pop_need {
                new_pop.push(local_best_ind.clone());
            }

            pop = new_pop;
            println!("[{}, {}] max_fitness: {:.3}", j, k, max_fitness);
        }

        ga::assign_ind(&mut net, &best_ind);
        let mut matrix = [[0, 0], [0, 0]];
        for data in validation_set.get_datas() {
            let result = net.forward(&data.inputs);
            confusion_count(&mut matrix, &result, &data.labels, threshold);
        }
        valid_acc.push((matrix[0][0] + matrix[1][1]) as f64 / validation_set.len() as f64);
        matrix_vec.push(matrix);
        let mut matrix_t = [[0, 0], [0, 0]];
        for data in training_set.get_datas() {
            let result = net.forward(&data.inputs);
            confusion_count(&mut matrix_t, &result, &data.labels, threshold);
        }
        train_acc.push((matrix_t[0][0] + matrix_t[1][1]) as f64 / training_set.len() as f64);
        io::save(&net.layers, format!("models/{}/{}.json", folder, j))?;
    }
    let duration = start.elapsed();
    println!("Time used: {} sec", duration.as_secs());

    graph::draw_acc_2hist(
        [&valid_acc, &train_acc],
        "Training & Validation Accuray",
        ("Iterations", "Accuracy"),
        "img/ga_validacc.png".into(),
    )?;
    graph::draw_confustion(matrix_vec, "img/confusion_matrix.png".into())?;
    Ok(())
}
