use rand::seq::SliceRandom;

use super::Individual;

/// binary deterministic tournament with reinsertion
pub fn d_tornament(pop: &Vec<Individual>) -> Vec<Individual> {
    let mut results: Vec<Individual> = vec![];
    for _ in 0..pop.len() {
        let players: Vec<_> = pop.choose_multiple(&mut rand::thread_rng(), 2).collect();

        if players[0].fitness > players[1].fitness {
            results.push(players[0].clone());
        } else {
            results.push(players[1].clone());
        }
    }
    results
}
