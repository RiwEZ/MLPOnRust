use rand::prelude::SliceRandom;

pub struct DataSet {
    pub datas: Vec<Data>,
}

impl DataSet {
    pub fn new(datas: Vec<Data>) -> DataSet {
        DataSet {datas}
    }

    pub fn get_samples(&self) -> Vec<Data> {
        let mut shuffled_datas = self.datas.clone();
        shuffled_datas.shuffle(&mut rand::thread_rng());
        shuffled_datas
    }
}
#[derive(Debug)]
#[derive(Clone)]
pub struct Data {
    pub inputs: Vec<f64>,
    pub labels: Vec<f64>
}

pub fn xor_dataset() -> DataSet {
    let inputs = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let labels = vec![[0.0], [1.0], [1.0], [0.0]];
    let mut datas: Vec<Data> = vec![];
    for i in 0..4 {
        datas.push(Data {inputs: inputs[i].to_vec(), labels: labels[i].to_vec()});
    }
    
    DataSet::new(datas)
}