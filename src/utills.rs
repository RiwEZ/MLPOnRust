use rand::Rng;

pub struct DataSet {
    pub datas: Vec<Data>,
}

impl DataSet {
    pub fn new(datas: Vec<Data>) -> DataSet {
        DataSet {datas}
    }

    pub fn get_sample(&self) -> Data {
        let index = rand::thread_rng().gen_range(0..self.datas.len());
        self.datas[index].clone()
    }
}
#[derive(Debug)]
pub struct Data {
    pub inputs: Vec<f64>,
    pub labels: Vec<f64>
}

impl Data {
    pub fn clone(&self) -> Data {
        Data {inputs: self.inputs.clone(), labels: self.labels.clone()}
    }
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