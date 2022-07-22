use crate::io::read_lines;
use std::error::Error;
use rand::prelude::SliceRandom;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct Data {
    pub inputs: Vec<f64>,
    pub labels: Vec<f64>
}
pub struct DataSet {
    datas: Vec<Data>,
}

impl DataSet {
    pub fn new(datas: Vec<Data>) -> DataSet {
        DataSet {datas}
    }
    
    pub fn cross_valid_set(&self, percent: f64) -> Vec<(DataSet, DataSet)> {
        if percent < 0.0 && percent > 1.0 {
            panic!("argument percent must be in range [0, 1]")
        }
        let k = (percent * (self.datas.len() as f64)).ceil() as usize; // fold size
        let n = (self.datas.len() as f64 / k as f64).ceil() as usize; // number of folds
        let mut set: Vec<(DataSet, DataSet)> = vec![]; 

        let mut curr: usize = 0;
        
        for _ in 0..n {
            let r_pt: usize = if curr + k > self.datas.len() {self.datas.len()} else {curr + k};

            let validation_set: Vec<Data> = self.datas[curr..r_pt].to_vec();
            let training_set: Vec<Data> = 
                if curr > 0 {
                    let mut temp = self.datas[0..curr].to_vec();
                    temp.append(&mut self.datas[r_pt..self.datas.len()].to_vec());
                    temp
                }
                else {
                    self.datas[r_pt..self.datas.len()].to_vec()
                };

            set.push((DataSet::new(training_set), DataSet::new(validation_set)));
            curr += k
        }
        set
    }
    
    pub fn get_datas(&self) -> Vec<Data> {
        self.datas.clone()
    }

    pub fn get_shuffled(&self) -> Vec<Data> {
        let mut shuffled_datas = self.datas.clone();
        shuffled_datas.shuffle(&mut rand::thread_rng());
        shuffled_datas
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

pub fn flood_std_sc(input: f64) -> f64 {
    let std = 120.451671938513000;
    let mean = 340.303963198868000;
    (input-mean)/std
}

pub fn flood_stdsc_rev(input: f64) -> f64 {
    let std = 120.451671938513000;
    let mean = 340.303963198868000;
    input * std + mean
}

pub fn flood_dataset() -> Result<DataSet, Box<dyn Error>> {
    #[derive(Debug, Deserialize)]
    struct Record {
        s1_t3: f64,
        s1_t2: f64,
        s1_t1: f64,
        s1_t0: f64,
        s2_t3: f64,
        s2_t2: f64,
        s2_t1: f64,
        s2_t0: f64,
        t7: f64
    }

    let mut datas: Vec<Data> = vec![];
    let mut reader = csv::Reader::from_path("data/flood_dataset.csv")?;
    for record in reader.deserialize() {
        let record: Record = record?;
        let mut inputs: Vec<f64> = vec![];
        // station 1
        inputs.push(flood_std_sc(record.s1_t3));
        inputs.push(flood_std_sc(record.s1_t2));
        inputs.push(flood_std_sc(record.s1_t1));
        inputs.push(flood_std_sc(record.s1_t0));
        // station 2
        inputs.push(flood_std_sc(record.s2_t3));
        inputs.push(flood_std_sc(record.s2_t2));
        inputs.push(flood_std_sc(record.s2_t1));
        inputs.push(flood_std_sc(record.s2_t0));

        let labels: Vec<f64> = vec![f64::from(flood_std_sc(record.t7))];
        datas.push(Data {inputs, labels});
    }
    Ok(DataSet::new(datas))
}

pub fn cross_dataset() -> Result<DataSet, Box<dyn Error>> {
    let mut datas: Vec<Data> = vec![];
    let mut lines = read_lines("data/cross.pat")?;
    while let (Some(_), Some(Ok(l1)), Some(Ok(l2))) = (lines.next(), lines.next(), lines.next()) {
        let mut inputs: Vec<f64> = vec![];
        let mut labels: Vec<f64> = vec![];
        for w in l1.split(" ") {
            let v: f64 = w.parse().unwrap();
            inputs.push(v);
        }   
        for w in l2.split(" ") {
            let v: f64 = w.parse().unwrap();
            labels.push(v);
        }   
        datas.push(Data {inputs, labels});
    }
    Ok(DataSet::new(datas))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_test() -> Result<(), Box<dyn Error>> {
        /*         let dt = flood_dataset()?;
        dt.cross_valid_set(0.1);

        //println!("\n{:?}", dt.get_datas());
        //println!("\n{:?}", dt.validation_set());
        */

        if let Ok(dt) = cross_dataset() {
            println!("{:?}", dt.get_datas());
        }

        Ok(())
    }
}