use super::io::read_lines;
use rand::prelude::SliceRandom;
use serde::Deserialize;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct Data {
    pub inputs: Vec<f64>,
    pub labels: Vec<f64>,
}
#[derive(Clone)]
pub struct DataSet {
    datas: Vec<Data>,
}

impl DataSet {
    pub fn new(datas: Vec<Data>) -> DataSet {
        DataSet { datas }
    }

    pub fn cross_valid_set(&self, percent: f64) -> Vec<(DataSet, DataSet)> {
        if percent < 0.0 && percent > 1.0 {
            panic!("argument percent must be in range [0, 1]")
        }
        let k = (percent * (self.datas.len() as f64)).ceil() as usize; // fold size
        let n = (self.datas.len() as f64 / k as f64).ceil() as usize; // number of folds
        let datas = self.get_shuffled().clone(); // shuffled data before slicing it
        let mut set: Vec<(DataSet, DataSet)> = vec![];

        let mut curr: usize = 0;
        for _ in 0..n {
            let r_pt: usize = if curr + k > datas.len() {
                datas.len()
            } else {
                curr + k
            };

            let validation_set: Vec<Data> = datas[curr..r_pt].to_vec();
            let training_set: Vec<Data> = if curr > 0 {
                let mut temp = datas[0..curr].to_vec();
                temp.append(&mut datas[r_pt..datas.len()].to_vec());
                temp
            } else {
                datas[r_pt..datas.len()].to_vec()
            };

            set.push((DataSet::new(training_set), DataSet::new(validation_set)));
            curr += k
        }
        set
    }

    pub fn data_points(&self) -> Vec<f64> {
        let mut data_points: Vec<f64> = vec![];
        for mut dt in self.datas.clone() {
            data_points.append(&mut dt.inputs);
            data_points.append(&mut dt.labels);
        }
        data_points
    }

    pub fn max(&self) -> f64 {
        self.data_points()
            .iter()
            .fold(f64::NAN, |max, &v| v.max(max))
    }

    pub fn min(&self) -> f64 {
        self.data_points()
            .iter()
            .fold(f64::NAN, |min, &v| v.min(min))
    }

    pub fn std(&self) -> f64 {
        let mean = self.mean();
        let data_points = self.data_points();
        let n = data_points.len() as f64;
        data_points
            .iter()
            .fold(0.0f64, |sum, &val| sum + (val - mean).powi(2) / n)
            .sqrt()
    }

    pub fn mean(&self) -> f64 {
        let data_points = self.data_points();
        let n = data_points.len() as f64;
        data_points.iter().fold(0.0f64, |mean, &val| mean + val / n)
    }

    pub fn len(&self) -> usize {
        self.datas.len()
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

pub fn minmax_norm(dataset: &DataSet, min: f64, max: f64) -> DataSet {
    let datas: Vec<Data> = dataset
        .get_datas()
        .into_iter()
        .map(|dt| {
            let inputs: Vec<f64> = dt.inputs.iter().map(|x| (x - min) / (max - min)).collect();
            let labels: Vec<f64> = dt.labels.iter().map(|x| (x - min) / (max - min)).collect();
            Data { inputs, labels }
        })
        .collect();
    DataSet::new(datas)
}

pub fn standardization(dataset: &DataSet, mean: f64, std: f64) -> DataSet {
    let datas: Vec<Data> = dataset
        .get_datas()
        .into_iter()
        .map(|dt| {
            let inputs: Vec<f64> = dt.inputs.iter().map(|x| (x - mean) / std).collect();
            let labels: Vec<f64> = dt.labels.iter().map(|x| (x - mean) / std).collect();
            Data { inputs, labels }
        })
        .collect();
    DataSet::new(datas)
}

pub fn un_standardization(value: f64, mean: f64, std: f64) -> f64 {
    value * std + mean
}

pub fn xor_dataset() -> DataSet {
    let inputs = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let labels = vec![[0.0], [1.0], [1.0], [0.0]];
    let mut datas: Vec<Data> = vec![];
    for i in 0..4 {
        datas.push(Data {
            inputs: inputs[i].to_vec(),
            labels: labels[i].to_vec(),
        });
    }

    DataSet::new(datas)
}

pub fn flood_dataset() -> Result<DataSet, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct Record {
        s1_t3: f64,
        s1_t2: f64,
        s1_t1: f64,
        s1_t0: f64,
        s2_t3: f64,
        s2_t2: f64,
        s2_t1: f64,
        s2_t0: f64,
        t7: f64,
    }

    let mut datas: Vec<Data> = vec![];
    let mut reader = csv::Reader::from_path("data/flood_dataset.csv")?;
    for record in reader.deserialize() {
        let record: Record = record?;
        let mut inputs: Vec<f64> = vec![];
        // station 1
        inputs.push(record.s1_t3);
        inputs.push(record.s1_t2);
        inputs.push(record.s1_t1);
        inputs.push(record.s1_t0);
        // station 2
        inputs.push(record.s2_t3);
        inputs.push(record.s2_t2);
        inputs.push(record.s2_t1);
        inputs.push(record.s2_t0);

        let labels: Vec<f64> = vec![f64::from(record.t7)];
        datas.push(Data { inputs, labels });
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
            // class 1 0 -> 1
            // class 0 1 -> 0
            labels.push(v);
            break;
        }
        datas.push(Data { inputs, labels });
    }
    Ok(DataSet::new(datas))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_test() -> Result<(), Box<dyn Error>> {
        /*
        let dt = flood_dataset()?.cross_valid_set(0.1);
        let training_set = &dt[0].0;
        let validation_set = &dt[0].1;

        println!("mean: {}, std: {}", validation_set.mean(), validation_set.std());
        println!("\n{:?}", validation_set.get_datas());
        println!("\n\n{:?}", standardization(validation_set).get_datas());
         */

        /*
        if let Ok(dt) = cross_dataset() {
            println!("{:?}", dt.get_datas());
        }
        */

        Ok(())
    }

    #[test]
    fn test_min_max() -> Result<(), Box<dyn Error>> {
        let dt = flood_dataset()?;
        assert_eq!(dt.max(), 628.0);
        assert_eq!(dt.min(), 95.0);
        Ok(())
    }
}
