use super::io::read_lines;
use rand::prelude::SliceRandom;
use serde::Deserialize;
use std::error::Error;

pub fn max(vec: &Vec<f64>) -> f64 {
    vec.iter().fold(f64::NAN, |max, &v| v.max(max))
}

pub fn min(vec: &Vec<f64>) -> f64 {
    vec.iter().fold(f64::NAN, |min, &v| v.min(min))
}

pub fn std(vec: &Vec<f64>, mean: f64) -> f64 {
    let n = vec.len() as f64;
    vec.iter()
        .fold(0.0f64, |sum, &val| sum + (val - mean).powi(2) / n)
        .sqrt()
}

pub fn mean(vec: &Vec<f64>) -> f64 {
    let n = vec.len() as f64;
    vec.iter().fold(0.0f64, |mean, &val| mean + val / n)
}

pub fn standardization(data: &Vec<f64>, mean: f64, std: f64) -> Vec<f64> {
    data.iter().map(|x| (x - mean) / std).collect()
}

pub fn minmax_norm(data: &Vec<f64>, min: f64, max: f64) -> Vec<f64> {
    data.iter().map(|x| (x - min) / (max - min)).collect()
}

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
        max(&self.data_points())
    }

    pub fn min(&self) -> f64 {
        min(&self.data_points())
    }

    pub fn std(&self) -> f64 {
        std(&self.data_points(), self.mean())
    }

    pub fn mean(&self) -> f64 {
        mean(&self.data_points())
    }

    pub fn len(&self) -> usize {
        self.datas.len()
    }

    pub fn standardization(&self) -> DataSet {
        // this kind of wrong
        let mean = self.mean();
        let std = self.std();
        let datas: Vec<Data> = self
            .get_datas()
            .into_iter()
            .map(|dt| {
                let inputs: Vec<f64> = standardization(&dt.inputs, mean, std);
                let labels: Vec<f64> = standardization(&dt.labels, mean, std);
                Data { inputs, labels }
            })
            .collect();
        DataSet::new(datas)
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

pub fn confusion_count(
    matrix: &mut [[i32; 2]; 2],
    result: &Vec<f64>,
    label: &Vec<f64>,
    threshold: f64,
) {
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

pub fn wdbc_dataset() -> Result<DataSet, Box<dyn Error>> {
    let mut labels: Vec<Vec<f64>> = vec![];
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(30);
    let mut lines = read_lines("data/wdbc.txt")?;
    for _ in 0..30 {
        features.push(vec![]);
    }
    while let Some(Ok(line)) = lines.next() {
        let arr: Vec<&str> = line.split(",").collect();
        // M (malignant) = 1.0, B (benign) = 0.0
        if arr[1] == "M" {
            labels.push(vec![0.0]);
        } else if arr[1] == "B" {
            labels.push(vec![1.0]);
        }
        let mut i = 0;
        for w in &arr[2..] {
            let v: f64 = w.parse()?;
            features[i].push(v);
            i += 1
        }
    }

    // standardization each column
    for data in features.iter_mut() {
        let mean = mean(&data);
        let std = std(&data, mean);
        let max = max(&data);
        let min = min(&data);
        *data = minmax_norm(&data, min, max);
    }

    let datas: Vec<Data> = labels
        .iter()
        .zip(0..)
        .map(|(label, i)| {
            let inputs: Vec<f64> = features.iter().map(|x| x[i]).collect();
            Data {
                labels: label.clone(),
                inputs,
            }
        })
        .collect();

    Ok(DataSet::new(datas))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_test() -> Result<(), Box<dyn Error>> {
        let dt = wdbc_dataset()?;
        println!("{:?}", dt.get_datas()[0].inputs.len());

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
