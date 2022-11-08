use super::io::read_lines;
use chrono::{DateTime, Duration, TimeZone, Utc};
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

    pub fn len(&self) -> usize {
        self.datas.len()
    }

    pub fn standardization(&self, valid_set: &DataSet) -> (DataSet, DataSet) {
        let size = self.datas[0].inputs.len();
        let features: Vec<(Vec<f64>, Vec<f64>)> = (0..size)
            .into_iter()
            .map(|i| {
                let feature = self.get_feature(i);
                let v_feature = valid_set.get_feature(i);
                let mean = mean(&feature);
                let std = std(&feature, mean);
                (
                    standardization(&feature, mean, std),
                    standardization(&v_feature, mean, std),
                )
            })
            .collect();

        let datas: Vec<Data> = self
            .datas
            .iter()
            .enumerate()
            .map(|(i, dt)| Data {
                labels: dt.labels.clone(),
                inputs: features.iter().map(|x| x.0[i]).collect(),
            })
            .collect();

        let v_datas: Vec<Data> = valid_set
            .datas
            .iter()
            .enumerate()
            .map(|(i, dt)| Data {
                labels: dt.labels.clone(),
                inputs: features.iter().map(|x| x.1[i]).collect(),
            })
            .collect();

        (DataSet::new(datas), DataSet::new(v_datas))
    }

    pub fn minmax_norm(&self, valid_set: &DataSet) -> (DataSet, DataSet) {
        let size = self.datas[0].inputs.len();
        let features: Vec<(Vec<f64>, Vec<f64>)> = (0..size)
            .into_iter()
            .map(|i| {
                let feature = self.get_feature(i);
                let v_feature = valid_set.get_feature(i);
                let min = min(&feature);
                let max = max(&feature);
                (
                    minmax_norm(&feature, min, max),
                    minmax_norm(&v_feature, min, max),
                )
            })
            .collect();

        let datas: Vec<Data> = self
            .datas
            .iter()
            .enumerate()
            .map(|(i, dt)| Data {
                labels: dt.labels.clone(),
                inputs: features.iter().map(|x| x.0[i]).collect(),
            })
            .collect();

        let v_datas: Vec<Data> = valid_set
            .datas
            .iter()
            .enumerate()
            .map(|(i, dt)| Data {
                labels: dt.labels.clone(),
                inputs: features.iter().map(|x| x.1[i]).collect(),
            })
            .collect();

        (DataSet::new(datas), DataSet::new(v_datas))
    }

    pub fn get_datas(&self) -> Vec<Data> {
        self.datas.clone()
    }

    pub fn get_feature(&self, i: usize) -> Vec<f64> {
        if i >= self.datas[0].inputs.len() {
            panic!("i should not exceed inputs feature size");
        }

        self.datas.iter().map(|data| data.inputs[i]).collect()
    }

    pub fn get_label(&self, i: usize) -> Vec<f64> {
        if i >= self.datas[0].labels.len() {
            panic!("i should not exceed inputs feature size");
        }

        self.datas.iter().map(|data| data.labels[i]).collect()
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
    let mut datas: Vec<Data> = vec![];
    let mut lines = read_lines("data/wdbc.txt")?;
    while let Some(Ok(line)) = lines.next() {
        let mut inputs: Vec<f64> = vec![];
        let mut labels: Vec<f64> = vec![]; // M (malignant) = 1.0, B (benign) = 0.0
        let arr: Vec<&str> = line.split(",").collect();
        if arr[1] == "M" {
            labels.push(1.0);
        } else if arr[1] == "B" {
            labels.push(0.0);
        }
        for w in &arr[2..] {
            let v: f64 = w.parse()?;
            inputs.push(v);
        }
        datas.push(Data { inputs, labels });
    }
    Ok(DataSet::new(datas))
}

/// Return `(desired = next five days, desired = next ten days)`
pub fn airquality_dataset() -> Result<(DataSet, DataSet), Box<dyn Error>> {
    // nx is not used
    #[derive(Deserialize, Debug)]
    struct Record {
        #[serde(rename = "Date")]
        date: String,
        #[serde(rename = "Time")]
        time: String,
        #[serde(rename = "PT08.S1(CO)")]
        pt_s1: i32,
        #[serde(rename = "C6H6(GT)")]
        benzene: f64,
        #[serde(rename = "PT08.S2(NMHC)")]
        pt_s2: i32,
        #[serde(rename = "PT08.S3(NOx)")]
        pt_s3: i32,
        #[serde(rename = "PT08.S4(NO2)")]
        pt_s4: i32,
        #[serde(rename = "PT08.S5(O3)")]
        pt_s5: i32,
        #[serde(rename = "T")]
        temp: f64,
        #[serde(rename = "RH")]
        rh: f64,
        #[serde(rename = "AH")]
        ah: f64,
    }
    #[derive(Debug)]
    struct RecData {
        datetime: DateTime<Utc>,
        input: Vec<f64>,
        output: Vec<f64>,
        pub okay: bool,
    }
    /// add to input if input is true else add to output
    fn rec_add(recdata: &mut RecData, v: f64, input: bool) {
        if v == -200.0 {
            recdata.okay = false;
        }
        if input {
            recdata.input.push(v);
        } else {
            recdata.output.push(v)
        };
    }

    impl RecData {
        pub fn new(record: &Record) -> RecData {
            let datetime_str = format!("{} {}", record.date, record.time);
            let datetime = Utc
                .datetime_from_str(&datetime_str, "%-m/%-d/%Y %-H:%M:%S")
                .unwrap();

            let mut recdata = RecData {
                datetime,
                input: vec![],
                output: vec![],
                okay: true,
            };
            rec_add(&mut recdata, record.pt_s1 as f64, true);
            rec_add(&mut recdata, record.pt_s2 as f64, true);
            rec_add(&mut recdata, record.pt_s3 as f64, true);
            rec_add(&mut recdata, record.pt_s4 as f64, true);
            rec_add(&mut recdata, record.pt_s5 as f64, true);
            rec_add(&mut recdata, record.temp, true);
            rec_add(&mut recdata, record.rh, true);
            rec_add(&mut recdata, record.ah, true);
            rec_add(&mut recdata, record.benzene, false);
            recdata
        }
    }

    let mut reader = csv::Reader::from_path("data/AirQualityUCI.csv")?;
    let mut rec_datas: Vec<RecData> = vec![];
    for record in reader.deserialize() {
        let rec_data = RecData::new(&record.unwrap());
        if rec_data.okay {
            rec_datas.push(rec_data)
        };
    }

    //  Duration::days(5)
    let mut datas_five: Vec<Data> = vec![];
    let mut datas_ten: Vec<Data> = vec![];

    for (i, x) in rec_datas.iter().enumerate() {
        let next_five_days = x.datetime + Duration::days(5);
        let next_ten_dats = x.datetime + Duration::days(10);

        let mut labels_five: Vec<f64> = vec![];
        let mut labels_ten: Vec<f64> = vec![];

        for y in &rec_datas[i..] {
            if y.datetime == next_five_days {
                labels_five = y.output.clone();
            }
            if y.datetime == next_ten_dats {
                labels_ten = y.output.clone();
                break;
            }
        }

        if labels_five.len() == 0 && labels_ten.len() == 0 {
            break;
        }
        if labels_five.len() != 0 {
            datas_five.push(Data {
                inputs: x.input.clone(),
                labels: labels_five,
            });
        }
        if labels_ten.len() != 0 {
            datas_ten.push(Data {
                inputs: x.input.clone(),
                labels: labels_ten,
            });
        }
    }
    Ok((DataSet::new(datas_five), DataSet::new(datas_ten)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_airquality() -> Result<(), Box<dyn Error>> {
        let (dt5, _) = airquality_dataset().unwrap();

        let dt = &dt5.cross_valid_set(0.1)[0];
        let (train, _) = dt.0.minmax_norm(&dt.1);

        for dt in train.get_datas().iter() {
            for v in dt.inputs.iter() {
                print!("{:.3e} ", v);
            }
            print!("{:.3e}\n", dt.labels[0]);
        }
        println!("{}", train.len());
        Ok(())
    }

    #[test]
    fn test_minmax_norm() {
        let datas: Vec<Data> = (0..=10)
            .into_iter()
            .map(|i| Data {
                labels: vec![0.0],
                inputs: vec![i as f64 * 10.0],
            })
            .collect();
        let v_datas: Vec<Data> = (0..=10)
            .into_iter()
            .map(|i| Data {
                labels: vec![0.0],
                inputs: vec![i as f64 * 5.0],
            })
            .collect();

        let (t, v) = DataSet::new(datas).minmax_norm(&DataSet::new(v_datas));

        let expected = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        for (i, x) in t.get_feature(0).iter().enumerate() {
            assert_eq!(*x, expected[i]);
        }
        let v_expected = vec![
            0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        ];
        for (i, x) in v.get_feature(0).iter().enumerate() {
            assert_eq!(*x, v_expected[i])
        }
    }
}
