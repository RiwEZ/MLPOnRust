use rand::prelude::SliceRandom;
use serde::Deserialize;
use std::error::Error;
use plotters::prelude::*;
use crate::io::read_lines;

pub fn draw_loss(x_vec: Vec<f64>, loss_vec: Vec<f64>, path: String) -> Result<(), Box<dyn Error>> {
    let max_loss = loss_vec.iter().fold(0.0f64, |max, &val| if val > max {val} else {max});
    let max_x = x_vec.iter().fold(0.0f64, |max, &val| if val > max {val} else {max});

    // plotting loss
    let root = BitMapBackend::new(&path, (1024, 768))
        .into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("loss", ("san-serif", 50).into_font())
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..max_x, 0f64..max_loss)?;
    
    chart.configure_mesh().draw()?;

    let mut i: usize = 0;
    chart
        .draw_series(LineSeries::new(loss_vec.into_iter().map(|x| {i += 1; (x_vec[i - 1], x)}), &RED))?
        .label("loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

#[derive(Debug)]
#[derive(Clone)]
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

pub fn flood_dataset() -> Result<DataSet, Box<dyn Error>> {
    fn standard_score(input: f64) -> f64 {
        let std = 120.451671938513000;
        let mean = 340.303963198868000;
        (input-mean)/std
    }

    
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
        inputs.push(standard_score(record.s1_t3));
        inputs.push(standard_score(record.s1_t2));
        inputs.push(standard_score(record.s1_t1));
        inputs.push(standard_score(record.s1_t0));
        // station 2
        inputs.push(standard_score(record.s2_t3));
        inputs.push(standard_score(record.s2_t2));
        inputs.push(standard_score(record.s2_t1));
        inputs.push(standard_score(record.s2_t0));

        let labels: Vec<f64> = vec![f64::from(standard_score(record.t7))];
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