use crate::activator;
use crate::model;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayBase;
use serde_json::{json, to_writer_pretty, Value};
use std::error::Error;
use std::fs::create_dir;
use std::fs::File;
use std::io::Read;
use std::io::{self, BufRead};
use std::path::Path;

pub fn save(layers: &Vec<model::Layer>, path: String) -> Result<(), Box<dyn Error>> {
    let mut json: Vec<Value> = vec![];

    for l in layers {
        json.push(json!({
            "inputs": l.inputs.len(),
            "outputs": l.outputs.len(),
            "w": l.w,
            "b": l.b,
            "act": l.act.name
        }));
    }
    let result = json!(json);
    let file = File::create(path)?;
    to_writer_pretty(&file, &result)?;
    Ok(())
}

pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn read_file<P>(filename: P) -> Result<String, Box<dyn Error>>
where
    P: AsRef<Path>,
{
    let mut file = File::open(filename)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

pub fn load<P>(filename: P) -> Result<model::Net, Box<dyn Error>>
where
    P: AsRef<Path>,
{
    let contents = read_file(filename)?;

    let json: Value = serde_json::from_str(&contents)?;
    let mut layers: Vec<model::Layer> = vec![];

    for l in json.as_array().unwrap() {
        // default layer activation is simeple linear f(x) = x
        let mut layer = model::Layer::new(
            l["inputs"].as_u64().unwrap() as usize,
            l["outputs"].as_u64().unwrap() as usize,
            0.0,
            activator::linear(),
        );
        // setting activation function
        if l["act"] == "sigmoid" {
            layer.act = activator::sigmoid();
        }

        // setting weights and bias
        let w: Array2<f64> = serde_json::from_str(&l["w"].to_string()).unwrap();
        let b: Array1<f64> = serde_json::from_str(&l["b"].to_string()).unwrap();
        layer.w = w;
        layer.b = b;
        layers.push(layer);
    }

    Ok(model::Net::from_layers(layers))
}

/// Check if specify folder exists in models and img folder, if not create it
///
/// Return models path and img path
pub fn check_dir(folder: &str) -> Result<(String, String), Box<dyn Error>> {
    let models_path = format!("models/{}", folder);
    if !Path::new(&models_path).exists() {
        create_dir(&models_path)?;
    }
    let img_path = format!("img/{}", folder);
    if !Path::new(&img_path).exists() {
        create_dir(&img_path)?;
    }
    Ok((models_path, img_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_test() -> Result<(), Box<dyn Error>> {
        let net = model::Net::new(vec![2, 2, 2]);
        let w = net.layers[0].w.clone();
        let b = net.layers[0].b.clone();

        save(&net.layers, "models/xor.json".to_string())?;

        let loaded_net = load("models/xor.json")?;
        assert_eq!(loaded_net.layers[0].w, w);
        assert_eq!(loaded_net.layers[0].b, b);
        Ok(())
    }
}
