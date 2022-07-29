use crate::model;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::Zip;
use model::Layer;
use crate::activator;
use crate::loss;

pub struct Loss {
    outputs: Array1<f64>,
    desired: Array1<f64>,
    pub func: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
    pub der: fn(&Array1<f64>, &Array1<f64>) -> Array1<f64>,
}

impl Loss {
    /// Mean Squared Error
    pub fn mse() -> Loss {
        fn f(output: f64, desired: f64) -> f64 {
            0.5 * (output - desired).powi(2)
        }
        fn func(outputs: &Array1<f64>, desired: &Array1<f64>) -> Array1<f64> {
            let mut result = Array::<f64, _>::zeros(outputs.len());
            Zip::from(&mut result)
                .and(outputs)
                .and(desired)
                .for_each(|v, &o, &d| *v = f(o, d));
            result
        }
        fn f_p(output: f64, desired: f64) -> f64 {
            output - desired
        }
        fn der(outputs: &Array1<f64>, desired: &Array1<f64>) -> Array1<f64> {
            let mut result = Array::<f64, _>::zeros(outputs.len());
            Zip::from(&mut result)
                .and(outputs)
                .and(desired)
                .for_each(|v, &o, &d| *v = f_p(o, d));
            result
        }

        Loss {
            outputs: arr1(&[0.0]),
            desired: arr1(&[0.0]),
            func,
            der,
        }
    }

    /// Binary Cross Entropy
    /*     pub fn bce() -> Loss {
        fn func(output: f64, desired: f64) -> f64 {
            -desired * output.ln() + (1.0 - desired) * (1.0 - output).ln()
        }
        fn der(output: f64, desired: f64) -> f64 {
            -(desired / output - (1.0 - desired) / (1.0 - output))
        }

        Loss {
            outputs: arr1(&[0.0]),
            desired: arr1(&[0.0]),
            func,
            der,
        }
    } */

    pub fn criterion(&mut self, outputs: &Array1<f64>, desired: &Array1<f64>) -> f64 {
        if outputs.len() != desired.len() {
            panic!("outputs size is not equal to desired size");
        }
        self.outputs = outputs.clone();
        self.desired = desired.clone();

        (self.func)(outputs, desired).fold(0.0, |v, x| v + x)
    }

    pub fn backward(&self, layers: &mut Vec<model::Layer>) {
        for l in (0..layers.len()).rev() {
            layers[l].prev_local_grads = layers[l].local_grads.clone(); // copied previous grad before update
            layers[l].prev_grads = layers[l].grads.clone();

            let a_der = layers[l].outputs.map(|x| (layers[l].act.der)(*x)); // apply derivative of act func on outputs
                                                                            
            let local_grad = if l == layers.len() - 1 {
                // output layer
                &(self.der)(&self.outputs, &self.desired) * &a_der
            } else {
                // hidden layer
                // calculate local_grad based on previous local_grad
                &layers[l + 1].w.t().dot(&layers[l + 1].local_grads) * &a_der
            };

            layers[l].local_grads = local_grad.clone();

            let previous_a = if l > 0 {
                layers[l - 1].outputs.map(|x| (layers[l - 1].act.func)(*x))
            } else {
                layers[l].inputs.clone()
            };
            
            for (j, mut row) in layers[l].grads.axis_iter_mut(Axis(0)).enumerate() {
                for (k, col) in row.iter_mut().enumerate() {
                    *col = previous_a[k] * local_grad[j]
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::utills;

    use super::*;

    #[test]
    fn test_mse_func() {
        assert_eq!(
            (Loss::mse().func)(&arr1(&[2.0, 5.0]), &arr1(&[1.0, 0.0])),
            arr1(&[0.5, 12.5])
        );
    }

    #[test]
    fn test_mse_der() {
        assert_eq!(
            (Loss::mse().der)(&arr1(&[2.0, 5.0]), &arr1(&[1.0, 0.0])),
            arr1(&[1.0, 5.0])
        );
    }

    #[test]
    fn test_mse() {
        let mut loss = Loss::mse();

        let l = loss.criterion(&arr1(&[2.0, 1.0, 0.0]), &arr1(&[0.0, 1.0, 2.0]));
        assert_eq!(l, 4.0);

        loss.criterion(
            &arr1(&[34.0, 37.0, 44.0, 47.0, 48.0]),
            &arr1(&[37.0, 40.0, 46.0, 44.0, 46.0]),
        );
        assert_eq!(l, 4.0);
    }

    #[test]
    fn test_backward() {
        let mut layers: Vec<model::Layer> = vec![];
        layers.push(Layer::new(2, 2, 1.0, activator::sigmoid()));
        layers.push(Layer::new(2, 1, 1.0, activator::linear()));
        let mut net = model::Net::from_layers(layers);
        let mut loss = loss::Loss::mse();

        /*         for l in 0..net.layers.len() {
            net.layers[l].w = Array2::<f64>::from_elem(net.layers[l].w.dim(), 1.0);
        }*/


        let dt = utills::data::xor_dataset();

        for _ in 0..5000 {
            let mut r_loss = 0.0;
            for d in dt.get_datas() {
                let result = net.forward(&d.inputs);
                r_loss += loss.criterion(&result, &d.labels);
                loss.backward(&mut net.layers);
                net.update(0.1, 0.1);
            }
            println!("loss: {}", r_loss);
        }

        println!("{}", net.forward(&arr1(&[0.0, 0.0])));
        println!("{}", net.forward(&arr1(&[1.0, 0.0])));
        println!("{}", net.forward(&arr1(&[0.0, 1.0])));
        println!("{}", net.forward(&arr1(&[1.0, 1.0])));
    }


    /*     #[test]
    fn test_bce_func() {
        println!("{}", (Loss::bce().func)(0.9, 0.0));
        println!("{}", (Loss::bce().func)(0.9, 1.0));
    } */
}
