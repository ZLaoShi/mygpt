use burn::{prelude::Backend, tensor::{activation::softmax, Int, Tensor, TensorData}};
use rand::prelude::*;

use crate::modeling::BigramModel;

pub fn genrate<B: Backend>(model:&BigramModel<B>, prompt:Vec<i32>, max_new_tokens: usize, device:&B::Device) -> Vec<i32> {
    let mut tokens = prompt;
    let mut rng = rand::rng();

    for i in tokens.len()..tokens.len() + max_new_tokens {
        let x = TensorData::new(tokens.clone(), [1, i]);
        let x = Tensor::<B, 2, Int>::from_data(x, device);

        let logits = model.forward(x);
        let logits = logits.slice([0..1, i - 1..i]);

        let props = softmax(logits, 2);
        let props = props.to_data().to_vec::<f32>().unwrap();

        let distr = rand::distr::weighted::WeightedIndex::new(&props).unwrap();
        let next_token = distr.sample(&mut rng) as i32;

        tokens.push(next_token);
    }
 
    tokens
}