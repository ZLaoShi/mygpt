use burn::{
    data::dataloader::batcher::Batcher,
    prelude::Backend,
    tensor::{Int, Tensor, TensorData},
};

use crate::dataset::TokenPair;

pub struct TokenPairBatch<B: Backend> {
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
}
pub struct TokenPairBatcher {
}

impl<B: Backend> Batcher<B, TokenPair, TokenPairBatch<B>> for TokenPairBatcher {
    fn batch(&self, items: Vec<TokenPair>, device: &B::Device) -> TokenPairBatch<B> {
        let inputs = items
            .iter()
            .map(|item| TensorData::new(item.input.clone(), [1, item.input.len()]))
            .map(|data| Tensor::from_data(data, device))
            .collect();

        let targets = items
            .iter()
            .map(|item| TensorData::new(item.target.clone(), [1, item.input.len()]))
            .map(|data| Tensor::from_data(data, device))
            .collect();

        let inputs = Tensor::cat(inputs, 0).to_device(device);
        let targets = Tensor::cat(targets, 0).to_device(device);

        TokenPairBatch { inputs, targets }
    }
}
