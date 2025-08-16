use burn::{module::Module, nn::{loss::CrossEntropyLoss, Embedding, EmbeddingConfig}, prelude::Backend, tensor::{Int, Tensor}};
use burn::prelude::Config;

#[derive(Debug, Config)]
pub struct  BigramModelConfig {
    vocab_size: usize,
    pub(crate) block_size: usize,
}

impl BigramModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramModel<B> {
        let token_embedding_table = EmbeddingConfig::new(self.vocab_size, self.vocab_size).init(device);

        BigramModel { token_embedding_table } 
    }
}
#[derive(Debug, Module)]
pub struct BigramModel<B:Backend> {
    token_embedding_table:Embedding<B>,
}

impl <B: Backend> BigramModel<B> {
    pub fn new(vocab_size:usize, device:&B::Device) -> Self {
        let token_embedding_table = EmbeddingConfig::new(vocab_size, vocab_size).init(device);

        Self { token_embedding_table }
    }

    pub fn forward(&self, idx:Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.token_embedding_table.forward(idx)
    }

    pub fn loss(&self, idx: Tensor<B, 2, Int>, targets:Tensor<B, 2, Int>,  device:&B::Device) -> Tensor<B, 1> {
        let logits = self.forward(idx);

        let [b, t, c] = logits.dims();
        let logits = logits.reshape([b * t, c]);
        let targets = targets.reshape([b * t]);

        CrossEntropyLoss::new(None, device).forward(logits, targets)
    }
}
