use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::optim::AdamConfig;

use crate::modeling:: BigramModelConfig;
use crate::tokenizing::Tokenizer;
use crate::training::{train, TrainingConfig};

pub mod dataset;
pub mod tokenizing;
pub mod batching;
pub mod modeling;
pub mod training;
pub mod genrating;

fn main() {
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let artifact_dir = "artifact";

    let text: &'static str = include_str!("input.txt");
    println!("text chars len: {}", text.chars().count());

    let tokenizer = Tokenizer::from_text(text);
    println!("vocab size: {}", tokenizer.vocab_size());
    println!("{}", tokenizer.decode(&tokenizer.encode("Hello world!")));

   train::<MyAutodiffBackend>
    (
      artifact_dir,
      tokenizer.encode(&text[0..10000]),
        TrainingConfig::new(
            BigramModelConfig::new(tokenizer.vocab_size(), 
            AdamConfig::new)),
            device
    );

    // let new_tokens = genrate(&trained_model, tokenizer.encode("A"), 100, &device);
    // println!("{}", tokenizer.decode(&new_tokens));
}
