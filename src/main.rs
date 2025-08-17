#![recursion_limit = "256"]

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::config::Config;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::record::{CompactRecorder, Recorder};

use crate::genrating::genrate;
use crate::modeling::BigramModelConfig;
use crate::tokenizing::Tokenizer;
use crate::training::{train, TrainingConfig};

pub mod batching;
pub mod dataset;
pub mod genrating;
pub mod modeling;
pub mod tokenizing;
pub mod training;

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

    let model = 
      if let Some(config) = TrainingConfig::load(format!("{artifact_dir}/config.json")).ok() {
          let record = CompactRecorder::new()
              .load(format!("{artifact_dir}/model").into(), &device)
              .expect("Trained model should exist; run train first");

          config.model.init::<MyBackend>(&device).load_record(record)
      } else {
          train::<MyAutodiffBackend>(
              artifact_dir,
              tokenizer.encode(&text[0..10000]),
              TrainingConfig::new(
                  BigramModelConfig::new(tokenizer.vocab_size(), 256),
                  AdamConfig::new(),
                  10,
                  64,
                  9,
                  1.0e-3,
              ),
              device.clone(),
          );

        let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Trained model should exist; run train first"); 
        let record = CompactRecorder::new()
            .load(format!("{artifact_dir}/model").into(), &device)
            .expect("Trained model should exist; run train first");

        config.model.init::<MyBackend>(&device).load_record(record)
      };

    let new_tokens = genrate(&model, tokenizer.encode("A"), 100, &device);
    println!("{}", tokenizer.decode(&new_tokens));
}
