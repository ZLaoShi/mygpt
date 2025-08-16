use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};

use crate::batching::{BatchType, Batcher};
use crate::genrating::genrate;
use crate::modeling::BigramModel;
use crate::tokenizing::Tokenizer;
use crate::training::train;

pub mod tokenizing;
pub mod batching;
pub mod modeling;
pub mod training;
pub mod genrating;


fn main() {
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();

    let text: &'static str = include_str!("input.txt");
    println!("text chars len: {}", text.chars().count());

    let tokenizer = Tokenizer::from_text(text);
    println!("vocab size: {}", tokenizer.vocab_size());
    println!("{}", tokenizer.decode(&tokenizer.encode("Hello world!")));

    let batcher = Batcher::from_tokens(tokenizer.encode(text));
    let (x,y) = batcher.batch::<MyBackend>(BatchType::Train, &device);

    println!("{x}");
    println!("{y}");

    let model = BigramModel::<MyBackend>::new(tokenizer.vocab_size(), &device);
    let logits = model.forward(x.clone());
    println!("{logits}");

    let loss = model.loss(x, y, &device);
    println!("{loss}");

    let new_tokens = genrate(&model, tokenizer.encode("A"), 100, &device);
    println!("{}", tokenizer.decode(&new_tokens));

    let trained_model = train::<MyAutodiffBackend>
    (
        tokenizer.vocab_size(), 
        &batcher, 
        &device
    );

    let new_tokens = genrate(&trained_model, tokenizer.encode("A"), 100, &device);
    println!("{}", tokenizer.decode(&new_tokens));
}
