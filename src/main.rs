use std::collections::{BTreeSet, HashMap};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor, TensorData};

pub struct Tokenizer{
    stoi:HashMap<char, i32>,
    itos:HashMap<i32, char>,
}

impl Tokenizer {
    pub fn from_text(text:&str) -> Self{
        let chars = text
            .chars()
            .collect::<BTreeSet<char>>();

        let stoi = chars
            .iter()
            .enumerate()
            .map(|(i, ch)|(*ch, i as i32))
            .collect::<HashMap<char, i32>>();

        let itos = chars
            .iter()
            .enumerate()
            .map(|(i, ch)|(i as i32, *ch))
            .collect::<HashMap<i32, char>>(); 

        Self { stoi, itos}
    }

    pub fn encode(&self, text:&str) -> Vec<i32> {
        text.chars()
            .map(|ch| *self.stoi.get(&ch).unwrap())
            .collect()
    }

    pub fn decode(&self, tokens: &[i32]) -> String {
        tokens.iter()
            .map(|i| *self.itos.get(&i).unwrap())
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }
}

const BATCH_SIZE:usize = 4;
const BLOCK_SIZE:usize = 8;

pub struct  Batcher {
    train_data:Vec<i32>,
    test_data:Vec<i32>
}

pub enum BatchType {
    Train,
    Test
}

impl Batcher {
    pub fn from_tokens(tokens: Vec<i32>) -> Self {
        let mut train_data = tokens;

        let split_at = (train_data.len() as f64 * 0.9) as usize;
        let test_data = train_data.split_off(split_at);

        println!("{}", train_data.len());
        println!("{}", test_data.len());

        Self { train_data, test_data }
    }

    pub fn batch<B: Backend>(&self, typ: BatchType, device:&B::Device) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let data = match typ {
            BatchType::Train => &self.train_data,
            BatchType::Test => &self.test_data
        };

        let mut rng = rand::rng();
        let ix = rand::seq::index::sample(
            &mut rng,
            data.len() - BLOCK_SIZE - 1,
            BATCH_SIZE);

        let x = ix.iter()
            .map(|i| data[i..i+BLOCK_SIZE].to_vec())
            .flatten()
            .collect();

        let y = ix.iter()
            .map(|i| data[i + 1..i + 1 + BLOCK_SIZE].to_vec())
            .flatten()
            .collect();

        let xd = TensorData::new(x, [BATCH_SIZE ,BLOCK_SIZE]); 
        let yd = TensorData::new(y, [BATCH_SIZE ,BLOCK_SIZE]); 

        let xt = Tensor::from_data(xd, device);
        let yt = Tensor::from_data(yd, device);

        (xt, yt)
    }
}

fn main() {
    type MyBackend = Wgpu;
    let device = WgpuDevice::default();

    let text = include_str!("input.txt");
    println!("text chars len: {}", text.chars().count());

    let tokenizer = Tokenizer::from_text(text);
    println!("vocab size: {}", tokenizer.vocab_size());
    println!("{}", tokenizer.decode(&tokenizer.encode("Hello world!")));

    let batcher = Batcher::from_tokens(tokenizer.encode(text));
    let (x,y) = batcher.batch::<MyBackend>(BatchType::Train, &device);

    println!("{x}");
    println!("{y}");
    
    // x = [[0, 1, 2, 3, 4, 5, 6, 7],[],[],[]]
    // y = [[1, 2, 3, 4, 5, 6, 7, 8],[],[],[]]


}
