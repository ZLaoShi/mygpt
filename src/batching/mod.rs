use burn::{prelude::Backend, tensor::{Int, Tensor, TensorData}};

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

