use burn::{optim::{AdamWConfig, GradientsParams, Optimizer}, tensor::{backend::AutodiffBackend, cast::ToElement}};

use crate::{batching::{BatchType, Batcher}, modeling::BigramModel};


pub fn train<B:AutodiffBackend>(vocab_size: usize, batcher: &Batcher, device:&B::Device) -> BigramModel<B> {
    let mut model = BigramModel::<B>::new(vocab_size, device);
    let mut optimizer = AdamWConfig::new().init::<B, BigramModel<B>>();

    for i in 0..100000 {
        let (x, y) = batcher.batch::<B>(BatchType::Train, device);

        let loss = model.loss(x, y, device);
        
        if i % 1000 == 0 {
            let last_loss = loss.clone().into_scalar().to_f32();
            println!("loss: {last_loss}");
        }
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        model = optimizer.step(1e-3, model, grads);
    }

    model
}