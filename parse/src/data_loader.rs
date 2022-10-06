use std::io::{Seek, SeekFrom};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::{fs::File, io::Read, path::Path};

use bytemuck::Zeroable;
use cozy_chess::Color;
use marlinformat::PackedBoard;
use rayon::prelude::*;

use crate::batch::Batch;
use crate::input_features::*;

const BUFFERED_BATCHES: usize = 64;

pub struct BatchReader {
    recv: Receiver<Vec<Batch>>,
    reuse: SyncSender<Vec<Batch>>,
    batches: Vec<Batch>,
    index: usize,
    dataset_size: u64,
}

impl BatchReader {
    pub fn new(
        path: &Path,
        feature_format: InputFeatureSetType,
        batch_size: usize,
    ) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let dataset_size = file.seek(SeekFrom::End(0))? / std::mem::size_of::<PackedBoard>() as u64;
        file.seek(SeekFrom::Start(0))?;
        let (send, recv) = sync_channel(2);
        let (reuse, reuse_recv) = sync_channel(2);
        std::thread::spawn(move || {
            dataloader_thread(send, reuse_recv, file, feature_format, batch_size)
        });
        let _ = reuse.send(batch_buffer(feature_format, batch_size));
        Ok(Self {
            recv,
            reuse,
            dataset_size,
            batches: batch_buffer(feature_format, batch_size),
            index: 0,
        })
    }

    pub fn dataset_size(&self) -> u64 {
        self.dataset_size
    }

    pub fn next_batch(&mut self) -> Option<&mut Batch> {
        loop {
            while self.index < self.batches.len() {
                let i = self.index;
                self.index += 1;
                if self.batches[i].len() > 0 {
                    return Some(&mut self.batches[i]);
                }
            }
            let _ = self.reuse.send(std::mem::take(&mut self.batches));
            self.batches = self.recv.recv().ok()?;
            self.index = 0;
        }
    }
}

fn dataloader_thread(
    send: SyncSender<Vec<Batch>>,
    reuse: Receiver<Vec<Batch>>,
    mut file: File,
    feature_format: InputFeatureSetType,
    batch_size: usize,
) {
    let mut board_buffer = vec![PackedBoard::zeroed(); batch_size * BUFFERED_BATCHES];
    for mut batches in reuse {
        let buffer = bytemuck::cast_slice_mut(&mut board_buffer);
        let mut bytes_read = 0;
        loop {
            match file.read(&mut buffer[bytes_read..]) {
                Ok(0) => break,
                Ok(some) => bytes_read += some,
                Err(_) => return,
            }
        }
        let elems = bytes_read / std::mem::size_of::<PackedBoard>();
        if elems == 0 {
            return;
        }
        let boards = &board_buffer[..elems];

        for batch in &mut batches {
            batch.clear();
        }

        boards
            .par_chunks(batch_size)
            .zip(batches.par_iter_mut())
            .for_each(|(boards, batch)| match feature_format {
                InputFeatureSetType::Board768 => process::<Board768>(batch, boards),
                InputFeatureSetType::HalfKp => process::<HalfKp>(batch, boards),
                InputFeatureSetType::HalfKa => process::<HalfKa>(batch, boards),
                InputFeatureSetType::Board768Cuda => process::<Board768Cuda>(batch, boards),
                InputFeatureSetType::HalfKpCuda => process::<HalfKpCuda>(batch, boards),
                InputFeatureSetType::HalfKaCuda => process::<HalfKaCuda>(batch, boards),
            });

        if send.send(batches).is_err() {
            break;
        }
    }
}

fn process<F: InputFeatureSet>(batch: &mut Batch, boards: &[PackedBoard]) {
    for packed in boards {
        (|| {
            let (board, cp, wdl, _) = packed.unpack()?;
            let cp = cp as f32;
            let wdl = wdl as f32 / 2.0;

            if cp.abs() > 3000.0 {
                return None;
            }

            let (cp, wdl) = match board.side_to_move() {
                Color::White => (cp, wdl),
                Color::Black => (-cp, 1.0 - wdl),
            };

            let entry = batch.make_entry(cp, wdl);
            F::add_features(board, entry);

            Some(())
        })();
    }
}

fn batch_buffer(feature_format: InputFeatureSetType, batch_size: usize) -> Vec<Batch> {
    let mut v = vec![];
    v.resize_with(BUFFERED_BATCHES, || {
        Batch::new(
            batch_size,
            feature_format.max_features(),
            feature_format.indices_per_feature(),
        )
    });
    v
}
