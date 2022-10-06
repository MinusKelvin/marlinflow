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
    recv: Receiver<Box<Batch>>,
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
        let (send, recv) = sync_channel(BUFFERED_BATCHES);
        std::thread::spawn(move || {
            dataloader_thread(send, file, feature_format, batch_size)
        });
        Ok(Self { recv, dataset_size })
    }

    pub fn dataset_size(&self) -> u64 {
        self.dataset_size
    }

    pub fn next_batch(&mut self) -> Option<Box<Batch>> {
        self.recv.recv().ok()
    }
}

fn dataloader_thread(
    send: SyncSender<Box<Batch>>,
    mut file: File,
    feature_format: InputFeatureSetType,
    batch_size: usize,
) {
    let mut board_buffer = vec![PackedBoard::zeroed(); batch_size * BUFFERED_BATCHES];
    let mut batches = vec![];
    loop {
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

        boards
            .par_chunks(batch_size)
            .map(|boards| match feature_format {
                InputFeatureSetType::Board768 => process::<Board768>(batch_size, boards),
                InputFeatureSetType::HalfKp => process::<HalfKp>(batch_size, boards),
                InputFeatureSetType::HalfKa => process::<HalfKa>(batch_size, boards),
                InputFeatureSetType::Board768Cuda => process::<Board768Cuda>(batch_size, boards),
                InputFeatureSetType::HalfKpCuda => process::<HalfKpCuda>(batch_size, boards),
                InputFeatureSetType::HalfKaCuda => process::<HalfKaCuda>(batch_size, boards),
            })
            .collect_into_vec(&mut batches);

        for batch in batches.drain(..) {
            if send.send(batch).is_err() {
                break;
            }
        }
    }
}

fn process<F: InputFeatureSet>(
    batch_size: usize,
    boards: &[PackedBoard],
) -> Box<Batch> {
    let mut batch = Box::new(Batch::new(
        batch_size,
        F::MAX_FEATURES,
        F::INDICES_PER_FEATURE,
    ));

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

    batch
}
