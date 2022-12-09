use cozy_chess::{Board, Color, Piece, Square, Rank, File};

use crate::batch::EntryFeatureWriter;

use super::InputFeatureSet;

pub struct Ice4Features;

impl InputFeatureSet for Ice4Features {
    const MAX_FEATURES: usize = 32;
    const INDICES_PER_FEATURE: usize = 2;
    const TENSORS_PER_BOARD: usize = 4;

    fn add_features(board: Board, mut entry: EntryFeatureWriter) {
        let stm = board.side_to_move();

        for &color in &Color::ALL {
            for &piece in &Piece::ALL {
                for square in board.pieces(piece) & board.colors(color) {
                    let stm_feature = board_feature(stm, color, piece, square);
                    let nstm_feature = board_feature(!stm, color, piece, square);
                    entry.add_feature(0, stm_feature as i64, 1.0);
                    entry.add_feature(1, nstm_feature as i64, 1.0);
                }
            }
        }

        let phase = (board.pieces(Piece::Knight).len()
            + board.pieces(Piece::Bishop).len()
            + 2 * board.pieces(Piece::Rook).len()
            + 4 * board.pieces(Piece::Queen).len()) as f32
            / 24.0;

        for &color in &Color::ALL {
            for &piece in &Piece::ALL {
                for square in board.pieces(piece) & board.colors(color) {
                    let tensor = match color == stm {
                        false => 2,
                        true => 3,
                    };
                    let mg_feature = pst_feature(color, piece, square, 0);
                    let eg_feature = pst_feature(color, piece, square, 1);
                    entry.add_feature(tensor, mg_feature as i64, phase);
                    entry.add_feature(tensor, eg_feature as i64, 1.0 - phase);
                }
            }
        }
    }
}

fn pst_feature(color: Color, piece: Piece, square: Square, phase: usize) -> usize {
    let square = match color {
        Color::White => square,
        Color::Black => square.flip_rank(),
    };
    let mut index = 0;
    index = index * Piece::NUM + piece as usize;
    index = index * File::NUM + square.file() as usize;
    index = index * 2 + phase;
    index = index * Rank::NUM + square.rank() as usize;
    index
}

fn board_feature(perspective: Color, color: Color, piece: Piece, square: Square) -> usize {
    let (square, color) = match perspective {
        Color::White => (square, color),
        Color::Black => (square.flip_rank(), !color),
    };
    let mut index = 0;
    index = index * Piece::NUM + piece as usize;
    index = index * File::NUM + square.file() as usize;
    index = index * Color::NUM + color as usize;
    index = index * Rank::NUM + square.rank() as usize;
    index
}
