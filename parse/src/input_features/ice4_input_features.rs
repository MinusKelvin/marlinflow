use cozy_chess::{Board, Color, File, Piece, Square, Rank};

use crate::batch::EntryFeatureWriter;

use super::InputFeatureSet;

pub struct Ice4InputFeatures;

impl InputFeatureSet for Ice4InputFeatures {
    const MAX_FEATURES: usize = 32;
    const INDICES_PER_FEATURE: &'static [usize] = &[2, 2, 2, 2, 3];

    fn add_features(board: Board, mut entry: EntryFeatureWriter) {
        let stm = board.side_to_move();

        let phase = (board.pieces(Piece::Knight).len()
            + board.pieces(Piece::Bishop).len()
            + 2 * board.pieces(Piece::Rook).len()
            + 4 * board.pieces(Piece::Queen).len()) as f32
            / 24.0;

        for &color in &Color::ALL {
            for &piece in &Piece::ALL {
                for square in board.pieces(piece) & board.colors(color) {
                    let tensor = match (square.file() >= File::E, color == stm) {
                        (false, true) => 0,
                        (true, true) => 1,
                        (false, false) => 2,
                        (true, false) => 3,
                    };
                    let feature = hm_pst_feature(color, piece, square);
                    entry.add_feature(tensor, &[feature as i64], phase);
                    entry.add_feature(tensor, &[feature as i64 + 192], 1.0 - phase);

                    let feature = file_feature(stm, color, piece, square.rank());
                    entry.add_feature(4, &[square.file() as i64, feature as i64], 1.0);
                }
            }
        }
    }
}

fn hm_pst_feature(color: Color, piece: Piece, square: Square) -> usize {
    let rank = match color {
        Color::White => square.rank(),
        Color::Black => square.rank().flip(),
    };
    let file = match square.file() >= File::E {
        true => square.file().flip(),
        false => square.file(),
    };
    let mut index = 0;
    index = index * Piece::NUM + piece as usize;
    index = index * Square::NUM / 2 + rank as usize * 4 + file as usize;
    index
}

fn file_feature(perspective: Color, color: Color, piece: Piece, rank: Rank) -> usize {
    let (rank, color) = match perspective {
        Color::White => (rank, color),
        Color::Black => (rank.flip(), !color),
    };
    let mut index = 0;
    index = index * Color::NUM + color as usize;
    index = index * Piece::NUM + piece as usize;
    index = index * Rank::NUM + rank as usize;
    index
}
