use cozy_chess::{get_bishop_moves, get_rook_moves, Board, Color, File, Piece, Rank, Square};

use crate::batch::EntryFeatureWriter;

use super::InputFeatureSet;

pub struct Ice4InputFeatures;

const PAWN_PST_OFFSET: usize = 0;
const KNIGHT_PST_OFFSET: usize = PAWN_PST_OFFSET + 64;
const BISHOP_PST_OFFSET: usize = KNIGHT_PST_OFFSET + 32;
const ROOK_PST_OFFSET: usize = BISHOP_PST_OFFSET + 32;
const QUEEN_PST_OFFSET: usize = ROOK_PST_OFFSET + 32;
const KING_PST_OFFSET: usize = QUEEN_PST_OFFSET + 32;
const PASSED_PAWN_PST_OFFSET: usize = KING_PST_OFFSET + 64;
const FEATURES: usize = PASSED_PAWN_PST_OFFSET + 64;

const PIECE_PST_OFFSETS: [usize; 6] = [
    PAWN_PST_OFFSET,
    KNIGHT_PST_OFFSET,
    BISHOP_PST_OFFSET,
    ROOK_PST_OFFSET,
    QUEEN_PST_OFFSET,
    KING_PST_OFFSET,
];

const HORIZONTALLY_MIRRORED: [bool; 6] = [false, true, true, true, true, false];

impl InputFeatureSet for Ice4InputFeatures {
    const MAX_FEATURES: usize = 48;
    const INDICES_PER_FEATURE: usize = 2;
    const TENSORS_PER_BOARD: usize = 1;

    fn add_features(board: Board, mut entry: EntryFeatureWriter) {
        let phase = (board.pieces(Piece::Knight).len()
            + board.pieces(Piece::Bishop).len()
            + 2 * board.pieces(Piece::Rook).len()
            + 4 * board.pieces(Piece::Queen).len()) as f32
            / 24.0;

        let mut features = [0i8; FEATURES];

        for &piece in &Piece::ALL {
            for square in board.pieces(piece) & board.colors(Color::White) {
                let square = match HORIZONTALLY_MIRRORED[piece as usize] {
                    true => hm_feature(square),
                    false => square as usize,
                };
                features[PIECE_PST_OFFSETS[piece as usize] + square] += 1;
            }
            for square in board.pieces(piece) & board.colors(Color::Black) {
                let square = match HORIZONTALLY_MIRRORED[piece as usize] {
                    true => hm_feature(square.flip_rank()),
                    false => square.flip_rank() as usize,
                };
                features[PIECE_PST_OFFSETS[piece as usize] + square] -= 1;
            }
        }

        for &color in &Color::ALL {
            for square in board.pieces(Piece::Pawn) & board.colors(color) {
                let mut passer_mask = square.file().adjacent() | square.file().bitboard();
                match color {
                    Color::White => {
                        for r in 0..=square.rank() as usize {
                            passer_mask &= !Rank::index(r).bitboard();
                        }
                    }
                    Color::Black => {
                        for r in square.rank() as usize + 1..8 {
                            passer_mask &= !Rank::index(r).bitboard();
                        }
                    }
                }

                if !passer_mask.is_disjoint(board.colored_pieces(!color, Piece::Pawn)) {
                    continue;
                }

                let (sq, inc) = match color {
                    Color::White => (square as usize, 1),
                    Color::Black => (square.flip_rank() as usize, -1),
                };
                features[PASSED_PAWN_PST_OFFSET + sq] += inc;
            }
        }

        for (i, &v) in features.iter().enumerate().filter(|&(_, &v)| v != 0) {
            entry.add_feature(0, i as i64, v as f32 * phase);
            entry.add_feature(0, (i + FEATURES) as i64, v as f32 * (1.0 - phase));
        }
    }
}

fn hm_feature(square: Square) -> usize {
    let square = match square.file() > File::D {
        true => square.flip_file(),
        false => square,
    };
    square.rank() as usize * 4 + square.file() as usize
}