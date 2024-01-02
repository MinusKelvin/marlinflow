use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_pawn_quiets,
    get_rook_moves, BitBoard, Board, Color, File, Piece, Rank, Square,
};

use crate::batch::EntryFeatureWriter;

use super::InputFeatureSet;

pub struct Ice4InputFeatures;

pub const ICE4_FEATURE_COUNT: usize = TOTAL_FEATURES;

macro_rules! offsets {
    ($name:ident: $($rest:tt)*) => {
        const $name: usize = 0;
        offsets!([] $($rest)*);
    };
    ([$($val:literal)*] $size:literal; $next:ident : $($rest:tt)*) => {
        const $next: usize = $($val +)* $size;
        offsets!([$($val)* $size] $($rest)*);
    };
    ([$($val:literal)*] $size:literal;) => {
        const TOTAL_FEATURES: usize = $($val +)* $size;
    };
}

offsets! {
    PAWN_PST: 48;
    KNIGHT_PST: 16;
    KNIGHT_QUADRANT: 3;
    BISHOP_PST: 16;
    BISHOP_QUADRANT: 3;
    ROOK_PST: 16;
    ROOK_QUADRANT: 3;
    QUEEN_PST: 16;
    QUEEN_QUADRANT: 3;
    KING_PST: 16;
    PASSED_PAWN_PST: 48;
    BISHOP_PAIR: 1;
    DOUBLED_PAWN: 8;
    TEMPO: 1;
    ISOLATED_PAWN: 1;
    SINGLE_PROTECTED_PAWN: 1;
    DOUBLE_PROTECTED_PAWN: 1;
    ROOK_ON_OPEN_FILE: 1;
    ROOK_ON_SEMIOPEN_FILE: 1;
    SHIELD_PAWNS: 4;
    KING_ON_OPEN_FILE: 1;
    KING_ON_SEMIOPEN_FILE: 1;
    MOBILITY: 6;
    KING_RING_ATTACKS: 1;
}

const PIECE_PST_OFFSETS: [usize; 6] = [
    PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST, QUEEN_PST, KING_PST,
];
const PIECE_QUAD_OFFSETS: [usize; 6] = [
    0,
    KNIGHT_QUADRANT,
    BISHOP_QUADRANT,
    ROOK_QUADRANT,
    QUEEN_QUADRANT,
    0,
];

impl InputFeatureSet for Ice4InputFeatures {
    const MAX_FEATURES: usize = 64;
    const INDICES_PER_FEATURE: usize = 2;
    const TENSORS_PER_BOARD: usize = 2;

    fn add_features(board: Board, mut entry: EntryFeatureWriter) {
        let phase = (board.pieces(Piece::Knight).len()
            + board.pieces(Piece::Bishop).len()
            + 2 * board.pieces(Piece::Rook).len()
            + 4 * board.pieces(Piece::Queen).len()) as f32
            / 24.0;

        let mut features = [0i8; TOTAL_FEATURES];

        for &piece in &Piece::ALL {
            for unflipped_square in board.pieces(piece) {
                let color = board.color_on(unflipped_square).unwrap();
                let (square, inc) = match color {
                    Color::White => (unflipped_square, 1),
                    Color::Black => (unflipped_square.flip_rank(), -1),
                };

                if piece == Piece::Rook {
                    let file = square.file().bitboard();
                    if board.pieces(Piece::Pawn).is_disjoint(file) {
                        features[ROOK_ON_OPEN_FILE] += inc;
                    } else if board.colored_pieces(color, Piece::Pawn).is_disjoint(file) {
                        features[ROOK_ON_SEMIOPEN_FILE] += inc;
                    }
                }

                if piece == Piece::King {
                    let file = square.file().bitboard();
                    if board.pieces(Piece::Pawn).is_disjoint(file) {
                        features[KING_ON_OPEN_FILE] += inc;
                    } else if board.colored_pieces(color, Piece::Pawn).is_disjoint(file) {
                        features[KING_ON_SEMIOPEN_FILE] += inc;
                    }
                }

                let square = match piece {
                    Piece::Knight | Piece::Bishop | Piece::Rook | Piece::Queen => {
                        let quad = (square.file() > File::D) as usize * 2
                            + (square.rank() > Rank::Fourth) as usize;
                        if quad != 0 {
                            features[PIECE_QUAD_OFFSETS[piece as usize] + quad - 1] += inc;
                        }
                        hm_feature(square)
                    }
                    Piece::King => square.rank() as usize / 2 * 4 + square.file() as usize / 2,
                    Piece::Pawn => match board.king(color).file() > File::D {
                        true => square.flip_file() as usize - 8,
                        false => square as usize - 8,
                    },
                };
                features[PIECE_PST_OFFSETS[piece as usize] + square] += inc;

                let mob = match piece {
                    Piece::Pawn => {
                        get_pawn_quiets(unflipped_square, color, board.occupied())
                            | (get_pawn_attacks(unflipped_square, color) & board.colors(!color))
                    }
                    Piece::Knight => get_knight_moves(unflipped_square),
                    Piece::Bishop => get_bishop_moves(unflipped_square, board.occupied()),
                    Piece::Rook => get_rook_moves(unflipped_square, board.occupied()),
                    Piece::Queen => {
                        get_bishop_moves(unflipped_square, board.occupied())
                            | get_rook_moves(unflipped_square, board.occupied())
                    }
                    Piece::King => get_king_moves(unflipped_square),
                };
                features[KING_RING_ATTACKS] +=
                    inc * (get_king_moves(board.king(!color)) & mob).len() as i8;
                features[MOBILITY + piece as usize] +=
                    inc * (mob & !board.colors(color)).len() as i8;
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

                let square = match board.king(color).file() > File::D {
                    true => square.flip_file(),
                    false => square,
                };

                let (sq, inc) = match color {
                    Color::White => (square as usize, 1),
                    Color::Black => (square.flip_rank() as usize, -1),
                };
                features[PASSED_PAWN_PST + sq - 8] += inc;
            }
        }

        let mut white_doubled_mask = board.colored_pieces(Color::White, Piece::Pawn).0 >> 8;
        let mut black_doubled_mask = board.colored_pieces(Color::Black, Piece::Pawn).0 << 8;
        for _ in 0..6 {
            white_doubled_mask |= white_doubled_mask >> 8;
            black_doubled_mask |= black_doubled_mask << 8;
        }
        for sq in board.colored_pieces(Color::White, Piece::Pawn) & BitBoard(white_doubled_mask) {
            features[DOUBLED_PAWN + sq.file() as usize] += 1;
        }
        for sq in board.colored_pieces(Color::Black, Piece::Pawn) & BitBoard(black_doubled_mask) {
            features[DOUBLED_PAWN + sq.file() as usize] -= 1;
        }

        let pawns = board.colored_pieces(Color::White, Piece::Pawn);
        let pawn_attacks_right = BitBoard((pawns & !File::A.bitboard()).0 << 7);
        let pawn_attacks_left = BitBoard((pawns & !File::H.bitboard()).0 << 9);
        features[SINGLE_PROTECTED_PAWN] +=
            ((pawn_attacks_left ^ pawn_attacks_right) & pawns).len() as i8;
        features[DOUBLE_PROTECTED_PAWN] +=
            ((pawn_attacks_left & pawn_attacks_right) & pawns).len() as i8;

        let pawns = board.colored_pieces(Color::Black, Piece::Pawn);
        let pawn_attacks_right = BitBoard((pawns & !File::A.bitboard()).0 >> 9);
        let pawn_attacks_left = BitBoard((pawns & !File::H.bitboard()).0 >> 7);
        features[SINGLE_PROTECTED_PAWN] -=
            ((pawn_attacks_left ^ pawn_attacks_right) & pawns).len() as i8;
        features[DOUBLE_PROTECTED_PAWN] -=
            ((pawn_attacks_left & pawn_attacks_right) & pawns).len() as i8;

        for color in Color::ALL {
            let inc = match color {
                Color::White => 1,
                Color::Black => -1,
            };
            if board.colored_pieces(color, Piece::Bishop).len() >= 2 {
                features[BISHOP_PAIR] += inc;
            }
            if color == board.side_to_move() {
                features[TEMPO] += inc;
            }

            for sq in board.colored_pieces(color, Piece::Pawn) {
                if sq
                    .file()
                    .adjacent()
                    .is_disjoint(board.colored_pieces(color, Piece::Pawn))
                {
                    features[ISOLATED_PAWN] += inc;
                }
            }

            let king = board.king(color);
            if king.rank() == Rank::First.relative_to(color) {
                let pawns = board.colored_pieces(color, Piece::Pawn);
                let mut shield_pawns = 0;
                for dx in -1..=1 {
                    for dy in 1..3 {
                        if let Some(sq) = king.try_offset(dx, dy * inc) {
                            if pawns.has(sq) {
                                shield_pawns += 1;
                                break;
                            }
                        }
                    }
                }
                features[SHIELD_PAWNS + shield_pawns] += inc;
            }
        }

        for (i, &v) in features.iter().enumerate().filter(|&(_, &v)| v != 0) {
            entry.add_feature(0, i as i64, v as f32);
        }

        entry.add_feature(1, 0, phase);
    }
}

fn hm_feature(square: Square) -> usize {
    let square = match square.file() > File::D {
        true => square.flip_file(),
        false => square,
    };
    let square = match square.rank() > Rank::Fourth {
        true => square.flip_rank(),
        false => square,
    };
    square.rank() as usize * 4 + square.file() as usize
}
