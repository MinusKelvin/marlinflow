use cozy_chess::{Board, Piece};

pub trait BucketingScheme {
    const BUCKET_COUNT: usize;

    fn bucket(board: &Board) -> i32;
}

pub struct NoBucketing;

impl BucketingScheme for NoBucketing {
    const BUCKET_COUNT: usize = 1;

    fn bucket(_board: &Board) -> i32 {
        0
    }
}

pub struct ModifiedMaterial;

impl BucketingScheme for ModifiedMaterial {
    const BUCKET_COUNT: usize = 16;

    fn bucket(board: &Board) -> i32 {
        let material = board.pieces(Piece::Pawn).len() as i32
            + 3 * board.pieces(Piece::Bishop).len() as i32
            + 3 * board.pieces(Piece::Knight).len() as i32
            + 5 * board.pieces(Piece::Rook).len() as i32
            + 8 * board.pieces(Piece::Queen).len() as i32;

        (material * 16 / 76).min(15)
    }
}