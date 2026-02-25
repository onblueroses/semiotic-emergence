use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Terrain {
    Open,
    Grass,
    Tree,
    Rock,
    Water,
    Bush,
}

impl Terrain {
    pub fn is_passable(&self) -> bool {
        !matches!(self, Terrain::Water)
    }

    pub fn ascii_char(&self) -> char {
        match self {
            Terrain::Open => '.',
            Terrain::Grass => ',',
            Terrain::Tree => 'T',
            Terrain::Rock => 'R',
            Terrain::Water => '~',
            Terrain::Bush => '#',
        }
    }
}
