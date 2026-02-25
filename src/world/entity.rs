#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PreyId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PredatorId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LineageId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position {
    pub x: u32,
    pub y: u32,
}

impl Position {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &Position) -> f32 {
        let dx = self.x as f32 - other.x as f32;
        let dy = self.y as f32 - other.y as f32;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn direction_to(&self, other: &Position) -> (f32, f32) {
        let dx = other.x as f32 - self.x as f32;
        let dy = other.y as f32 - self.y as f32;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 0.001 {
            (0.0, 0.0)
        } else {
            (dx / dist, dy / dist)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    pub fn dx(&self) -> i32 {
        match self {
            Direction::East => 1,
            Direction::West => -1,
            _ => 0,
        }
    }

    pub fn dy(&self) -> i32 {
        match self {
            Direction::South => 1,
            Direction::North => -1,
            _ => 0,
        }
    }
}
