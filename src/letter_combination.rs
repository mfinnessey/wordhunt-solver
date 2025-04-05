use crate::utilities::{ALPHABET_LENGTH, TILE_COUNT};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Index, IndexMut};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LetterCombination {
    frequencies: [u8; ALPHABET_LENGTH],
}

impl LetterCombination {
    pub fn new(frequencies: [u8; ALPHABET_LENGTH]) -> Self {
        let combination_tile_count: usize = frequencies.iter().map(|x| *x as usize).sum();
        if combination_tile_count != TILE_COUNT {
            panic!(
                "Attempted to create invalid letter combination with {} tiles instead of {} tiles",
                combination_tile_count, TILE_COUNT
            );
        }

        Self { frequencies }
    }
}

impl Ord for LetterCombination {
    fn cmp(&self, other: &Self) -> Ordering {
        for (freq1, freq2) in self.frequencies.iter().zip(other.frequencies.iter()) {
            if freq1 > freq2 {
                return Ordering::Less;
            }
            if freq2 > freq1 {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }
}

impl PartialOrd for LetterCombination {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/*impl PartialEq for LetterCombination {
    fn eq(&self, other: &Self) -> bool {
        self.frequencies == other.frequencies
    }
}*/

impl Index<usize> for LetterCombination {
    type Output = u8;
    fn index(&self, idx: usize) -> &Self::Output {
        if idx > ALPHABET_LENGTH - 1 {
            panic!(
                "Index {} for LetterFrequencies is outside the [0, 25]!",
                idx
            )
        }
        &self.frequencies[idx]
    }
}

impl IndexMut<usize> for LetterCombination {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        if idx > ALPHABET_LENGTH - 1 {
            panic!(
                "Index {} for LetterFrequencies is outside the [0, 25]!",
                idx
            )
        }
        &mut self.frequencies[idx]
    }
}

/*impl Eq for LetterCombination {}*/

impl From<LetterCombination> for [u8; ALPHABET_LENGTH] {
    fn from(frequencies: LetterCombination) -> Self {
        frequencies.frequencies
    }
}

impl From<[u8; ALPHABET_LENGTH]> for LetterCombination {
    fn from(frequencies: [u8; ALPHABET_LENGTH]) -> Self {
        LetterCombination::new(frequencies)
    }
}

impl fmt::Display for LetterCombination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut letters = ['a'; TILE_COUNT];
        let mut letters_idx = 0;
        for (frequencies_idx, letter_frequency) in self.frequencies.iter().enumerate() {
            for _ in 0..*letter_frequency {
                letters[letters_idx] = (b'A' + frequencies_idx as u8) as char;
                letters_idx += 1;
            }
        }

        let concatenated: String = letters.iter().collect();
        write!(f, "{}", concatenated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FREQUENCIES: [u8; ALPHABET_LENGTH] = [
        4, 4, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    ];

    #[test]
    fn test_letter_combination_display() {
        let lc = LetterCombination::new(FREQUENCIES);

        const EXPECTED: &str = "AAAABBBBCCCCEEZZ";
        let actual = format!("{}", lc);
        assert_eq!(actual, EXPECTED);
    }
}
