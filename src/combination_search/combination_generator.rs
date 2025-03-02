use crate::letter_combination::LetterCombination;
use crate::utilities::{ALPHABET_LENGTH, TILE_COUNT};

/// generates combinations with replacement for a specified C(N, R) as frequency
/// counts for the N options.
/// N is the number of elements to select from
/// R is the number of selections to make.
// this is implemented abstractly here primarily for practical unit testing.
// substantial inspiration taken from https://github.com/olivercalder/combinatorial/
pub struct SequentialCombinationGenerator<const N: usize, const R: usize> {
    /// indices into a (conceptual) array consisting of the elements that we select from
    element_indices: [usize; R],
    /// set iff all combinations (with replacement) have been iterated over
    completed: bool,
}

impl<const N: usize, const R: usize> SequentialCombinationGenerator<N, R> {
    pub fn new(indices: [usize; R]) -> Self {
        // verify that indices array is non-decreasing.
        for i in 1..R {
            if indices[i] < indices[i - 1] {
                panic!("Attempted to create sequential combination generator with decreasing element_indices array. Indices {} and {} are [{}, {}].",
		       i - 1, i, indices[i - 1], indices[i]);
            }
        }
        Self {
            element_indices: indices,
            completed: false,
        }
    }

    /// create the next set of indices from the current set of indices
    fn advance_indices(&mut self) {
        // iteration scheme generates all non-decreasing indices arrays
        for i in (0..R).rev() {
            // look for the first index that is not maxed out
            if self.element_indices[i] < N - 1 {
                // bump all subsequent indices to the value of the first non-maxed index plus one
                let next_index = self.element_indices[i] + 1;
                for j in i..R {
                    self.element_indices[j] = next_index;
                }
                return;
            }
        }

        self.completed = true;
    }
}

impl<const N: usize, const R: usize> Iterator for SequentialCombinationGenerator<N, R> {
    type Item = [u8; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.completed {
            return None;
        }

        // map indices into (conceptual) array of elements to the frequency
        // counts of those elements
        let mut frequencies = [0u8; N];
        for index in self.element_indices {
            frequencies[index] += 1;
        }

        self.advance_indices();

        Some(frequencies)
    }
}

/// generates combinations of letters from the given starting_Frequencies "onwards" in sequence.
/// the sequence is defined using the iteration scheme of SequentialCombinationGenerator above.
pub struct SequentialLetterCombinationGenerator {
    generator: SequentialCombinationGenerator<ALPHABET_LENGTH, TILE_COUNT>,
}

impl SequentialLetterCombinationGenerator {
    pub fn new(starting_frequencies: LetterCombination) -> Self {
        let indices = letter_combination_to_element_indices(starting_frequencies);
        Self {
            generator: SequentialCombinationGenerator::new(indices),
        }
    }
}

/// map letter frequencies into indices into an (conceptual) array of the letters
pub fn letter_combination_to_element_indices(lc: LetterCombination) -> [usize; TILE_COUNT] {
    let mut indices = [0usize; TILE_COUNT];
    let mut indices_idx = 0;
    for (letter_idx, mut letter_frequency) in
        <[u8; ALPHABET_LENGTH]>::from(lc).into_iter().enumerate()
    {
        while letter_frequency > 0 {
            indices[indices_idx] = letter_idx;
            indices_idx += 1;
            letter_frequency -= 1;
        }
    }
    indices
}

impl Iterator for SequentialLetterCombinationGenerator {
    type Item = LetterCombination;

    fn next(&mut self) -> Option<Self::Item> {
        self.generator.next().map(|frequencies| frequencies.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::letter::translate_word;
    use std::collections::HashSet;
    use trie_rs::TrieBuilder;

    // AAAABBBBCCCCEEZZ
    const FREQUENCIES: [u8; ALPHABET_LENGTH] = [
        4, 4, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    ];

    #[test]
    fn test_sequential_combination_generator() {
        // test with 5 choose 3 without loss of generality
        let initial_frequencies = [0, 0, 0];
        let generator: SequentialCombinationGenerator<5, 3> =
            SequentialCombinationGenerator::new(initial_frequencies);

        const EXPECTED_FREQUENCIES: [[u8; 5]; 35] = [
            [3, 0, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [2, 0, 1, 0, 0],
            [2, 0, 0, 1, 0],
            [2, 0, 0, 0, 1],
            [1, 2, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 0, 2, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 2, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 0, 2],
            [0, 3, 0, 0, 0],
            [0, 2, 1, 0, 0],
            [0, 2, 0, 1, 0],
            [0, 2, 0, 0, 1],
            [0, 1, 2, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 0, 2, 0],
            [0, 1, 0, 1, 1],
            [0, 1, 0, 0, 2],
            [0, 0, 3, 0, 0],
            [0, 0, 2, 1, 0],
            [0, 0, 2, 0, 1],
            [0, 0, 1, 2, 0],
            [0, 0, 1, 0, 2],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 3, 0],
            [0, 0, 0, 2, 1],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 3],
        ];

        let expected = HashSet::from(EXPECTED_FREQUENCIES);
        let actual: Vec<[u8; 5]> = generator.collect();

        // verify generation length and elements (generation order doesn't matter)
        assert_eq!(actual.len(), 35);
        assert_eq!(
            expected
                .symmetric_difference(&HashSet::from_iter(actual.into_iter()))
                .count(),
            0
        );
    }

    #[test]
    fn test_letter_combination_to_element_indices() {
        let lc = LetterCombination::new(FREQUENCIES);

        const EXPECTED: [usize; TILE_COUNT] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 25, 25];
        let actual = letter_combination_to_element_indices(lc);
        assert_eq!(actual, EXPECTED);
    }
}
