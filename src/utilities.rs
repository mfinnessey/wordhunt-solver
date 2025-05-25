use crate::letter::{translate_word, Letter};
use std::fs::File;
use std::io::{self, BufRead};
use trie_rs::{Trie, TrieBuilder};

pub mod snapshot_utilities;
pub mod test_utilities;

/// point values for word lengths from 0 to 16 (removing the factor of 100)
pub const POINTS: [u8; 17] = [0, 0, 0, 1, 4, 8, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54];

pub const TILE_COUNT: usize = 16;

pub const ALPHABET_LENGTH: usize = 26;

/// number of letter combinations to evaluate at once - chosen essentially arbitrarily
pub const BATCH_SIZE: usize = 500;

/// the starting value of the iterator over all letter combinations
pub const ALL_A_FREQUENCIES: [u8; ALPHABET_LENGTH] = [
    TILE_COUNT as u8,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
];

/// create a trie from the given wordlist filepath
pub fn create_trie(word_list_file_path: &str) -> (Trie<Letter>, u32) {
    let mut word_count = 0;
    // read from file
    if let Ok(file) = File::open(word_list_file_path) {
        let lines = io::BufReader::new(file).lines();

        // build trie of letters
        let mut builder: TrieBuilder<Letter> = TrieBuilder::new();
        for line in lines.map_while(Result::ok) {
            match translate_word(&line) {
                Ok(word) => {
                    builder.push(word);
                    word_count += 1;
                }
                // fail brutally with invalid word-lists
                Err(e) => panic!("Unable to process word {} because {e:?}", line),
            }
        }
        (builder.build(), word_count)
    } else {
        panic!("Could not open specified file!");
    }
}

/// create a trie from the given wordlist filepath
pub fn create_word_vector(word_list_file_path: &str) -> (Vec<[u8; ALPHABET_LENGTH]>, u32) {
    let mut results = Vec::new();
    // read from file
    if let Ok(file) = File::open(word_list_file_path) {
        let lines = io::BufReader::new(file).lines();

        // build up the vector
        for line in lines.map_while(Result::ok) {
            match translate_word(&line) {
                Ok(word) => {
                    let mut freqs = [0; ALPHABET_LENGTH];
                    for letter in word {
                        freqs[<Letter as Into<usize>>::into(letter)] += 1;
                    }
                    results.push(freqs);
                }
                // fail brutally with invalid word-lists
                Err(e) => panic!("unable to process word {} because {e:?}", line),
            }
        }
        let len: u32 = results
            .len()
            .try_into()
            .expect("word list too long - are you using a correct word list?");
        (results, len)
    } else {
        panic!("could not open specified file!");
    }
}

/// create a trie from the given wordlist filepath
pub fn create_word_vector_with_scores(
    word_list_file_path: &str,
) -> (Vec<([u8; ALPHABET_LENGTH], u8)>, u32) {
    let mut results = Vec::new();
    // read from file
    if let Ok(file) = File::open(word_list_file_path) {
        let lines = io::BufReader::new(file).lines();

        // build up the vector
        for line in lines.map_while(Result::ok) {
            match translate_word(&line) {
                Ok(word) => {
                    let mut freqs = [0; ALPHABET_LENGTH];
                    for letter in word {
                        freqs[<Letter as Into<usize>>::into(letter)] += 1;
                    }
                    // sum will be <= 16 and thus not overflow
                    results.push((freqs, POINTS[<u8 as Into<usize>>::into(freqs.iter().sum())]));
                }
                // fail brutally with invalid word-lists
                Err(e) => panic!("unable to process word {} because {e:?}", line),
            }
        }
        let len: u32 = results
            .len()
            .try_into()
            .expect("word list too long - are you using a correct word list?");
        (results, len)
    } else {
        panic!("could not open specified file!");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const WORDLIST_FILE_PATH: &str = "tests/wordlist.txt";
    const INVALID_WORDLIST_FILEPATH: &str = "tests/invalid_wordlist.txt";
    const DNE_FILE_PATH: &str = "tests/DNE";

    #[test]
    fn test_create_trie() {
        let (trie, word_count) = create_trie(WORDLIST_FILE_PATH);

        assert_eq!(word_count, 3);

        // hardcode expected trie contents - there is a risk of erroneous divergence
        // here, but we avoid the risk of replicating errors in I/O.
        const EXPECTED_WORDS: [&str; 3] = ["ALPHABET", "CAROLINA", "ZOOLOGICAL"];
        let expected: HashSet<Vec<Letter>> = HashSet::from_iter(
            EXPECTED_WORDS
                .iter()
                .map(|word| translate_word(word).unwrap()),
        );
        let trie_contents: HashSet<Vec<Letter>> = HashSet::from_iter(trie.iter());
        assert_eq!(expected.symmetric_difference(&trie_contents).count(), 0);
    }

    #[test]
    #[should_panic(
        expected = "Unable to process word 1 because \"Could not decode character. Only uppercase English letters are accepted.\""
    )]
    fn test_create_trie_invalid_word() {
        create_trie(INVALID_WORDLIST_FILEPATH);
    }

    #[test]
    #[should_panic(expected = "Could not open specified file!")]
    fn test_create_trie_file_dne() {
        create_trie(DNE_FILE_PATH);
    }

    #[test]
    fn test_create_word_vector() {
        let (word_vector, word_count) = create_word_vector(WORDLIST_FILE_PATH);

        assert_eq!(word_count, 3);

        // hardcode expected vector contents - there is a risk of erroneous divergence
        // here, but we avoid the risk of replicating errors in I/O.
        // words are ["ALPHABET", "CAROLINA", "ZOOLOGICAL"]
        // translated to frequencies by hand
        let expected_vec: Vec<[u8; ALPHABET_LENGTH]> = vec![
            [
                2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            ],
            [
                2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            [
                1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            ],
        ];

        assert_eq!(expected_vec, word_vector);
    }

    #[test]
    #[should_panic(
        expected = "unable to process word 1 because \"Could not decode character. Only uppercase English letters are accepted.\""
    )]
    fn test_create_word_vector_invalid_word() {
        create_word_vector(INVALID_WORDLIST_FILEPATH);
    }

    #[test]
    #[should_panic(expected = "could not open specified file!")]
    fn test_create_word_vector_file_dne() {
        const DNE_FILE_PATH: &str = "tests/DNE";
        create_word_vector(DNE_FILE_PATH);
    }

    #[test]
    fn test_create_word_vector_with_scores() {
        let (word_vector_with_scores, word_count) =
            create_word_vector_with_scores(WORDLIST_FILE_PATH);

        assert_eq!(word_count, 3);

        // hardcode expected vector contents - there is a risk of erroneous divergence
        // here, but we avoid the risk of replicating errors in I/O.
        // words are ["ALPHABET", "CAROLINA", "ZOOLOGICAL"]
        // translated to frequencies by hand

        let expected_vec: Vec<([u8; ALPHABET_LENGTH], u8)> = vec![
            (
                [
                    2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                ],
                POINTS[8],
            ),
            (
                [
                    2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                POINTS[8],
            ),
            (
                [
                    1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                ],
                POINTS[10],
            ),
        ];

        assert_eq!(expected_vec, word_vector_with_scores);
    }

    #[test]
    #[should_panic(
        expected = "unable to process word 1 because \"Could not decode character. Only uppercase English letters are accepted.\""
    )]
    fn test_create_word_vector_with_scores_invalid_word() {
        create_word_vector_with_scores(INVALID_WORDLIST_FILEPATH);
    }

    #[test]
    #[should_panic(expected = "could not open specified file!")]
    fn test_create_word_vector_with_scoresfile_dne() {
        const DNE_FILE_PATH: &str = "tests/DNE";
        create_word_vector_with_scores(DNE_FILE_PATH);
    }
}
