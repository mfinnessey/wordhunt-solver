use crate::letter::{translate_word, Letter};
use std::fs::File;
use std::io::{self, BufRead};
use trie_rs::{Trie, TrieBuilder};

/// point values for word lengths from 0 to 16
pub const POINTS: [u32; 17] = [0, 0, 0, 1, 4, 8, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54];

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_create_trie() {
        const TEST_TRIE_FILE_PATH: &str = "tests/test_trie.txt";
        let (trie, word_count) = create_trie(TEST_TRIE_FILE_PATH);

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
        const TEST_TRIE_FAIL_FILE_PATH: &str = "tests/test_trie_fail.txt";
        create_trie(TEST_TRIE_FAIL_FILE_PATH);
    }

    #[test]
    #[should_panic(expected = "Could not open specified file!")]
    fn test_create_trie_file_dne() {
        const DNE_FILE_PATH: &str = "tests/DNE";
        create_trie(DNE_FILE_PATH);
    }
}
