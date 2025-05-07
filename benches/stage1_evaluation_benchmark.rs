use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use trie_rs::Trie;
use wordhunt_solver::letter::Letter;
use wordhunt_solver::utilities::{create_trie, create_word_vector, ALPHABET_LENGTH};

use std::fmt::Formatter;
use wordhunt_solver::combination_search::bounding_functions::{
    combination_score_all_possible_trie_paths, combination_score_all_possible_words,
};
use wordhunt_solver::letter_combination::LetterCombination;
use wordhunt_solver::utilities::test_utilities::read_random_combinations;

const WORDLIST_FILENAME: &str = "wordlist-actual.txt";

fn benchmark_stage1_search(c: &mut Criterion) {
    let trie = create_trie(WORDLIST_FILENAME);
    let word_vector = create_word_vector(WORDLIST_FILENAME);

    assert_eq!(
        trie.1, word_vector.1,
        "dictionary data structures are not the same size!"
    );

    let letter_combinations = read_random_combinations("random_letter_combinations");

    let mut group = c.benchmark_group("stage 1 search algorithms");
    group.sample_size(20);

    let trie_input = TrieDisplayWrapper::new((trie.0, letter_combinations.clone()));
    let word_freqs_input = WordFreqsDisplayWrapper::new((word_vector.0, letter_combinations));

    group.bench_with_input(
        BenchmarkId::new("trie", &trie_input),
        &trie_input,
        |b, i| {
            b.iter(|| {
                let combinations = &i.letter_combinations;
                let trie = &i.trie;
                for combination in combinations {
                    combination_score_all_possible_trie_paths(trie, *combination);
                }
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("vector", &word_freqs_input),
        &word_freqs_input,
        |b, i| {
            b.iter(|| {
                let combinations = &i.letter_combinations;
                let word_freqs = &i.word_freqs;
                for combination in combinations {
                    combination_score_all_possible_words(word_freqs, *combination);
                }
            })
        },
    );
}

/// wrapper class to get the stupid display implementation
/// that criterion demands
struct TrieDisplayWrapper {
    pub trie: Trie<Letter>,
    pub letter_combinations: Vec<LetterCombination>,
}

impl TrieDisplayWrapper {
    fn new(input: (Trie<Letter>, Vec<LetterCombination>)) -> Self {
        Self {
            trie: input.0,
            letter_combinations: input.1,
        }
    }
}

impl std::fmt::Display for TrieDisplayWrapper {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "it's a trie and a bunch of letter combinations!")
    }
}

/// wrapper class to get the stupid display implementation
/// that criterion demands
struct WordFreqsDisplayWrapper {
    pub word_freqs: Vec<[u8; ALPHABET_LENGTH]>,
    pub letter_combinations: Vec<LetterCombination>,
}

impl WordFreqsDisplayWrapper {
    fn new(input: (Vec<[u8; ALPHABET_LENGTH]>, Vec<LetterCombination>)) -> Self {
        Self {
            word_freqs: input.0,
            letter_combinations: input.1,
        }
    }
}

impl std::fmt::Display for WordFreqsDisplayWrapper {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "it's a whole lot of letter combinations!!")
    }
}

criterion_group!(stage1_search_benches, benchmark_stage1_search);
criterion_main!(stage1_search_benches);
