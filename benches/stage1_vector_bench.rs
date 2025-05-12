use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use trie_rs::Trie;
use wordhunt_solver::letter::Letter;
use wordhunt_solver::utilities::{
    create_word_vector, create_word_vector_with_scores, ALPHABET_LENGTH,
};

use std::fmt::Formatter;
use wordhunt_solver::combination_search::bounding_functions::{
    combination_score_all_possible_words, combination_score_all_possible_words_with_scores,
};
use wordhunt_solver::letter_combination::LetterCombination;
use wordhunt_solver::utilities::test_utilities::read_random_combinations;

const WORDLIST_FILENAME: &str = "wordlist-actual.txt";

fn benchmark_stage1_search(c: &mut Criterion) {
    let word_vector = create_word_vector(WORDLIST_FILENAME);
    let word_vector_with_scores = create_word_vector_with_scores(WORDLIST_FILENAME);

    assert_eq!(
        word_vector.1, word_vector_with_scores.1,
        "dictionary data structures are not the same size!"
    );

    let letter_combinations = read_random_combinations("random_letter_combinations");

    let mut group = c.benchmark_group("stage 1 search algorithms");

    let word_freqs_input =
        WordFreqsDisplayWrapper::new((word_vector.0, letter_combinations.clone()));
    let word_freqs_with_scores_input =
        WordVectorWithScoresDisplayWrapper::new((word_vector_with_scores.0, letter_combinations));

    group.bench_with_input(
        BenchmarkId::new("trie", &word_freqs_with_scores_input),
        &word_freqs_with_scores_input,
        |b, i| {
            b.iter(|| {
                let combinations = &i.letter_combinations;
                let word_vec_with_scores = &i.word_vector_with_scores;
                for combination in combinations {
                    combination_score_all_possible_words_with_scores(
                        word_vec_with_scores,
                        *combination,
                    );
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
struct WordVectorWithScoresDisplayWrapper {
    pub word_vector_with_scores: Vec<([u8; ALPHABET_LENGTH], u8)>,
    pub letter_combinations: Vec<LetterCombination>,
}

impl WordVectorWithScoresDisplayWrapper {
    fn new(input: (Vec<([u8; ALPHABET_LENGTH], u8)>, Vec<LetterCombination>)) -> Self {
        Self {
            word_vector_with_scores: input.0,
            letter_combinations: input.1,
        }
    }
}

impl std::fmt::Display for WordVectorWithScoresDisplayWrapper {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "it's a word vector with scores and a bunch of letter combinations!"
        )
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

criterion_group!(stage1_search_vector_benches, benchmark_stage1_search);
criterion_main!(stage1_search_vector_benches);
