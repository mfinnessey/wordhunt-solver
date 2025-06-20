use crate::letter::Letter;
use crate::letter_combination::LetterCombination;
use crate::utilities::{ALPHABET_LENGTH, BATCH_SIZE, POINTS};
use std::collections::VecDeque;
use trie_rs::inc_search::Answer;
use trie_rs::Trie;

/// get the summed scores of all words of length <= 16
/// that can be built from a given combination of 16 letters represented
/// as frequency counts
// this is an upper bound on the score of any board that is a permutation of these letters
// this bound ignores all placement considerations.
pub fn combination_score_all_possible_trie_paths(
    dictionary: &Trie<Letter>,
    letter_frequencies: LetterCombination,
) -> u32 {
    // compute score by bfs through trie
    let mut score: u32 = 0;
    let mut queue = VecDeque::new();
    // convert into slice for iteration
    let letter_counts: [u8; ALPHABET_LENGTH] = letter_frequencies.into();

    // build inc search starting from each available letter
    for (i, count) in letter_counts.iter().enumerate() {
        if *count > 0 {
            let mut inc_search = dictionary.inc_search();
            if inc_search.query(&Letter::from(i)).is_some() {
                let mut new_counts = letter_counts;
                new_counts[i] -= 1;
                queue.push_back((inc_search, new_counts));
            }
        }
    }

    while let Some(cur) = queue.pop_front() {
        let cur_search = cur.0;
        let remaining_counts = cur.1;
        // build out queries to all potential successors
        for (i, count) in remaining_counts.iter().enumerate() {
            if *count > 0 {
                let mut new_search = cur_search.clone();
                match new_search.query(&Letter::from(i)) {
                    // continuation only
                    Some(Answer::Prefix) => {
                        let mut new_counts = remaining_counts;
                        new_counts[i] -= 1;
                        queue.push_back((new_search, new_counts));
                    }
                    // score and continuation
                    Some(Answer::PrefixAndMatch) => {
                        score += <u8 as Into<u32>>::into(POINTS[new_search.prefix_len()]);

                        let mut new_counts = remaining_counts;
                        new_counts[i] -= 1;
                        queue.push_back((new_search, new_counts));
                    }
                    // score only
                    Some(Answer::Match) => {
                        score += <u8 as Into<u32>>::into(POINTS[new_search.prefix_len()]);
                    }
                    // no score, no continuation
                    None => (),
                }
            }
        }
    }

    score
}

/// get the summed scores of all words of length <= 16
/// that can be built from a given combination of 16 letters represented
/// as frequency counts
// this is an upper bound on the score of any board that is a permutation of these letters
// this bound ignores all placement considerations.
pub fn combination_score_all_possible_words(
    word_list: &Vec<[u8; ALPHABET_LENGTH]>,
    letter_frequencies: LetterCombination,
) -> u32 {
    let mut score: u32 = 0;

    for word_freqs in word_list.iter() {
        let mut fits = true;
        for (word_freq, ref letter_freq) in word_freqs
            .iter()
            .zip(<[u8; ALPHABET_LENGTH]>::from(letter_frequencies))
        {
            if word_freq > letter_freq {
                fits = false;
                break;
            }
        }
        if fits {
            score += &POINTS[word_freqs.iter().sum::<u8>() as usize].into();
        }
    }

    score
}

pub fn combination_score_all_possible_words_with_scores(
    words: &Vec<([u8; ALPHABET_LENGTH], u8)>,
    letter_frequencies: LetterCombination,
) -> u32 {
    let mut score: u32 = 0;

    for (word_freqs, word_score) in words.iter() {
        let mut fits = true;
        for (word_freq, ref letter_freq) in word_freqs
            .iter()
            .zip(<[u8; ALPHABET_LENGTH]>::from(letter_frequencies))
        {
            if word_freq > letter_freq {
                fits = false;
                break;
            }
        }
        if fits {
            score += *word_score as u32;
        }
    }

    score
}

pub fn combination_score_all_possible_words_with_scores_tiled(
    words: &[([u8; ALPHABET_LENGTH], u8)],
    letter_combinations: &[LetterCombination; BATCH_SIZE],
) -> [u32; BATCH_SIZE] {
    let mut scores = [0; BATCH_SIZE];

    // need to reserve space for the scores array (16 KB)
    // sizeof each element is 26 * 8 + 8 = 216b
    // l1 data cache on zen5 is 48 KB / core while l2 is 1 MB / core
    // with smt, assume that each thread has access to half the cache
    // attempt to keep everything in l1 cache (should optimize this,
    // but that's another project ;)
    // 48 KB / 2 = 24 KB
    // 24 KB - 16 KB = 8 KB
    // 8 KB / 27B -> 296 items
    // round down to 290 to leave a little extra cache space
    const DICTIONARY_TILE_SIZE: usize = 290;
    let mut tile_base = 0;

    while tile_base + DICTIONARY_TILE_SIZE < words.len() {
        for (word_freqs, word_score) in words[tile_base..tile_base + DICTIONARY_TILE_SIZE].iter() {
            for (letter_frequencies, score) in letter_combinations.iter().zip(scores.iter_mut()) {
                let mut fits = true;
                for (word_freq, ref letter_freq) in word_freqs
                    .iter()
                    .zip(<[u8; ALPHABET_LENGTH]>::from(*letter_frequencies))
                {
                    if word_freq > letter_freq {
                        fits = false;
                        break;
                    }
                }
                if fits {
                    *score += *word_score as u32;
                }
            }
        }

        tile_base += DICTIONARY_TILE_SIZE;
    }

    // handle (potential) partial last tile
    for (word_freqs, word_score) in words[tile_base..].iter() {
        for (letter_frequencies, score) in letter_combinations.iter().zip(scores.iter_mut()) {
            let mut fits = true;
            for (word_freq, ref letter_freq) in word_freqs
                .iter()
                .zip(<[u8; ALPHABET_LENGTH]>::from(*letter_frequencies))
            {
                if word_freq > letter_freq {
                    fits = false;
                    break;
                }
            }
            if fits {
                *score += *word_score as u32;
            }
        }
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::letter::translate_word;
    use crate::utilities::test_utilities::read_random_combinations;
    use crate::utilities::{
        create_trie, create_word_vector, create_word_vector_with_scores, ALL_A_FREQUENCIES,
        ALPHABET_LENGTH,
    };
    use trie_rs::TrieBuilder;

    #[test]
    fn test_get_combination_score() {
        // build a custom trie to verify expected score on
        const WORDS: [&str; 20] = [
            "A",
            "AA",
            "AAA",
            "AAAA",
            "AAAAA",
            "AAAAAA",
            "AAAAAAA",
            "AAAAAAAA",
            "AAAAAAAAA",
            "AAAAAAAAAA",
            "AAAAAAAAAAA",
            "AAAAAAAAAAAA",
            "AAAAAAAAAAAAA",
            "AAAAAAAAAAAAAA",
            "AAAAAAAAAAAAAAA",
            "AAAAAAAAAAAAAAAA",
            "AZC",
            "AZAZ",
            "AZA",
            "AZACB",
        ];
        let mut builder: TrieBuilder<Letter> = TrieBuilder::new();
        for word in WORDS {
            builder.push(translate_word(word).unwrap());
        }
        let trie = builder.build();

        // can score all combination lengths (and don't score insufficiently long words)
        let all_a_points: u32 = POINTS.iter().map(|n| *n as u32).sum();
        let all_a_lc = LetterCombination::new(ALL_A_FREQUENCIES);
        assert_eq!(
            combination_score_all_possible_trie_paths(&trie, all_a_lc),
            all_a_points
        );

        // should score AAA, AZA, AZC (can take multiple branches from a node) but not AZAZ (exhaust letters)
        const THREES_POINTS: u32 = 3 * POINTS[3] as u32;
        const THREES_FREQS: [u8; ALPHABET_LENGTH] = [
            3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        let threes_lc = LetterCombination::new(THREES_FREQS);
        assert_eq!(
            combination_score_all_possible_trie_paths(&trie, threes_lc),
            THREES_POINTS
        );

        // should score AZA, AZC, and AZACB (use C at different positions, continue past non-scoring nodes)
        const AZ_POINTS: u32 = (2 * POINTS[3] + POINTS[5]) as u32;
        const AZ_FREQS: [u8; ALPHABET_LENGTH] = [
            2, 1, 1, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        let az_lc = LetterCombination::new(AZ_FREQS);
        assert_eq!(
            combination_score_all_possible_trie_paths(&trie, az_lc),
            AZ_POINTS
        );
    }

    #[test]
    fn test_combination_score_all_possible_words() {
        // test from first principles - measure the points that should be scored against
        // an artifical word list
        let mut expected_points = 0;
        const FREQS: [u8; ALPHABET_LENGTH] = [
            2, 1, 1, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];

        // maximal fit
        let fits_16 = FREQS;
        expected_points += POINTS[16];

        // this should fit but score no points
        let fits_1 = [
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        expected_points += POINTS[1];

        // this should fit and score some points
        let fits_3 = [
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        expected_points += POINTS[3];

        // this shouldn't fit (and should score no points)
        let no_fit = [
            3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];

        let word_list = vec![fits_16, fits_1, fits_3, no_fit];

        assert_eq!(
            u32::from(expected_points),
            combination_score_all_possible_words(&word_list, FREQS.into())
        );
    }

    #[test]
    fn test_combination_score_all_possible_words_tiled() {
        // test from first principles - measure the points that should be scored against
        // an artifical word list
        let mut expected_points = 0;
        const FREQS: [u8; ALPHABET_LENGTH] = [
            2, 1, 1, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];

        // maximal fit
        let fits_16 = FREQS;
        expected_points += POINTS[16];

        // this should fit but score no points
        let fits_1 = [
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        expected_points += POINTS[1];

        // this should fit and score some points
        let fits_3 = [
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        expected_points += POINTS[3];

        // this shouldn't fit (and should score no points)
        let no_fit = [
            3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];

        let word_list = vec![
            (fits_16, POINTS[16]),
            (fits_1, POINTS[1]),
            (fits_3, POINTS[3]),
            (no_fit, POINTS[5]),
        ];

        let combinations = [FREQS.into(); BATCH_SIZE];

        assert_eq!(
            u32::from(expected_points),
            combination_score_all_possible_words_with_scores_tiled(&word_list, &combinations)[0]
        );
    }

    #[test]
    fn test_return_same_result() {
        const WORDLIST_FILENAME: &str = "wordlist-actual.txt";
        let combinations: Vec<LetterCombination> =
            read_random_combinations("tests/test_letter_combinations");

        const COMBINATIONS_COUNT: u32 = 1_000;
        let trie = create_trie(WORDLIST_FILENAME);
        let word_vector = create_word_vector(WORDLIST_FILENAME);
        let word_vector_with_scores = create_word_vector_with_scores(WORDLIST_FILENAME);
        // check same word list length commutatively
        assert_eq!(trie.1, word_vector.1);
        assert_eq!(word_vector.1, word_vector_with_scores.1);

        let mut scores_trie = [0; COMBINATIONS_COUNT as usize];
        let mut scores_vec = [0; COMBINATIONS_COUNT as usize];
        let mut scores_vec_with_scores = [0; COMBINATIONS_COUNT as usize];
        let mut scores_vec_with_scores_tiled = [0; COMBINATIONS_COUNT as usize];

        let mut tiled_batch = [LetterCombination::new(ALL_A_FREQUENCIES); BATCH_SIZE];

        for (i, combination) in combinations.iter().enumerate() {
            scores_trie[i] = combination_score_all_possible_trie_paths(&trie.0, *combination);
            scores_vec[i] = combination_score_all_possible_words(&word_vector.0, *combination);
            scores_vec_with_scores[i] = combination_score_all_possible_words_with_scores(
                &word_vector_with_scores.0,
                *combination,
            );

            let batch_idx = i % BATCH_SIZE;
            tiled_batch[batch_idx] = combination.to_owned();

            if batch_idx == BATCH_SIZE - 1 {
                let tiled_result = combination_score_all_possible_words_with_scores_tiled(
                    &word_vector_with_scores.0,
                    tiled_batch[0..500].try_into().unwrap(),
                );
                for tiled_idx in 0..BATCH_SIZE {
                    scores_vec_with_scores_tiled[(i / BATCH_SIZE) * BATCH_SIZE + tiled_idx] =
                        tiled_result[tiled_idx];
                }
            }
        }

        // check equality commutatively
        assert_eq!(scores_trie, scores_vec);
        assert_eq!(scores_vec, scores_vec_with_scores);
        assert_eq!(scores_vec_with_scores, scores_vec_with_scores_tiled);
    }
}
