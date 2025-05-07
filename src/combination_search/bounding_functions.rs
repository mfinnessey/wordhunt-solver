use crate::letter::Letter;
use crate::letter_combination::LetterCombination;
use crate::utilities::{ALPHABET_LENGTH, POINTS};
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
    let mut score = 0;
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
                        score += POINTS[new_search.prefix_len()];

                        let mut new_counts = remaining_counts;
                        new_counts[i] -= 1;
                        queue.push_back((new_search, new_counts));
                    }
                    // score only
                    Some(Answer::Match) => {
                        score += POINTS[new_search.prefix_len()];
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
            score += &POINTS[word_freqs.iter().sum::<u8>() as usize];
        }
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::letter::translate_word;
    use crate::utilities::ALL_A_FREQUENCIES;
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
        let all_a_points: u32 = POINTS.iter().sum();
        let all_a_lc = LetterCombination::new(ALL_A_FREQUENCIES);
        assert_eq!(
            combination_score_all_possible_trie_paths(&trie, all_a_lc),
            all_a_points
        );

        // should score AAA, AZA, AZC (can take multiple branches from a node) but not AZAZ (exhaust letters)
        const THREES_POINTS: u32 = 3 * POINTS[3];
        const THREES_FREQS: [u8; ALPHABET_LENGTH] = [
            3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        let threes_lc = LetterCombination::new(THREES_FREQS);
        assert_eq!(
            combination_score_all_possible_trie_paths(&trie, threes_lc),
            THREES_POINTS
        );

        // should score AZA, AZC, and AZACB (use C at different positions, continue past non-scoring nodes)
        const AZ_POINTS: u32 = 2 * POINTS[3] + POINTS[5];
        const AZ_FREQS: [u8; ALPHABET_LENGTH] = [
            2, 1, 1, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        let az_lc = LetterCombination::new(AZ_FREQS);
        assert_eq!(
            combination_score_all_possible_trie_paths(&trie, az_lc),
            AZ_POINTS
        );
    }
}
