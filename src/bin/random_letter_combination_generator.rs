use rand::random_range;
use std::collections::HashSet;
use std::fs;
use wordhunt_solver::letter_combination::LetterCombination;
use wordhunt_solver::utilities::{ALPHABET_LENGTH, TILE_COUNT};

fn main() {
    /// number of combinations to generate
    const COMBINATIONS_COUNT: usize = 10_000_000;

    let mut combinations = HashSet::new();

    // generate the combinations
    while combinations.len() < COMBINATIONS_COUNT {
        combinations.insert(generate_random_combination());

        if combinations.len() % (COMBINATIONS_COUNT / 100) == 0 {
            println!(
                "generation {:.0} percent complete",
                100.0 * (combinations.len() as f64 / COMBINATIONS_COUNT as f64)
            );
        }
    }

    assert_eq!(
        COMBINATIONS_COUNT,
        combinations.len(),
        "failed to generate specified number of combinations"
    );

    // write generated combinations to disk
    let aggregated_combinations: Vec<LetterCombination> = Vec::from_iter(combinations);
    let encoded_aggregated_combinations = bincode::serialize(&aggregated_combinations).unwrap();
    fs::write(
        "random_letter_combinations",
        encoded_aggregated_combinations,
    )
    .unwrap();
}

/// generate a random letter combination
fn generate_random_combination() -> LetterCombination {
    let mut freqs = [0; ALPHABET_LENGTH];

    for _ in 0..TILE_COUNT {
        let idx = random_range(0..ALPHABET_LENGTH);
        freqs[idx] += 1;
    }

    LetterCombination::from(freqs)
}
