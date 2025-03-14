use std::env;

use wordhunt_solver::board::Board;
use wordhunt_solver::combination_search::{
    bounding_functions::combination_score_all_possible_trie_paths, CombinationSearch,
};
use wordhunt_solver::letter::Letter;
use wordhunt_solver::letter_combination::LetterCombination;
use wordhunt_solver::utilities::{create_trie, ALL_A_FREQUENCIES};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        panic!("Please enter only one argument - the relative filepath of the word list.");
    }

    // read words into trie
    let file_path: &str = &args[1];
    println!("Reading words from {} into word trie.", file_path);
    let (trie, word_count) = create_trie(file_path);
    println!(
        "Finished creating word trie containing {} words.",
        word_count
    );

    let all_a_combination = LetterCombination::new(ALL_A_FREQUENCIES);

    let combination_evaluator = CombinationSearch::new(
        &trie,
        all_a_combination,
        combination_score_all_possible_trie_paths,
        1640,
        std::thread::available_parallelism().unwrap().get(),
    );
    combination_evaluator.evaluate_combinations(120);

    // temporary test grid from https://www.youtube.com/watch?v=3cgr_GgA5ns
    /*    let test_board_chars = ['M', 'H', 'O', 'N', 'I', 'T', 'E', 'R', 'L', 'A', 'S', 'N', 'S', 'E', 'R', 'U'];
    let test_board_letters: [Letter; 16] = test_board_chars.into_iter().filter_map(|c| translate_letter(&c)).collect::<Vec<Letter>>().try_into().unwrap();
    let mut letter_counts: [u8; 26] = [0; 26];
    for letter in test_board_letters.iter() {
    letter_counts[letter.clone() as usize] += 1;
    }
    println!("Maximum possible score for test board letters is {}.", get_combination_score(&trie, letter_counts));

    let mut test_board = Board::new(test_board_letters, &trie);

    println!("Score for test board is {}", test_board.maximum_score() * 100);
    let mut words = test_board.get_words();
    words.sort_by_key(|a| a.len());
    println!("Words in board are {:?}", words); */
}
