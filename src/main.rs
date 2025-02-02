use std::fs::File;
use std::io::{self, BufRead};
use std::env;
use letter::{Letter, translate_letter, translate_word};
use trie_rs::{TrieBuilder, Trie};
use board::Board;
use combination_search::{CombinationEvaluator, get_combination_score, ALL_A_FREQUENCIES, LetterCombination};
mod board;
mod letter;
mod utilities;
mod combination_search;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
	panic!("Please enter only one argument - the relative filepath of the word list.");
    }

    // read words into trie    
    let file_path: &str = &args[1];
    println!("Reading words from {} into word trie.", file_path);
    let (trie, word_count) = create_trie(file_path);
    println!("Finished creating word trie containing {} words.", word_count);

    let all_a_combination = LetterCombination::new(ALL_A_FREQUENCIES);

    let combination_evaluator = CombinationEvaluator::new(&trie, all_a_combination, get_combination_score, 164000, 16);
    combination_evaluator.check_combinations(10);


    
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

/// create a trie from the given wordlist filepath
fn create_trie(word_list_file_path: &str) -> (Trie<Letter>, u32) {
    let mut word_count = 0;
    // read from file
    if let Ok(file) = File::open(word_list_file_path){
	let lines = io::BufReader::new(file).lines();

	// build trie of letters
	let mut builder: TrieBuilder<Letter> = TrieBuilder::new();
	for line in lines.map_while(Result::ok){
	    match translate_word(&line) {
		Ok(word) => {
		    builder.push(word);
		    word_count += 1;
		},
		Err(e) => println!("Unable to process word {} because {e:?}", line),
	    }
	}
	(builder.build(), word_count)
    }
    else {
	panic!("Could not open specified file!");
    }
}
