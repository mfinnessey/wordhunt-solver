use std::env;
use std::sync::{Arc, Mutex};

use wordhunt_solver::combination_search::combination_generator::SequentialLetterCombinationGenerator;
use wordhunt_solver::combination_search::{
    bounding_functions::combination_score_all_possible_words_with_scores, CombinationSearch,
};
use wordhunt_solver::utilities::create_word_vector_with_scores;
use wordhunt_solver::utilities::snapshot_utilities::read_next_progress_information_from_directory;

fn main() {
    let args: Vec<String> = env::args().collect();

    let progress_information = match args.len() {
	2 => {
	    None
	}
	3 => {
	    let snapshots_directory = &args[2];
	    match read_next_progress_information_from_directory(snapshots_directory) {
		Ok((snapshot_num, read_progress_information)) => {
		    println!("finished reading progress information from snapshot {} in directory {}", snapshot_num, snapshots_directory);
		    Some(read_progress_information)
		}
		Err(msg) => panic!("could not read ProgressInformation from {} due to {}", snapshots_directory, msg)
	    }
	}
	n => panic!("usage is path/to/wordlist [/path/to/snapshots/directory] but you provided {} arguments", n - 1),
    };

    // read wordlist into trie
    let word_list_filepath: &str = &args[1];
    println!("reading words from {} into word trie.", word_list_filepath);
    let (word_vector_with_scores, word_count) = create_word_vector_with_scores(word_list_filepath);
    println!(
        "finished creating word trie containing {} words.",
        word_count
    );

    let combination_terminator: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    let combination_terminator_ref = combination_terminator.clone();
    ctrlc::set_handler(move || *combination_terminator_ref.lock().unwrap() = true)
        .expect("FAILED TO SET CTRL-C HANDLER");

    let combination_evaluator = CombinationSearch::new(
        &word_vector_with_scores,
        combination_score_all_possible_words_with_scores,
        1640,
        std::thread::available_parallelism().unwrap().get(),
        combination_terminator,
    );

    const SECS_PER_HOUR: u64 = 3600;
    const SNAPSHOT_FREQUENCY: u64 = 12 * SECS_PER_HOUR;

    let combination_generator_creator = SequentialLetterCombinationGenerator::new;
    combination_evaluator.evaluate_combinations(
        SNAPSHOT_FREQUENCY,
        progress_information,
        combination_generator_creator,
    );

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
