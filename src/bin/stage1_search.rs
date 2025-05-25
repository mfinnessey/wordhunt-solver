use std::env;
use std::sync::{Arc, Mutex};

use wordhunt_solver::board::Board;
use wordhunt_solver::combination_search::combination_generator::SequentialLetterCombinationGenerator;
use wordhunt_solver::combination_search::{
    bounding_functions::combination_score_all_possible_words_with_scores_tiled, CombinationSearch,
};
use wordhunt_solver::letter::{translate_letter, Letter};
use wordhunt_solver::utilities::snapshot_utilities::read_next_progress_information_from_directory;
use wordhunt_solver::utilities::{create_trie, create_word_vector_with_scores, TILE_COUNT};

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

    // read wordlist into vector
    let word_list_filepath: &str = &args[1];
    println!("reading words from {} into word trie.", word_list_filepath);
    let (word_vector_with_scores, word_vector_word_count) =
        create_word_vector_with_scores(word_list_filepath);
    println!(
        "finished creating word trie containing {} words.",
        word_vector_word_count
    );

    // verify target score
    const TARGET_SCORE: u32 = 5354;

    // grid from https://www.youtube.com/watch?v=3cgr_GgA5ns
    let (trie, trie_word_count) = create_trie(word_list_filepath);
    assert_eq!(trie_word_count, word_vector_word_count);
    let test_board_chars = [
        'M', 'H', 'O', 'N', 'I', 'T', 'E', 'R', 'L', 'A', 'S', 'N', 'S', 'E', 'R', 'U',
    ];
    let test_board_letters: [Letter; TILE_COUNT] = test_board_chars
        .into_iter()
        .filter_map(|c| translate_letter(&c))
        .collect::<Vec<Letter>>()
        .try_into()
        .unwrap();
    let mut test_board = Board::new(test_board_letters, &trie);
    assert_eq!(test_board.maximum_score(), &TARGET_SCORE);

    // set up early termination infrastructure
    let combination_terminator: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
    let combination_terminator_ref = combination_terminator.clone();
    ctrlc::set_handler(move || *combination_terminator_ref.lock().expect("another thread erroneously accessed and paniced while holding the combination terminator") = true)
        .expect("FAILED TO SET CTRL-C HANDLER");

    // launch combination evaluation
    let num_threads = match std::thread::available_parallelism() {
        Ok(n) => n.get(),
        Err(e) => panic!("failed to get available parllelism due to error: {}", e),
    };
    let combination_evaluator = CombinationSearch::new(
        &word_vector_with_scores,
        combination_score_all_possible_words_with_scores_tiled,
        1640,
        num_threads,
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
}
