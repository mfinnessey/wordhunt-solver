use std::collections::{BinaryHeap, VecDeque};
use std::ops::{Index, IndexMut};
use std::cmp::Ordering;
use std::{iter, thread, time};
use trie_rs::Trie;
use trie_rs::inc_search::Answer;
use crate::utilities::POINTS;
use crate::letter::Letter;

use std::sync::mpsc::{Sender, Receiver};
use std::sync::{Arc, mpsc};
use crossbeam_deque::{Injector, Stealer, Worker};
use crossbeam_utils::thread as crossbeam_thread;
use crossbeam::thread::ScopedJoinHandle;

// TODO benchmark this to see what makes sense
const PULL_LIMIT: usize = 5000;

const ALPHABET_LENGTH: usize = 26;
const TILE_COUNT: u8 = 16;

/// the starting value of the iterator over all letter combinations
const ALL_A_FREQUENCIES: [u8; ALPHABET_LENGTH] = [TILE_COUNT, 0, 0, 0, 0,
						  0, 0, 0, 0, 0,
						  0, 0, 0, 0, 0,
						  0, 0, 0, 0, 0,
						  0, 0, 0, 0, 0,
						  0,];
// TODO standardize on combination / frequencies terminology
/// the ending value of the iterator over all letter combinations
const ALL_Z_FREQUENCIES: [u8; ALPHABET_LENGTH] = [0, 0, 0, 0, 0,
						  0, 0, 0, 0, 0,
						  0, 0, 0, 0, 0,
						  0, 0, 0, 0, 0,
						  0, 0, 0, 0, 0,
						  TILE_COUNT,];
/// the size of individual checkpints to write
const CHECKPOINT_SIZE: usize = 1000000;
    
#[derive(Copy, Clone)]
struct LetterCombination {
    frequencies: [u8; ALPHABET_LENGTH],
}

impl LetterCombination {
    fn new(frequencies: [u8; ALPHABET_LENGTH]) -> Self {
	Self {frequencies}
    }
}

/// Ord implementation in REVERSE ORDER of generation to make BinaryHeap a min-heap
impl Ord for LetterCombination {
    fn cmp(&self, other: &Self) -> Ordering {
	for (freq1, freq2) in self.frequencies.iter().zip(other.frequencies.iter()) {
	    if freq1 > freq2 {
		return Ordering::Greater;
	    }
	    if freq2 > freq1 {
		return Ordering::Less;
	    }
	}
	Ordering::Equal
    }
}

impl PartialOrd for LetterCombination {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
	Some(self.cmp(other))
    }
}

impl PartialEq for LetterCombination {
    fn eq(&self, other: &Self) -> bool {
	self.frequencies == other.frequencies
    }
}

impl Index<usize> for LetterCombination {
    type Output = u8;
    fn index(&self, idx: usize) -> &Self::Output {
	if idx > ALPHABET_LENGTH - 1 {
	    panic!("Index {} for LetterFrequencies is outside the [0, 25]!", idx)
	}
	&self.frequencies[idx]
    }
}

impl IndexMut<usize> for LetterCombination {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
	if idx > ALPHABET_LENGTH - 1 {
	    panic!("Index {} for LetterFrequencies is outside the [0, 25]!", idx)
	}
	&mut self.frequencies[idx]
    }
}

impl Eq for LetterCombination {}

impl From<LetterCombination> for [u8; 26] {
    fn from(frequencies: LetterCombination) -> Self {
	frequencies.frequencies
    }
}

// TODO make this more generic in terms of length and count and then write
// some tests for this!
/// generate the combinations of letters that will be evaluated.
struct LetterCombinationGenerator {
    /// we represent each combination of letters as a set of frequency counts.
    /// (think pushing a histogram of tiles around)
    frequencies: LetterCombination,
    /// the leftmost index that has a non-zero count.
    left: usize,
    /// the right index at which we are moving the current frequency around
    right: usize,
}

impl LetterCombinationGenerator {
    pub fn new(initial_frequencies: LetterCombination) -> Self {
	Self {
	    frequencies: initial_frequencies,
	    left: 0,
	    right: 0,
	}
    }
}

impl Iterator for LetterCombinationGenerator {
    type Item = LetterCombination;
    
    fn next(&mut self) -> Option<Self::Item> {
	let cur = Some(self.frequencies);
	
	// end state where all tiles are as far to the right as possible
        if self.frequencies[ALPHABET_LENGTH - 1] == TILE_COUNT {
	    return None;
	} 

	// advance the iterator
	// we can uniquely generate all combinations of letters by "pushing"
	// the frequencies from 'A' to 'Z' one by one.
	// e.g. roughly
	// 16 0 ... 0
	// 8 7 ... 1
	// 0 0 ... 16
	// A B ... Z
	if self.right == ALPHABET_LENGTH - 1 {
	    // put the tile one to the right of left
	    self.frequencies[ALPHABET_LENGTH - 1] -= 1;
	    self.frequencies[self.left + 1] += 1;

	    // advance the left barrier if necessary
	    if self.frequencies[self.left] == 0 {
		self.left += 1;
	    }

	    // start moving from the left barrier again
	    self.right = self.left;
	}
	// push the current tile to the right
	else {
	    self.frequencies[self.right] -= 1;
	    self.right += 1;
	    self.frequencies[self.right] += 1;
	}
	
	cur
    }
}

type PassMsg = (LetterCombination, u32);

/// write persistent progress checkpoints to disk.
/// writes 1) tuples of (successful combination, score) and 2) the maximal combination
/// of letters that we have contiguously processed all values from the beginning through
struct CheckpointWriter {
    pass_rx: Receiver<PassMsg>,    
    fail_rx: Receiver<LetterCombination>,
    /// elements read from pass_rx that are out of order with respect to the contiguous maximum
    pending_pass: BinaryHeap<PassMsg>,
    /// elements read from fail_rx that are out of order with respect to the contiguous maximum
    pending_fail: BinaryHeap<LetterCombination>,
    /// creates the next element after the range of contiguous combinations that have been processed
    next_combination_generator: LetterCombinationGenerator,
    /// passing elements that we have not yet written in their individual checkpoint files
    pass_buffer: Vec<(LetterCombination, u32)>,
}

impl CheckpointWriter {
    fn new(pass_rx: Receiver<PassMsg>,
	   fail_rx: Receiver<LetterCombination>) -> Self {
	Self {
	    pass_rx,
	    fail_rx,
	    pending_pass: BinaryHeap::new(),
	    pending_fail: BinaryHeap::new(),
	    next_combination_generator: LetterCombinationGenerator::new(LetterCombination::new(ALL_A_FREQUENCIES)),
	    pass_buffer: Vec::new(),
	}
    }

    /// write progress checkpoint to disk
    // TODO test this - just throw random combinations in with the reduced combination space??
    // we love generics for testing!
    fn write_checkpoints(&mut self) {
	let next_combination_opt = self.next_combination_generator.next();
	loop {
	    // TODO sleep every so often

	    let mut from_pending = false;
	    // TODO don't unwrap this!!! will panic at end
	    let next_combination = next_combination_opt.unwrap();
	    
	    // try to get the value from the failing heap first (I expect most combination to fail)
	    if let Some(fail_heap_top) = self.pending_fail.peek() {
		if *fail_heap_top == next_combination {
		    // process contiguous value			    
		    self.pending_fail.pop();

		    // mark that we successfully got the combination from a pending heap
		    from_pending = true;
		}
	    }
	    // try the passing heap next
	    if !from_pending && self.pending_pass.peek().is_some() {
		// unwrap is safe by construction due to check above
		let pass_heap_top = self.pending_pass.peek().unwrap();
		
		if pass_heap_top.0 == next_combination {
		    // process contiguous value			    
		    self.pass_buffer.push(*pass_heap_top);
		    self.pending_pass.pop();

		    // mark that we successfully got the combination from a pending heap
		    from_pending = true;
		}
		if !from_pending {
		 // TODO read from the channels!   
		}
		
		// TODO make these try_recv to avoid blocking if we get in unbalanced
		// portions
		if let Ok(read) = self.pass_rx.recv() {
		    let combination = read.0;
		    // we got the next value we want - so we can directly
		    // buffer it for writing to disk
		    if combination == next_combination {
			self.pass_buffer.push(read);
		    }
		    // we got something out of order - we need to push
		    // it into the pending heap
		    else {
			self.pending_pass.push(read);
		    }
		}
		else {
		    // all passing combinations have been received
		}
	    }
	    else {
		let read = self.fail_rx.recv();
	    }
	}
    }
}

/// evaluates all combinations of letters using a given wordlist using a given metric
struct CombinationEvaluator<'a> {
    word_list: &'a Trie<Letter>,
    combinations: LetterCombinationGenerator,
    /// retain the score instead of simply a pass/fail bool so that we can eliminate combinations
    /// based off of lower bounds from stage 2
    metric: fn(&Trie<Letter>, LetterCombination) -> u32,
    target: u32,
    num_worker_threads: usize,
}

/// the information that a worker thread is provided with to process combinations
/// in conjunction with the overall program.
struct WorkerInformation<'a> {
    word_list: &'a Trie<Letter>,
    metric: fn(&Trie<Letter>, LetterCombination) -> u32,
    target: u32,
    local: Worker<LetterCombination>,
    global: Arc<Injector<LetterCombination>>,
    stealers: Arc<Vec<Stealer<LetterCombination>>>,
    pass_tx: Sender<PassMsg>,
    fail_tx: Sender<LetterCombination>,
    all_combinations_generated: Arc<bool>,
}

impl<'a> WorkerInformation<'a> {
    // this object's constructor is designed to simplify the signature of evaluate_combinations
    #[allow(clippy::too_many_arguments)]
    fn new(word_list: &'a Trie<Letter>, metric: fn(&Trie<Letter>, LetterCombination) -> u32,
	   target: u32, local: Worker<LetterCombination>, global: Arc<Injector<LetterCombination>>,
	   stealers: Arc<Vec<Stealer<LetterCombination>>>, pass_tx: Sender<PassMsg>, fail_tx: Sender<LetterCombination>,
	   all_combinations_generated: Arc<bool> ) -> Self {
	Self {
	    word_list,
	    metric,
	    target,
	    local,
	    global,
	    stealers,
	    pass_tx,
	    fail_tx,
	    all_combinations_generated
	}
    }
}


impl<'a> CombinationEvaluator<'a> {
    pub fn new(word_list: &'a Trie<Letter>, starting_frequencies: LetterCombination, metric: fn(&Trie<Letter>, LetterCombination) -> u32,
	       target: u32, num_worker_threads: usize) -> Self {
	Self {
	    word_list,
	    combinations: LetterCombinationGenerator::new(starting_frequencies),
	    metric,
	    target,
	    num_worker_threads,
	}
    }

    /// spawn threads to create combination evaluation architecture
    pub fn check_combinations(self) {
	let global_queue = Injector::new();

	let mut workers: Vec<Worker<LetterCombination>> = Vec::new();
	let mut stealers: Vec<Stealer<LetterCombination>> = Vec::new();

	for _ in 0..self.num_worker_threads {
	    let w = Worker::new_fifo();
	    let stealer = w.stealer();
	    workers.push(w);
	    stealers.push(stealer);
	}

	let all_combinations_generated = false;	
	let (pass_tx, pass_rx) = mpsc::channel();
	let (fail_tx, pass_rx) = mpsc::channel();

	let stealers_vec_ref = Arc::new(stealers);
	let all_combinations_generated_ref = Arc::new(all_combinations_generated);
	let global_queue_ref = Arc::new(global_queue);

	crossbeam_thread::scope(|s| {
	    // spawn the generator thread
	    let global_queue_ref_for_generator = global_queue_ref.clone();
	    let generator_handle = s.spawn(move |_| push_combinations(self.combinations, global_queue_ref_for_generator));

	    // TODO add checkpoint writer in its own thread	    
	    
	    let mut worker_threads: Vec<ScopedJoinHandle<'_, ()>> = Vec::new();	    
	    for _ in 0..self.num_worker_threads {
		// using unwrap as is safe by construction - we have pushed num_worker_threads elements
		let thread_worker = workers.pop().unwrap();
		let stealers_ref = stealers_vec_ref.clone();
		let global_ref = global_queue_ref.clone();

		let worker_information = WorkerInformation::new(self.word_list, self.metric, self.target, thread_worker,
								global_ref, stealers_ref, pass_tx.clone(), fail_tx.clone(),
								all_combinations_generated_ref.clone());


		let handle = s.spawn(move |_| evaluate_combinations(worker_information));
		worker_threads.push(handle);
	    }

	    if generator_handle.join().is_err(){
		panic!("Generator thread paniced!")
	    }

	    for worker_thread_handle in worker_threads {
		if worker_thread_handle.join().is_err() {
		    panic!("Worker thread paniced!")
		}
	    }

	    // TODO join checkpoint writer thread
	}).unwrap(); // unwrap as we would just panic anyways
    }
}

fn push_combinations(combinations: LetterCombinationGenerator, queue: Arc<Injector<LetterCombination>>){
    for combination in combinations {
	queue.push(combination);
    }
}

fn evaluate_combinations(worker_information: WorkerInformation) {
    // unpack convenience struct
    let word_list = worker_information.word_list;
    let metric = worker_information.metric;
    let target = worker_information.target;
    let local = worker_information.local;
    let global = worker_information.global;
    let stealers = worker_information.stealers;
    let pass_tx = worker_information.pass_tx;
    let fail_tx = worker_information.fail_tx;
    let all_combinations_generated = worker_information.all_combinations_generated;
    
    loop {
	// modified from crossbeam::deque docs
	// pop a task from the local queue, if not empty.
	let combination = local.pop().or_else(|| {
            // otherwise, we need to look for a task elsewhere.
            iter::repeat_with(|| {
		// try stealing a batch of tasks from the global queue.
		global.steal_batch_with_limit_and_pop(&local, PULL_LIMIT)
                // or try stealing a task from one of the other threads.
                    .or_else(|| stealers.iter().map(|s| s.steal()).collect())
            })
            // loop while no task was stolen and any steal operation needs to be retried.
		.find(|s| !s.is_retry())
            // extract the stolen task, if there is one.
		.and_then(|s| s.success())
	});
	

	match combination {
	    // process the combination and write the result to the appropriate channel
	    Some(letters) => {
		let score = (metric)(word_list, letters);
		if score > target {
		    if pass_tx.send((letters, score)).is_err() {
			panic!("Pass rx dropped before completion of processing.")
		    }
		}
		else if fail_tx.send(letters).is_err() {
		    panic!("Fail rx dropped before completion of processing.")
		} 
	    },
	    None => {
		if *all_combinations_generated {
		    break;
		}
		else {
		    println!("All queues are dry but not all letter combinations are exhausted. Sleeping.");
		    thread::sleep(time::Duration::from_secs(1));
		}
	    }
	}
    }
}

/// get the summed scores of all words of length <= 16
/// that can be built from a given combination of 16 letters represented
/// as frequency counts
// TODO test this - correctness issue. idea: just used a custom-made reduced dimensionality trie with
// interesting test cases
// TODO consider making non pub
pub fn get_combination_score(dictionary: &Trie<Letter>, letter_frequencies: LetterCombination) -> u32 {
    // compute score by bfs through trie
    let mut score = 0;
    let mut queue = VecDeque::new();
    // convert into slice for iteration
    let letter_counts: [u8; ALPHABET_LENGTH] = letter_frequencies.into();
    
    // build inc search starting from each available letter
    for (i, count) in letter_counts.iter().enumerate() {
	if *count > 0 {
	    let mut inc_search = dictionary.inc_search();
	    if inc_search.query(&Letter::from(i)).is_some(){
		let mut new_counts = letter_counts;
		new_counts[i] -= 1;
		queue.push_back((inc_search, new_counts));
	    }
	}
    }

    while let Some(cur) = queue.pop_front(){
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
		    },
		    // score and continuation
		    Some(Answer::PrefixAndMatch) | Some(Answer::Match) => {
			if new_search.prefix_len() >= 3 {
			    score += POINTS[new_search.prefix_len() - 3];
			}
			
			let mut new_counts = remaining_counts;
			new_counts[i] -= 1;
			queue.push_back((new_search, new_counts));
		    },
		    // non-viable continuation
		    None => (),
		}
	    }
	}
    }

    score
}
