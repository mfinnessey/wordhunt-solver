use std::collections::VecDeque;
use std::ops::{Index, IndexMut};
use std::cmp::Ordering;
use std::{fs, iter, thread, time};
use std::time::SystemTime;
use trie_rs::Trie;
use trie_rs::inc_search::Answer;
use crate::utilities::POINTS;
use crate::letter::Letter;
use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};

use std::sync::{Arc, Mutex, Condvar, RwLock, mpsc::channel};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::atomic::{AtomicBool, Ordering as MemoryOrdering};
use crossbeam_deque::{Injector, Stealer, Worker};
use crossbeam_utils::thread as crossbeam_thread;
use crossbeam::thread::ScopedJoinHandle;

// TODO benchmark this if need be
/// the maximum batch size that a worker thread will pull from the global queue at once
const PULL_LIMIT: usize = 10_000;

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

/// the amount of entries that must be in the global queue before the worker threads
/// are released
const STARTUP_PUSH_COUNT: u32 = 200_000;

/// the number of worker threads to run
const WORKER_THREAD_COUNT: u8 = 16;

#[derive(Copy, Clone, Serialize, Deserialize)]
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
    /// inform the generator thread of pass counts for statistics output
    pass_count_tx: Sender<u64>,
    /// synchronization stuff
    /// all combinations have been generated - the condition that will ultimately terminate each thread
    all_combinations_generated: Arc<RwLock<bool>>,
    /// the thread should stop for a snapshot
    stop_for_snapshot: &'a AtomicBool,
    /// the snapshot number of the current snapshot
    snapshot_number: Arc<RwLock<u64>>,
    /// the generator thread has (temporarily) stopped generating new combinations (for the purposes of a snapshot)
    generator_thread_stopped: Arc<RwLock<bool>>,
    /// set by the last worker thread to stop for a snapshot to signal the generator thread that the queues are empty
    queues_empty: Arc<(Mutex<bool>, Condvar)>,
    /// the number of worker threads that have stopped for a snapshot
    workers_stopped_for_snapshot: Arc<Mutex<u8>>,
    /// set by the generator thread to notify the worker threads that the snapshot has been completed
    snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
    /// where to write snapshots
    snapshot_directory: Arc<&'a Path>,
}

impl<'a> WorkerInformation<'a> {
    // this object's constructor is designed to simplify the signature of evaluate_combinations
    #[allow(clippy::too_many_arguments)]
    fn new(word_list: &'a Trie<Letter>, metric: fn(&Trie<Letter>, LetterCombination) -> u32,
	   target: u32, local: Worker<LetterCombination>, global: Arc<Injector<LetterCombination>>,
	   stealers: Arc<Vec<Stealer<LetterCombination>>>, pass_count_tx: Sender<u64>,
	   all_combinations_generated: Arc<RwLock<bool>>, stop_for_snapshot: &'a AtomicBool,
	   snapshot_number: Arc<RwLock<u64>>, generator_thread_stopped: Arc<RwLock<bool>>,
	   queues_empty: Arc<(Mutex<bool>, Condvar)>, workers_stopped_for_snapshot: Arc<Mutex<u8>>,
	   snapshot_complete: Arc<(Mutex<bool>, Condvar)>, snapshot_directory: Arc<&'a Path>) -> Self {
	Self {
	    word_list,
	    metric,
	    target,
	    local,
	    global,
	    stealers,
	    pass_count_tx,
	    all_combinations_generated,	    
	    stop_for_snapshot,
	    snapshot_number,
	    generator_thread_stopped,
	    queues_empty,
	    workers_stopped_for_snapshot,
	    snapshot_complete,
	    snapshot_directory,
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
	// create snapshot directory with time-based unique identifier
	let cur_time = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs();
	let path_string = "snapshots-".to_string() + &cur_time.to_string();
	let snapshot_directory = Path::new(&path_string);
	match fs::create_dir(snapshot_directory) {
	    Ok(_) => (),
	    Err(_) => panic!("Could not create checkpoints directory"),
	}
	
	let global_queue = Injector::new();

	let mut workers: Vec<Worker<LetterCombination>> = Vec::new();
	let mut stealers: Vec<Stealer<LetterCombination>> = Vec::new();
	let mut stopped_mutexes: Vec<Mutex<()>> = Vec::new();

	for _ in 0..self.num_worker_threads {
	    let w = Worker::new_fifo();
	    let stealer = w.stealer();
	    workers.push(w);
	    stealers.push(stealer);
	    stopped_mutexes.push(Mutex::new(()));
	}

	let (pass_count_tx, pass_count_rx) = channel();
	
	let all_combinations_generated = Arc::new(RwLock::new(false));
	let stop_for_snapshot = AtomicBool::new(false);
	let snapshot_number: Arc<RwLock<u64>> = Arc::new(RwLock::new(0));
	let generator_thread_stopped = Arc::new(RwLock::new(false));

	let queues_empty_mutex = Mutex::new(false);
	let queues_empty_condvar = Condvar::new();
	let queues_empty = Arc::new((queues_empty_mutex, queues_empty_condvar));
	let workers_stopped_for_snapshot: Arc<Mutex<u8>> = Arc::new(Mutex::new(0));
	let snapshot_complete_mutex = Mutex::new(false);
	let snapshot_complete_condvar = Condvar::new();
	let snapshot_complete = Arc::new((snapshot_complete_mutex, snapshot_complete_condvar));
	
	let stealers_vec_ref = Arc::new(stealers);
	let global_queue_ref = Arc::new(global_queue);
	let snapshot_directory_ref = Arc::new(snapshot_directory);

	crossbeam_thread::scope(|s| {
	    // spawn the generator thread
	    let global_queue_ref_for_generator = global_queue_ref.clone();
	    let all_combinations_generated = &all_combinations_generated;
	    let stop_for_snapshot = &stop_for_snapshot;
	    let snapshot_number = &snapshot_number;
	    let generator_thread_stopped = &generator_thread_stopped;
	    let queues_empty = &queues_empty;
	    let snapshot_complete = &snapshot_complete;
	    let generator_handle = s.spawn(move |_| generate_combinations(self.combinations, global_queue_ref_for_generator,
									  Arc::clone(&all_combinations_generated), stop_for_snapshot,
									  Arc::clone(&snapshot_number), Arc::clone(&generator_thread_stopped),
									  Arc::clone(&queues_empty), Arc::clone(&snapshot_complete), pass_count_rx));
	    let mut worker_threads: Vec<ScopedJoinHandle<'_, ()>> = Vec::new();
	    for thread_num in 0..self.num_worker_threads {
		// using unwrap as is safe by construction - we have pushed num_worker_threads elements
		let thread_worker = workers.pop().unwrap();
		let stealers_ref = stealers_vec_ref.clone();
		let global_ref = global_queue_ref.clone();

		let worker_information = WorkerInformation::new(self.word_list, self.metric, self.target, thread_worker,
								global_ref, stealers_ref, pass_count_tx.clone(),
								Arc::clone(&all_combinations_generated),
								&stop_for_snapshot, Arc::clone(&snapshot_number), Arc::clone(&generator_thread_stopped),
								Arc::clone(&queues_empty), Arc::clone(&workers_stopped_for_snapshot), Arc::clone(&snapshot_complete),
								Arc::clone(&snapshot_directory_ref));
		let handle = s.builder().name(thread_num.to_string() + "thread").spawn(
		    move |_| evaluate_combinations(worker_information)).unwrap();
		worker_threads.push(handle);
	    }

	    // TODO spawn snapshot thread

	    if generator_handle.join().is_err(){
		panic!("Generator thread paniced!")
	    }

	    for worker_thread_handle in worker_threads {
		if worker_thread_handle.join().is_err() {
		    panic!("Worker thread paniced!")
		}
	    }

	    // TODO join snapshot thread
	}).unwrap(); // unwrap as we would just panic anyways
    }
}

// TODO comment out the design
// TODO SPAWN THIS IN ITS OWN THREAD
fn trigger_snapshots(stop_for_snapshot: &AtomicBool, snapshot_number: RwLock<u32>, snapshot_complete: Mutex<bool>,
		     execution_completed: Mutex<bool>){
    loop {	
	// mark for the other threads to begin taking a snapshot
	stop_for_snapshot.store(true, MemoryOrdering::SeqCst);

	// take checkpoints every 15 minutes, waking every 10 seconds to check if execution has
	// completed
	// TODO write this more cleanly
	for _ in 0..90 {
	    if *execution_completed.lock().unwrap() {
		return;
	    }
	    
	    thread::sleep(time::Duration::from_secs(10));	    
	}


	// confirm that the global thread has completed the previous snapshot
	while !*snapshot_complete.lock().unwrap() {
	    thread::sleep(time::Duration::from_millis(200));
	}
	
	*snapshot_number.write().unwrap() += 1;		
    }
}

fn generate_combinations(combinations: LetterCombinationGenerator, queue: Arc<Injector<LetterCombination>>,
			 all_combinations_generated: Arc<RwLock<bool>>, stop_for_snapshot: &AtomicBool,
			 snapshot_number: Arc<RwLock<u64>>, generator_thread_stopped: Arc<RwLock<bool>>,
			 queues_empty: Arc<(Mutex<bool>, Condvar)>, snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
			 pass_count_rx: Receiver<u64>) {

    const TOTAL_COMBINATIONS_COUNT: u64 = 103_077_446_706;

    let start_time = SystemTime::now();

    let mut overall_combinations_generated_count: u64 = 0;
    let mut last_snapshot_combination_count: u64 = 0;
    let mut overall_pass_count: u64 = 0;
    
    // begin by spinning up
    let mut spin_up_pushes_remaining = STARTUP_PUSH_COUNT;
    let mut spinning_up = true;
    
    for combination in combinations {
	overall_combinations_generated_count += 1;
	
	// can consider checking every x iteratons only if the atomic cas
	// winds up being too expensive (guessing that it can be parallelized / branch
	// predicted efficiently enough)
	if stop_for_snapshot.load(MemoryOrdering::SeqCst) {
	    // notify worker threads that we have stopped generating new combinations
	    {
		let mut generation_stopped = generator_thread_stopped.write().unwrap();
		*generation_stopped= true;
	    }

	    // write the next combination to disk (the snapshot consists of the results up to this
	    // point and the next thing to be considered)
	    let encoded_next = bincode::serialize(&combination).unwrap();
	    let generator_snapshot_path = "checkpoints/gen".to_string() + &(*snapshot_number.read().unwrap()).to_string();
	    // intentionally panic if the filesystem operation fails	    
	    fs::write(generator_snapshot_path, encoded_next).unwrap();
	    
	    // stop until we are notified that the queues have been emptied
	    // n.b. if the generator thread were to be the last thread to stop,
	    // (very unlikely given the amount of work that the worker threads have to do)
	    // then the above flag would prevent the worker thread from stopping until
	    // this thread has stopped.
	    // we are thus practically guaranteed to be signaled.
	    let (ref lock, ref condvar) = *queues_empty;
	    let mut queues_emptied = lock.lock().unwrap();
	    while !*queues_emptied {
		queues_emptied = condvar.wait(queues_emptied).unwrap();
	    }
	    
	    // check that a snapshot invariant (empty queues at completion) has been satisfied
	    if queue.len() != 0 {
		panic!("Global queue was not empty at end of snapshot");
	    }

	    // defer on marking snapshot as complete to prevent another snapshot from being
	    // triggered before we've spun back up (unlikely but an annoyingly possible case)

	    // compute and output progress statistics
	    let batch_count = overall_combinations_generated_count - last_snapshot_combination_count;
	    let mut batch_pass_count: u64 = 0;
	    loop {
		let pass_count_msg = pass_count_rx.try_recv();
		match pass_count_msg {
		    Ok(count) => batch_pass_count += count,
		    // these statistics are informational only - ignore errors
		    _ => break,
		}
	    }
	    let batch_fail_count = batch_count - batch_pass_count;

	    let batch_pass_rate: f64;
	    // prevent divide by zero	    
	    if batch_count != 0 {
		batch_pass_rate = (batch_pass_count as f64) / (batch_count as f64);
	    }
	    else {
		batch_pass_rate = 0.0;
	    }

	    overall_pass_count += batch_pass_count;
	    let overall_fail_count: u64 = overall_combinations_generated_count - overall_pass_count;
	    let overall_pass_rate: f64;
	    // prevent divide by zero	    
	    if overall_combinations_generated_count != 0 {
		overall_pass_rate = (overall_pass_count as f64) / (overall_combinations_generated_count as f64);
	    }
	    else {
		overall_pass_rate = 0.0;
	    }

	    let remaining_combinations_count = TOTAL_COMBINATIONS_COUNT - overall_combinations_generated_count;
	    let remaining_combinations_percentage: f64 = (remaining_combinations_count as f64) / (TOTAL_COMBINATIONS_COUNT as f64);

	    let elapsed_time = SystemTime::now().duration_since(start_time);
	    let elapsed_time_mins;
	    match elapsed_time {
		Ok(time) => elapsed_time_mins = (time.as_secs() as f64) / 60.0,
		Err(_) => elapsed_time_mins = 0.0,
	    }

	    // ratio of remaining work to completed work applied to elapsed time
	    let estimated_time_remaining_mins = (remaining_combinations_percentage / (1.0 - remaining_combinations_percentage))
		* elapsed_time_mins;

	    println!("*****");	    
	    println!("Wrote checkpoint to disk.");
	    println!("Batch: Total {} | Pass {} | Fail {} | Pass Rate {:.2}",
		     batch_count, batch_pass_count, batch_fail_count, batch_pass_rate);
	    println!("Overall: Total {} | Pass {} | Fail {} | Pass Rate {:.2}",
		     overall_combinations_generated_count, overall_pass_count, overall_fail_count, overall_pass_rate);
	    println!("Remaining: {} ({:.2}%) of Total {})",
		     remaining_combinations_count, remaining_combinations_percentage, TOTAL_COMBINATIONS_COUNT);
	    println!("Ran for {:.1} minutes. Estimate {:.1} minutes remaining",
		     elapsed_time_mins, estimated_time_remaining_mins);
	    println!("*****");

	    last_snapshot_combination_count = overall_combinations_generated_count;

	    // spin back up before releasing the hounds (limit context switches)
	    spinning_up = true;
	    spin_up_pushes_remaining = STARTUP_PUSH_COUNT;
	}

	if spinning_up {
	    spin_up_pushes_remaining -= 1;
	    if spin_up_pushes_remaining == 0 {
		spinning_up = false;
		// mark that the snapshot has been completed
		stop_for_snapshot.store(false, MemoryOrdering::SeqCst);
		// AND RELEASE THE HOUNDS
		*snapshot_complete.0.lock().unwrap() = true;
		snapshot_complete.1.notify_all();
	    }
	}

	queue.push(combination);
    }

    *all_combinations_generated.write().unwrap() = true;
}

fn evaluate_combinations(worker_information: WorkerInformation) {
    // unpack convenience struct
    let word_list = worker_information.word_list;
    let metric = worker_information.metric;
    let target = worker_information.target;
    let local = worker_information.local;
    let global = worker_information.global;
    let stealers = worker_information.stealers;
    let pass_count_tx = worker_information.pass_count_tx;
    let all_combinations_generated = worker_information.all_combinations_generated;    
    let stop_for_snapshot = worker_information.stop_for_snapshot;
    let snapshot_number = worker_information.snapshot_number;
    let generator_thread_stopped = worker_information.generator_thread_stopped;
    let queues_empty = worker_information.queues_empty;
    let workers_stopped_for_snapshot = worker_information.workers_stopped_for_snapshot;
    let snapshot_complete = worker_information.snapshot_complete;
    let snapshot_directory = worker_information.snapshot_directory;

    let mut passed = Vec::new();
    let mut batch_pass_count: u64 = 0;
    
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
	    // process the combination
	    Some(letters) => {
		let score = (metric)(word_list, letters);
		if score > target {
		    passed.push((letters, score));
		    batch_pass_count += 1;
		}
	    },
	    None => {
		// write the final snapshot (final results) to disk
		if *all_combinations_generated.read().unwrap() {
		    write_worker_snapshot(&passed, None, thread::current().name().unwrap(), &snapshot_directory);
		    return;
		}
		// write a snapshot
		if stop_for_snapshot.load(MemoryOrdering::SeqCst) {
		    let mut ready_to_write_snapshot = false;		    
		    {
			let mut stopped_worker_count = workers_stopped_for_snapshot.lock().unwrap();
			// this is the last worker thread stopping - verify that the generator has already stopped
			if *stopped_worker_count == WORKER_THREAD_COUNT - 1  {
			    let generator_stopped = generator_thread_stopped.read().unwrap();
			    if *generator_stopped {
				pass_count_tx.send(batch_pass_count);				
				
				// notify the generator thread that all queues have been empty
				let mut queues_empty_predicate = queues_empty.0.lock().unwrap();
				*queues_empty_predicate = true;
				let ref queues_empty_cvar = queues_empty.1;
				queues_empty_cvar.notify_all();
				ready_to_write_snapshot = true;

				// workers stopped for snapshot is only used here - can safely reset
				*stopped_worker_count = 0;
			    }
			    // don't allow stopping if we were to be the last worker thread to stop
			    // but the global thread has not stopped yet (busy wait)
			}
			// not the last worker thread stopping, so can freely stop and write snapshot
			else {
			    *stopped_worker_count += 1;
			    pass_count_tx.send(batch_pass_count);
			    ready_to_write_snapshot = true;
			}

			// drop stoped_worker_count mutex
		    }

		    if ready_to_write_snapshot {
			batch_pass_count = 0;
			write_worker_snapshot(&passed, Some(*snapshot_number.read().unwrap()), thread::current().name().unwrap(),
					      &snapshot_directory);

			// block until the global thread has completed the snapshot
			let mut snapshot_complete_predicate = snapshot_complete.0.lock().unwrap();
			let ref snapshot_complete_condvar = snapshot_complete.1;
			while !*snapshot_complete_predicate {
			    snapshot_complete_predicate = snapshot_complete_condvar.wait(snapshot_complete_predicate).unwrap();
			}
		    }
		}
		// workers are running ahead of the global thread - block for a while to limit context switches
		// TODO consider exponential backoff here if this happens in practice
		else {
		    println!("All queues are dry but not all letter combinations are exhausted. Sleeping.");
		    thread::sleep(time::Duration::from_secs(1));
		}
	    }
	}
    }
}

// TODO verify ability to deserialize these
/// write a snapshot from a worker thread. consists of the passed combinations along with their scores.
fn write_worker_snapshot(passing_combinations: &Vec<(LetterCombination, u32)>,
			 snapshot_number: Option<u64>, thread_name: &str,
			 snapshot_directory: &Arc<&Path>){
    let encoded_passing = bincode::serialize(passing_combinations).unwrap();
    let snapshot_id: String;

    if snapshot_number.is_some(){
	snapshot_id = snapshot_number.unwrap().to_string();
    }
    else {
	snapshot_id = "FINAL".to_string();
    }

    let snapshot_name = thread_name.to_owned() + &snapshot_id;
    let mut worker_snapshot_path = snapshot_directory.to_path_buf();
    worker_snapshot_path.push(snapshot_name);
    
    // intentionally panic if the filesystem operation fails	    
    fs::write(worker_snapshot_path, encoded_passing).unwrap();
}

/// get the summed scores of all words of length <= 16
/// that can be built from a given combination of 16 letters represented
/// as frequency counts
// this is an upper bound on the score of any board that is a  permutation of these letters
// TODO test this - correctness issue. idea: just used a custom-made reduced dimensionality trie with
// interesting structure
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
			    // TODO verify -3 indexing still appropriate
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
