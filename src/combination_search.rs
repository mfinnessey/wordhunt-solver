use crate::letter::Letter;
use crate::utilities::POINTS;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::ops::{Index, IndexMut};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use std::{fmt, fs, iter, thread, time};
use trie_rs::inc_search::Answer;
use trie_rs::Trie;

use crossbeam::thread::ScopedJoinHandle;
use crossbeam_deque::{Injector, Stealer, Worker};
use crossbeam_utils::thread as crossbeam_thread;
use std::sync::atomic::{AtomicBool, Ordering as MemoryOrdering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc::channel, Arc, Condvar, Mutex, RwLock};

// TODO benchmark this if need be
/// the maximum batch size that a worker thread will pull from the global queue at once
const PULL_LIMIT: usize = 10_000;
/// the (approximate) maximum number of elements in the global queue before the generator
/// threaed will sleep
const MAX_TARGET_GLOBAL_QUEUE_LEN: usize = 2 * (WORKER_THREAD_COUNT as usize) * PULL_LIMIT;

const ALPHABET_LENGTH: usize = 26;
const TILE_COUNT: usize = 16;

/// the starting value of the iterator over all letter combinations
pub const ALL_A_FREQUENCIES: [u8; ALPHABET_LENGTH] = [
    TILE_COUNT as u8,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
];

/// the amount of entries that must be in the global queue before the worker threads
/// are released
const STARTUP_PUSH_COUNT: u32 = 200_000;

/// the number of worker threads to run
const WORKER_THREAD_COUNT: u8 = 16;

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct LetterCombination {
    frequencies: [u8; ALPHABET_LENGTH],
}

impl LetterCombination {
    pub fn new(frequencies: [u8; ALPHABET_LENGTH]) -> Self {
        let combination_tile_count: usize = frequencies.iter().map(|x| *x as usize).sum();
        if combination_tile_count != TILE_COUNT {
            panic!(
                "Attempted to create invalid letter combination with {} tiles instead of {} tiles",
                combination_tile_count, TILE_COUNT
            );
        }

        Self { frequencies }
    }
}

impl Ord for LetterCombination {
    fn cmp(&self, other: &Self) -> Ordering {
        for (freq1, freq2) in self.frequencies.iter().zip(other.frequencies.iter()) {
            if freq1 > freq2 {
                return Ordering::Less;
            }
            if freq2 > freq1 {
                return Ordering::Greater;
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
            panic!(
                "Index {} for LetterFrequencies is outside the [0, 25]!",
                idx
            )
        }
        &self.frequencies[idx]
    }
}

impl IndexMut<usize> for LetterCombination {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        if idx > ALPHABET_LENGTH - 1 {
            panic!(
                "Index {} for LetterFrequencies is outside the [0, 25]!",
                idx
            )
        }
        &mut self.frequencies[idx]
    }
}

impl Eq for LetterCombination {}

impl From<LetterCombination> for [u8; ALPHABET_LENGTH] {
    fn from(frequencies: LetterCombination) -> Self {
        frequencies.frequencies
    }
}

impl From<[u8; ALPHABET_LENGTH]> for LetterCombination {
    fn from(frequencies: [u8; ALPHABET_LENGTH]) -> Self {
        LetterCombination::new(frequencies)
    }
}

impl fmt::Display for LetterCombination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut letters = ['a'; TILE_COUNT];
        let mut letters_idx = 0;
        for (frequencies_idx, letter_frequency) in self.frequencies.iter().enumerate() {
            for _ in 0..*letter_frequency {
                letters[letters_idx] = (b'A' + frequencies_idx as u8) as char;
                letters_idx += 1;
            }
        }

        let concatenated: String = letters.iter().collect();
        write!(f, "{}", concatenated)
    }
}

/// generates combinations with replacement for a specified C(N, R) as frequency
/// counts for the N options.
/// N is the number of elements to select from
/// R is the number of selections to make.
// this is implemented abstractly here primarily for practical unit testing.
// substantial inspiration taken from https://github.com/olivercalder/combinatorial/
pub struct SequentialCombinationGenerator<const N: usize, const R: usize> {
    /// indices into a (conceptual) array consisting of the elements that we select from
    element_indices: [usize; R],
    /// set iff all combinations (with replacement) have been iterated over
    completed: bool,
}

impl<const N: usize, const R: usize> SequentialCombinationGenerator<N, R> {
    pub fn new(indices: [usize; R]) -> Self {
        // verify that indices array is non-decreasing.
        for i in 1..R {
            if indices[i] < indices[i - 1] {
                panic!("Attempted to create sequential combination generator with decreasing element_indices array. Indices {} and {} are [{}, {}].",
		i - 1, i, indices[i - 1], indices[i]);
            }
        }
        Self {
            element_indices: indices,
            completed: false,
        }
    }

    /// create the next set of indices from the current set of indices
    fn advance_indices(&mut self) {
        // iteration scheme generates all non-decreasing indices arrays
        for i in (0..R).rev() {
            // look for the first index that is not maxed out
            if self.element_indices[i] < N - 1 {
                // bump all subsequent indices to the value of the first non-maxed index plus one
                let next_index = self.element_indices[i] + 1;
                for j in i..R {
                    self.element_indices[j] = next_index;
                }
                return;
            }
        }

        self.completed = true;
    }
}

impl<const N: usize, const R: usize> Iterator for SequentialCombinationGenerator<N, R> {
    type Item = [u8; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.completed {
            return None;
        }

        // map indices into (conceptual) array of elements to the frequency
        // counts of those elements
        let mut frequencies = [0u8; N];
        for index in self.element_indices {
            frequencies[index] += 1;
        }

        self.advance_indices();

        Some(frequencies)
    }
}

/// generates combinations of letters from the given starting_Frequencies "onwards" in sequence.
/// the sequence is defined using the iteration scheme of SequentialCombinationGenerator above.
pub struct SequentialLetterCombinationGenerator {
    generator: SequentialCombinationGenerator<ALPHABET_LENGTH, TILE_COUNT>,
}

impl SequentialLetterCombinationGenerator {
    fn new(starting_frequencies: LetterCombination) -> Self {
        let indices = letter_combination_to_element_indices(starting_frequencies);
        Self {
            generator: SequentialCombinationGenerator::new(indices),
        }
    }
}

/// map letter frequencies into indices into an (conceptual) array of the letters
fn letter_combination_to_element_indices(lc: LetterCombination) -> [usize; TILE_COUNT] {
    let mut indices = [0usize; TILE_COUNT];
    let mut indices_idx = 0;
    for (letter_idx, mut letter_frequency) in
        <[u8; ALPHABET_LENGTH]>::from(lc).into_iter().enumerate()
    {
        while letter_frequency > 0 {
            indices[indices_idx] = letter_idx;
            indices_idx += 1;
            letter_frequency -= 1;
        }
    }
    indices
}

impl Iterator for SequentialLetterCombinationGenerator {
    type Item = LetterCombination;

    fn next(&mut self) -> Option<Self::Item> {
        self.generator.next().map(|frequencies| frequencies.into())
    }
}

// TODO replace LetterCombinationGenerator with a generic for anything that is an iterator over
// letter combinations
/// evaluates all combinations of letters using a given wordlist using a given metric
pub struct CombinationEvaluator<'a, CombinationGenerator: Iterator<Item = LetterCombination>> {
    word_list: &'a Trie<Letter>,
    combinations: CombinationGenerator,
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
    snapshot_complete: Arc<(Arc<Mutex<bool>>, Condvar)>,
    /// where to write snapshots
    snapshots_directory: Arc<PathBuf>,
}

impl<'a> WorkerInformation<'a> {
    // this object's constructor is designed to simplify the signature of evaluate_combinations
    #[allow(clippy::too_many_arguments)]
    fn new(
        word_list: &'a Trie<Letter>,
        metric: fn(&Trie<Letter>, LetterCombination) -> u32,
        target: u32,
        local: Worker<LetterCombination>,
        global: Arc<Injector<LetterCombination>>,
        stealers: Arc<Vec<Stealer<LetterCombination>>>,
        pass_count_tx: Sender<u64>,
        all_combinations_generated: Arc<RwLock<bool>>,
        stop_for_snapshot: &'a AtomicBool,
        snapshot_number: Arc<RwLock<u64>>,
        generator_thread_stopped: Arc<RwLock<bool>>,
        queues_empty: Arc<(Mutex<bool>, Condvar)>,
        workers_stopped_for_snapshot: Arc<Mutex<u8>>,
        snapshot_complete: Arc<(Arc<Mutex<bool>>, Condvar)>,
        snapshots_directory: Arc<PathBuf>,
    ) -> Self {
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
            snapshots_directory,
        }
    }
}

impl<'a, Generator: Iterator<Item = LetterCombination> + std::marker::Send>
    CombinationEvaluator<'a, Generator>
{
    /// spawn threads to create combination evaluation architecture
    pub fn check_combinations(self, snapshot_frequency: u64) {
        // create snapshots directory with time-based unique identifier
        let cur_time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let path_string = "snapshots-".to_string() + &cur_time.to_string();
        let snapshots_directory = Path::new(&path_string);
        match fs::create_dir(snapshots_directory) {
            Ok(_) => (),
            Err(_) => panic!("Could not create snapshots directory"),
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
        let snapshot_complete_mutex = Arc::new(Mutex::new(false));
        let snapshot_complete_condvar = Condvar::new();
        let snapshot_complete =
            Arc::new((snapshot_complete_mutex.clone(), snapshot_complete_condvar));

        let stealers_vec_ref = Arc::new(stealers);
        let global_queue_ref = Arc::new(global_queue);
        let snapshots_directory_ref = Arc::new(snapshots_directory.to_owned());
        let execution_completed = Arc::new(Mutex::new(false));

        crossbeam_thread::scope(|s| {
            // spawn the generator thread
            let global_queue_ref_for_generator = global_queue_ref.clone();
            let all_combinations_generated = &all_combinations_generated;
            let stop_for_snapshot = &stop_for_snapshot;
            let snapshot_number = &snapshot_number;
            let generator_thread_stopped = &generator_thread_stopped;
            let queues_empty = &queues_empty;
            let snapshot_complete = &snapshot_complete;
            let snapshots_directory_ref = &snapshots_directory_ref;
            let generator_handle = s.spawn(move |_| {
                generate_combinations(
                    self.combinations,
                    global_queue_ref_for_generator,
                    Arc::clone(all_combinations_generated),
                    stop_for_snapshot,
                    Arc::clone(snapshot_number),
                    Arc::clone(generator_thread_stopped),
                    Arc::clone(queues_empty),
                    Arc::clone(snapshot_complete),
                    pass_count_rx,
                    Arc::clone(snapshots_directory_ref),
                )
            });

            // give the generator thread time to generate some combinations to avoid the worker
            // threads just thrashing on context switches to start
            thread::sleep(time::Duration::from_secs(1));
            let mut worker_threads: Vec<ScopedJoinHandle<'_, ()>> = Vec::new();
            for thread_num in 0..self.num_worker_threads {
                // using unwrap as is safe by construction - we have pushed num_worker_threads elements
                let thread_worker = workers.pop().unwrap();
                let stealers_ref = stealers_vec_ref.clone();
                let global_ref = global_queue_ref.clone();

                // TODO clone this struct
                let worker_information = WorkerInformation::new(
                    self.word_list,
                    self.metric,
                    self.target,
                    thread_worker,
                    global_ref,
                    stealers_ref,
                    pass_count_tx.clone(),
                    Arc::clone(all_combinations_generated),
                    stop_for_snapshot,
                    Arc::clone(snapshot_number),
                    Arc::clone(generator_thread_stopped),
                    Arc::clone(queues_empty),
                    Arc::clone(&workers_stopped_for_snapshot),
                    Arc::clone(snapshot_complete),
                    Arc::clone(snapshots_directory_ref),
                );
                let handle = s
                    .builder()
                    .name(thread_num.to_string() + "thread")
                    .spawn(move |_| evaluate_combinations(worker_information))
                    .unwrap();
                worker_threads.push(handle);
            }

            // spawn snapshot initiator thread
            let execution_completed_temp_ref = &execution_completed;
            let snapshot_handle = s.spawn(move |_| {
                trigger_snapshots(
                    stop_for_snapshot,
                    Arc::clone(snapshot_number),
                    Arc::clone(&snapshot_complete_mutex),
                    Arc::clone(execution_completed_temp_ref),
                    snapshot_frequency,
                )
            });

            if generator_handle.join().is_err() {
                panic!("Generator thread paniced!")
            }

            for worker_thread_handle in worker_threads {
                if worker_thread_handle.join().is_err() {
                    panic!("Worker thread paniced!")
                }
            }

            // execution is complete after the last worker terminates
            *execution_completed.lock().unwrap() = true;
            if snapshot_handle.join().is_err() {
                panic!("Snapshot thread paniced!")
            }
        })
        .unwrap(); // unwrap as we would just panic anyways
    }
}

/// implementation for raw stage 1 search
impl<'a> CombinationEvaluator<'a, SequentialLetterCombinationGenerator> {
    pub fn new(
        word_list: &'a Trie<Letter>,
        starting_frequencies: LetterCombination,
        metric: fn(&Trie<Letter>, LetterCombination) -> u32,
        target: u32,
        num_worker_threads: usize,
    ) -> Self {
        Self {
            word_list,
            combinations: SequentialLetterCombinationGenerator::new(starting_frequencies),
            metric,
            target,
            num_worker_threads,
        }
    }
}

// TODO comment out the design
// TODO SPAWN THIS IN ITS OWN THREAD
fn trigger_snapshots(
    stop_for_snapshot: &AtomicBool,
    snapshot_number: Arc<RwLock<u64>>,
    snapshot_complete: Arc<Mutex<bool>>,
    execution_completed: Arc<Mutex<bool>>,
    snapshot_frequency_secs: u64,
) {
    const CHECK_FOR_COMPLETION_INTERVAL_SECS: u64 = 10;
    let sleep_loop_count = snapshot_frequency_secs / CHECK_FOR_COMPLETION_INTERVAL_SECS;
    loop {
        // take snapshots every 15 minutes, waking every 10 seconds to check if execution has
        // completed
        // TODO write this more cleanly
        for _ in 0..sleep_loop_count {
            if *execution_completed.lock().unwrap() {
                return;
            }

            thread::sleep(time::Duration::from_secs(
                CHECK_FOR_COMPLETION_INTERVAL_SECS,
            ));
        }

        println!("Taking snapshot.");
        // mark for the other threads to begin taking a snapshot
        *snapshot_complete.lock().unwrap() = false;
        stop_for_snapshot.store(true, MemoryOrdering::SeqCst);

        // confirm that the global thread has completed the previous snapshot before
        // triggering another snapshot
        while !*snapshot_complete.lock().unwrap() {
            thread::sleep(time::Duration::from_millis(200));
        }

        *snapshot_number.write().unwrap() += 1;
    }
}

// thread needs a lot of state passed in. not bundling into a struct as there is a singular generator thread
#[allow(clippy::too_many_arguments)]
fn generate_combinations(
    combinations: impl Iterator<Item = LetterCombination>,
    queue: Arc<Injector<LetterCombination>>,
    all_combinations_generated: Arc<RwLock<bool>>,
    stop_for_snapshot: &AtomicBool,
    snapshot_number: Arc<RwLock<u64>>,
    generator_thread_stopped: Arc<RwLock<bool>>,
    queues_empty: Arc<(Mutex<bool>, Condvar)>,
    snapshot_complete: Arc<(Arc<Mutex<bool>>, Condvar)>,
    pass_count_rx: Receiver<u64>,
    snapshots_directory: Arc<PathBuf>,
) {
    const TOTAL_COMBINATIONS_COUNT: u64 = 103_077_446_706;

    let start_time = SystemTime::now();

    let mut overall_combinations_generated_count: u64 = 0;
    let mut last_snapshot_combination_count: u64 = 0;
    let mut overall_pass_count: u64 = 0;

    // not spinning up to start
    let mut spinning_up = false;
    let mut spin_up_pushes_remaining = 0;

    for combination in combinations {
        overall_combinations_generated_count += 1;

        // block the generator thread if the queue is ludicrously big.
        let mut stopped = false;
        while overall_combinations_generated_count % (MAX_TARGET_GLOBAL_QUEUE_LEN as u64) == 0
            && queue.len() >= MAX_TARGET_GLOBAL_QUEUE_LEN
        {
            if !stopped {
                println!("Generator ran ahead, sleeping for now.");
                stopped = true;
            }

            // jump out of here to take the snapshot if need be
            if stop_for_snapshot.load(MemoryOrdering::SeqCst) {
                break;
            }
            thread::sleep(time::Duration::from_secs(1));
        }

        if stopped {
            println!("Generator thread resuming after running ahead.");
        }

        // can consider checking every x iteratons only if the atomic cas
        // winds up being too expensive (guessing that it can be parallelized / branch
        // predicted efficiently enough)
        if !spinning_up && stop_for_snapshot.load(MemoryOrdering::SeqCst) {
            // notify worker threads that we have stopped generating new combinations
            {
                let mut generation_stopped = generator_thread_stopped.write().unwrap();
                *generation_stopped = true;
            }

            println!(
                "Generator stopped for snapshot with {} items remaining in global queue",
                queue.len()
            );

            // write the next combination to disk (the snapshot consists of the results up to this
            // point and the next thing to be considered)
            let encoded_next = bincode::serialize(&combination).unwrap();

            let snapshot_name = "gen".to_owned() + &(*snapshot_number.read().unwrap()).to_string();
            let mut generator_snapshot_path = snapshots_directory.to_path_buf();
            generator_snapshot_path.push(snapshot_name);

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

            // reset queues emptied for next snapshot
            *queues_emptied = false;

            // check that a snapshot invariant (empty queues at completion) has been satisfied
            if queue.len() != 0 {
                panic!(
                    "Global queue was not empty ({} combination(s)) at end of snapshot",
                    queue.len()
                );
            } else {
                println!("Invariant satisfied.");
            }

            // defer on marking snapshot as complete to prevent another snapshot from being
            // triggered before we've spun back up (unlikely but an annoyingly possible case)

            // compute and output progress statistics
            let batch_count =
                overall_combinations_generated_count - last_snapshot_combination_count;
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

            let batch_pass_rate: f64 =
	    // prevent divide by zero
	    if batch_count != 0 {
		(batch_pass_count as f64) / (batch_count as f64)
	    }
	    else {
		0.0
	    };

            overall_pass_count += batch_pass_count;
            let overall_fail_count: u64 = overall_combinations_generated_count - overall_pass_count;
            let overall_pass_rate: f64 =
	    // prevent divide by zero
	    if overall_combinations_generated_count != 0 {
		(overall_pass_count as f64) / (overall_combinations_generated_count as f64)
	    }
	    else {
		0.0
	    };

            let remaining_combinations_count =
                TOTAL_COMBINATIONS_COUNT - overall_combinations_generated_count;
            let remaining_combinations_percentage: f64 = 100.0
                * (1.0 - (remaining_combinations_count as f64) / (TOTAL_COMBINATIONS_COUNT as f64));

            let elapsed_time = SystemTime::now().duration_since(start_time);
            let elapsed_time_hours = match elapsed_time {
                Ok(time) => (time.as_secs_f64()) / 3600.0,
                Err(_) => 0.0,
            };

            // ratio of remaining work to completed work applied to elapsed time
            let hours_per_combination =
                elapsed_time_hours / (overall_combinations_generated_count as f64);
            let estimated_time_remaining_hours =
                (hours_per_combination * TOTAL_COMBINATIONS_COUNT as f64) - elapsed_time_hours;

            println!("*****");
            println!("Wrote snapshot to disk.");
            println!(
                "Batch: Total {} | Pass {} | Fail {} | Pass Rate {:.2}",
                batch_count, batch_pass_count, batch_fail_count, batch_pass_rate
            );
            println!(
                "Overall: Total {} | Pass {} | Fail {} | Pass Rate {:.2}",
                overall_combinations_generated_count,
                overall_pass_count,
                overall_fail_count,
                overall_pass_rate
            );
            println!(
                "Remaining: {} ({:.2}%) of Total {})",
                remaining_combinations_count,
                remaining_combinations_percentage,
                TOTAL_COMBINATIONS_COUNT
            );
            println!(
                "Ran for {:.1} hours. Estimate {:.1} hours remaining",
                elapsed_time_hours, estimated_time_remaining_hours
            );
            println!("Next combination is: {}.", combination);
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
                if stop_for_snapshot.load(MemoryOrdering::SeqCst) {
                    stop_for_snapshot.store(false, MemoryOrdering::SeqCst);
                    // AND RELEASE THE HOUNDS
                    *snapshot_complete.0.lock().unwrap() = true;
                    snapshot_complete.1.notify_all();
                }
            }
        }

        queue.push(combination);
    }

    println!("Generated all combinations");
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
    let snapshots_directory = worker_information.snapshots_directory;

    let mut passed = Vec::new();
    let mut batch_pass_count: u64 = 0;

    loop {
        // modified from crossbeam::deque docs
        // pop a task from the local queue, if not empty.
        let combination = local.pop().or_else(|| {
            // otherwise, we need to look for a task elsewhere.
            iter::repeat_with(|| {
                // try stealing a batch of tasks from the global queue.
                global
                    .steal_batch_with_limit_and_pop(&local, PULL_LIMIT)
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
            }
            None => {
                // write the final snapshot (final results) to disk
                if *all_combinations_generated.read().unwrap() {
                    write_worker_snapshot(
                        &passed,
                        None,
                        thread::current().name().unwrap(),
                        &snapshots_directory,
                    );
                    return;
                }
                // write a snapshot
                if stop_for_snapshot.load(MemoryOrdering::SeqCst) {
                    let mut ready_to_write_snapshot = false;
                    {
                        let mut stopped_worker_count = workers_stopped_for_snapshot.lock().unwrap();
                        // this is the last worker thread stopping - verify that the generator has already stopped
                        if *stopped_worker_count == WORKER_THREAD_COUNT - 1 {
                            let generator_stopped = generator_thread_stopped.read().unwrap();
                            if *generator_stopped {
                                // ignore failures on writing statistics - not important to correctness
                                let _ = pass_count_tx.send(batch_pass_count);

                                // notify the generator thread that all queues have been empty
                                let mut queues_empty_predicate = queues_empty.0.lock().unwrap();
                                *queues_empty_predicate = true;
                                let queues_empty_cvar = &queues_empty.1;
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
                            // ignore failures on writing statistics - not important to correctness
                            let _ = pass_count_tx.send(batch_pass_count);
                            ready_to_write_snapshot = true;
                        }

                        // drop stoped_worker_count mutex
                    }

                    if ready_to_write_snapshot {
                        batch_pass_count = 0;
                        write_worker_snapshot(
                            &passed,
                            Some(*snapshot_number.read().unwrap()),
                            thread::current().name().unwrap(),
                            &snapshots_directory,
                        );

                        // block until the global thread has completed the snapshot
                        let mut snapshot_complete_predicate = snapshot_complete.0.lock().unwrap();
                        let snapshot_complete_condvar = &snapshot_complete.1;
                        while !*snapshot_complete_predicate {
                            snapshot_complete_predicate = snapshot_complete_condvar
                                .wait(snapshot_complete_predicate)
                                .unwrap();
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
fn write_worker_snapshot(
    passing_combinations: &Vec<(LetterCombination, u32)>,
    snapshot_number: Option<u64>,
    thread_name: &str,
    snapshots_directory: &Arc<PathBuf>,
) {
    let encoded_passing = bincode::serialize(passing_combinations).unwrap();
    let snapshot_id: String = if let Some(num) = snapshot_number {
        num.to_string()
    } else {
        "FINAL".to_string()
    };

    let snapshot_name = thread_name.to_owned() + &snapshot_id;
    let mut worker_snapshot_path = snapshots_directory.to_path_buf();
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
pub fn get_combination_score(
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
                    Some(Answer::PrefixAndMatch) | Some(Answer::Match) => {
                        if new_search.prefix_len() >= 3 {
                            // TODO verify -3 indexing still appropriate
                            score += POINTS[new_search.prefix_len() - 3];
                        }

                        let mut new_counts = remaining_counts;
                        new_counts[i] -= 1;
                        queue.push_back((new_search, new_counts));
                    }
                    // non-viable continuation
                    None => (),
                }
            }
        }
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // AAAABBBBCCCCEEZZ
    const FREQUENCIES: [u8; ALPHABET_LENGTH] = [
        4, 4, 4, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    ];

    #[test]
    fn test_sequential_combination_generator() {
        // test with 5 choose 3 without loss of generality
        let initial_frequencies = [0, 0, 0];
        let generator: SequentialCombinationGenerator<5, 3> =
            SequentialCombinationGenerator::new(initial_frequencies);

        const EXPECTED_FREQUENCIES: [[u8; 5]; 35] = [
            [3, 0, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [2, 0, 1, 0, 0],
            [2, 0, 0, 1, 0],
            [2, 0, 0, 0, 1],
            [1, 2, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 0, 2, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 2, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 0, 2],
            [0, 3, 0, 0, 0],
            [0, 2, 1, 0, 0],
            [0, 2, 0, 1, 0],
            [0, 2, 0, 0, 1],
            [0, 1, 2, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 0, 2, 0],
            [0, 1, 0, 1, 1],
            [0, 1, 0, 0, 2],
            [0, 0, 3, 0, 0],
            [0, 0, 2, 1, 0],
            [0, 0, 2, 0, 1],
            [0, 0, 1, 2, 0],
            [0, 0, 1, 0, 2],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 3, 0],
            [0, 0, 0, 2, 1],
            [0, 0, 0, 1, 2],
            [0, 0, 0, 0, 3],
        ];

        let expected = HashSet::from(EXPECTED_FREQUENCIES);
        let actual: Vec<[u8; 5]> = generator.collect();

        // verify generation length and elements (generation order doesn't matter)
        assert_eq!(actual.len(), 35);
        assert_eq!(
            expected
                .symmetric_difference(&HashSet::from_iter(actual.into_iter()))
                .count(),
            0
        );
    }

    #[test]
    fn test_letter_combination_to_element_indices() {
        let lc = LetterCombination::new(FREQUENCIES);

        const EXPECTED: [usize; TILE_COUNT] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 25, 25];
        let actual = letter_combination_to_element_indices(lc);
        assert_eq!(actual, EXPECTED);
    }

    #[test]
    fn test_letter_combination_display() {
        let lc = LetterCombination::new(FREQUENCIES);

        const EXPECTED: &str = "AAAABBBBCCCCEEZZ";
        let actual = format!("{}", lc);
        assert_eq!(actual, EXPECTED);
    }
}
