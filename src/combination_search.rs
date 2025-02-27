use crate::letter::Letter;
use crate::letter_combination::LetterCombination;
use crate::utilities::{ALL_A_FREQUENCIES, ALPHABET_LENGTH, POINTS, TILE_COUNT};
use std::collections::VecDeque;
use std::path::Path;
use std::time::SystemTime;
use std::{fs, iter, thread, time};
use trie_rs::inc_search::Answer;
use trie_rs::Trie;

use crossbeam::thread::ScopedJoinHandle;
use crossbeam_deque::{Injector, Stealer, Worker};
use crossbeam_utils::thread as crossbeam_thread;
use std::sync::atomic::{AtomicBool, Ordering as MemoryOrdering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

// TODO benchmark this if need be
/// the maximum batch size that a worker thread will pull from the global queue at once
const PULL_LIMIT: usize = 10_000;

/// time to wait between starting the generator thread and the worker threads to limit
/// thrashing from context-switches on an under-filled global queue.
const SPIN_UP_WAIT: time::Duration = time::Duration::from_millis(200);

/// data generated for a passing combination
type PassMsg = (LetterCombination, u32);

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
    /// queue accessors
    local: Worker<LetterCombination>,
    global: Arc<Injector<LetterCombination>>,
    stealers: Arc<Vec<Stealer<LetterCombination>>>,
    /// where the worker stores its passing combinations for a given batch (to eventually be snapshotted by the generator thread)
    pass_vector: Arc<Mutex<Vec<PassMsg>>>,
    /// synchronization stuff
    /// all combinations have been generated - the condition that will ultimately terminate each thread
    all_combinations_generated: Arc<RwLock<bool>>,
    /// the thread should stop for a snapshot
    stop_for_snapshot: &'a AtomicBool,
    /// the generator thread has (temporarily) stopped generating new combinations (for the purposes of a snapshot)
    generator_thread_stopped: Arc<RwLock<bool>>,
    /// set by the last worker thread to stop for a snapshot to signal the generator thread that the workers have stopped (i.e.
    /// the queues are empty)
    workers_stopped: Arc<(Mutex<bool>, Condvar)>,
    /// the number of worker threads that are running - used to determine which thread is the last to stop for a snapshot.
    num_running_workers: Arc<Mutex<usize>>,
    /// set by the snapshot thread to notify the worker threads that the snapshot has been completed
    snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
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
        pass_vector: Arc<Mutex<Vec<PassMsg>>>,
        all_combinations_generated: Arc<RwLock<bool>>,
        stop_for_snapshot: &'a AtomicBool,
        generator_thread_stopped: Arc<RwLock<bool>>,
        workers_stopped: Arc<(Mutex<bool>, Condvar)>,
        num_running_workers: Arc<Mutex<usize>>,
        snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
    ) -> Self {
        Self {
            word_list,
            metric,
            target,
            local,
            global,
            pass_vector,
            stealers,
            all_combinations_generated,
            stop_for_snapshot,
            generator_thread_stopped,
            workers_stopped,
            num_running_workers,
            snapshot_complete,
        }
    }
}

impl<'a, Generator: Iterator<Item = LetterCombination> + std::marker::Send>
    CombinationEvaluator<'a, Generator>
{
    /// spawn threads to create combination evaluation architecture
    pub fn begin_combination_evaluation(self, snapshot_frequency: u64) {
        let global_queue = Injector::new();

        let mut workers: Vec<Worker<LetterCombination>> = Vec::new();
        let mut stealers: Vec<Stealer<LetterCombination>> = Vec::new();
        let mut stopped_mutexes: Vec<Mutex<()>> = Vec::new();
        let mut pass_vectors: Vec<Arc<Mutex<Vec<PassMsg>>>> = Vec::new();

        for _ in 0..self.num_worker_threads {
            let w = Worker::new_fifo();
            let stealer = w.stealer();
            let pass_vector = Arc::new(Mutex::new(Vec::new()));

            workers.push(w);
            stealers.push(stealer);
            stopped_mutexes.push(Mutex::new(()));
            pass_vectors.push(pass_vector);
        }

        let next_combination = Arc::new(Mutex::new(LetterCombination::new(ALL_A_FREQUENCIES)));
        let batch_count = Arc::new(Mutex::new(0));

        let all_combinations_generated = Arc::new(RwLock::new(false));
        let stop_for_snapshot = AtomicBool::new(false);
        let generator_thread_stopped = Arc::new(RwLock::new(false));

        let workers_stopped_mutex = Mutex::new(false);
        let workers_stopped_condvar = Condvar::new();
        let workers_stopped = Arc::new((workers_stopped_mutex, workers_stopped_condvar));
        let num_running_workers: Arc<Mutex<usize>> = Arc::new(Mutex::new(self.num_worker_threads));
        let workers_snapshot_complete_mutex = Mutex::new(false);
        let workers_snapshot_complete_condvar = Condvar::new();
        let workers_snapshot_complete = Arc::new((
            workers_snapshot_complete_mutex,
            workers_snapshot_complete_condvar,
        ));
        let generator_snapshot_complete_mutex = Mutex::new(false);
        let generator_snapshot_complete_condvar = Condvar::new();
        let generator_snapshot_complete = Arc::new((
            generator_snapshot_complete_mutex,
            generator_snapshot_complete_condvar,
        ));

        let stealers_vec_ref = Arc::new(stealers);
        let global_queue_ref = Arc::new(global_queue);
        let execution_completed = Arc::new(Mutex::new(false));

        crossbeam_thread::scope(|s| {
            // spawn the generator thread
            let global_queue_ref_for_generator = global_queue_ref.clone();
            let all_combinations_generated = &all_combinations_generated;
            let stop_for_snapshot = &stop_for_snapshot;
            let generator_thread_stopped = &generator_thread_stopped;
            let workers_stopped = &workers_stopped;
            let generator_snapshot_complete = &generator_snapshot_complete;
            let next_combination = &next_combination;
            let batch_count = &batch_count;
            let generator_handle = s.spawn(move |_| {
                generate_combinations(
                    self.combinations,
                    global_queue_ref_for_generator,
                    2 * self.num_worker_threads * PULL_LIMIT,
                    Arc::clone(all_combinations_generated),
                    stop_for_snapshot,
                    Arc::clone(generator_thread_stopped),
                    Arc::clone(generator_snapshot_complete),
                    Arc::clone(next_combination),
                    Arc::clone(batch_count),
                )
            });

            thread::sleep(SPIN_UP_WAIT);

            let mut worker_threads: Vec<ScopedJoinHandle<'_, ()>> = Vec::new();
            println!("Spawning {} worker threads.", self.num_worker_threads);
            for thread_num in 0..self.num_worker_threads {
                // using unwrap as is safe by construction - we have pushed num_worker_threads elements
                let thread_worker = workers.pop().unwrap();
                let stealers_ref = stealers_vec_ref.clone();
                let global_ref = global_queue_ref.clone();
                let pass_vector: Arc<Mutex<Vec<PassMsg>>> = Arc::clone(&pass_vectors[thread_num]);

                // TODO clone this struct and separate non-shared stuff out
                let worker_information = WorkerInformation::new(
                    self.word_list,
                    self.metric,
                    self.target,
                    thread_worker,
                    global_ref,
                    stealers_ref,
                    pass_vector.clone(),
                    Arc::clone(all_combinations_generated),
                    stop_for_snapshot,
                    Arc::clone(generator_thread_stopped),
                    Arc::clone(workers_stopped),
                    Arc::clone(&num_running_workers),
                    Arc::clone(&workers_snapshot_complete),
                );
                let handle = s
                    .builder()
                    .name(thread_num.to_string())
                    .spawn(move |_| evaluate_combinations(worker_information))
                    .unwrap();
                worker_threads.push(handle);
            }

            // spawn snapshot thread
            let execution_completed = &execution_completed;
            let snapshot_handle = s.spawn(move |_| {
                take_snapshots(
                    snapshot_frequency,
                    Arc::clone(execution_completed),
                    stop_for_snapshot,
                    Arc::clone(workers_stopped),
                    Arc::clone(generator_thread_stopped),
                    Arc::clone(&workers_snapshot_complete),
                    Arc::clone(generator_snapshot_complete),
                    pass_vectors,
                    Arc::clone(&num_running_workers),
                    self.num_worker_threads,
                    global_queue_ref,
                    Arc::clone(next_combination),
                    Arc::clone(batch_count),
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

struct ProgressStatistics {
    start_time: SystemTime,
    pass_count: u64,
    evaluated_count: u64,
    batch_number: u64,
}

impl ProgressStatistics {
    fn new(start_time: SystemTime) -> Self {
        Self {
            start_time,
            pass_count: 0,
            evaluated_count: 0,
            batch_number: 0,
        }
    }

    fn update_with_batch(
        &mut self,
        batch_pass_count: u64,
        batch_evaluated_count: u64,
        next_combination: &LetterCombination,
        print_statistics: bool,
    ) {
        // update state
        self.pass_count += batch_pass_count;
        self.evaluated_count += batch_evaluated_count;
        self.batch_number += 1;

        if !print_statistics {
            return;
        }

        // print statistics
        const TOTAL_COMBINATIONS_COUNT: u64 = 103_077_446_706;

        let batch_fail_count = batch_evaluated_count - batch_pass_count;

        let batch_pass_rate: f64 =
	// prevent divide by zero
	    if batch_evaluated_count != 0 {
		(batch_pass_count as f64) / (batch_evaluated_count as f64)
	    }
	else {
	    0.0
	};

        let overall_fail_count: u64 = self.evaluated_count - self.pass_count;
        let overall_pass_rate: f64 =
	// prevent divide by zero
	    if self.evaluated_count != 0 {
		(self.pass_count as f64) / (self.evaluated_count as f64)
	    }
	else {
	    0.0
	};

        let remaining_combinations_count = TOTAL_COMBINATIONS_COUNT - self.evaluated_count;
        let remaining_combinations_percentage: f64 =
            100.0 * (remaining_combinations_count as f64) / (TOTAL_COMBINATIONS_COUNT as f64);

        let elapsed_time = SystemTime::now().duration_since(self.start_time);
        let elapsed_time_hours = match elapsed_time {
            Ok(time) => (time.as_secs_f64()) / 3600.0,
            Err(_) => 0.0,
        };

        // ratio of remaining work to completed work applied to elapsed time
        let hours_per_combination = elapsed_time_hours / (self.evaluated_count as f64);
        let estimated_time_remaining_hours =
            (hours_per_combination * TOTAL_COMBINATIONS_COUNT as f64) - elapsed_time_hours;

        println!("*****");
        println!("Wrote snapshot to disk.");
        println!(
            "Batch: Total {} | Pass {} | Fail {} | Pass Rate {:.2}",
            batch_evaluated_count, batch_pass_count, batch_fail_count, batch_pass_rate
        );
        println!(
            "Overall: Total {} | Pass {} | Fail {} | Pass Rate {:.2}",
            self.evaluated_count, self.pass_count, overall_fail_count, overall_pass_rate
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
        println!("Next combination is: {}.", next_combination);
        println!("*****");
    }
}

// no way around passing everything in, and no sense in a convenience struct
// given that it's a singular thread
#[allow(clippy::too_many_arguments)]
/// periodically take snapshots
fn take_snapshots(
    snapshot_frequency_secs: u64,
    execution_completed: Arc<Mutex<bool>>,
    stop_for_snapshot: &AtomicBool,
    workers_stopped: Arc<(Mutex<bool>, Condvar)>,
    generator_stopped: Arc<RwLock<bool>>,
    workers_snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
    generator_snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
    worker_pass_vectors: Vec<Arc<Mutex<Vec<PassMsg>>>>,
    num_running_workers: Arc<Mutex<usize>>,
    num_workers: usize,
    global_queue: Arc<Injector<LetterCombination>>,
    generator_next_combination: Arc<Mutex<LetterCombination>>,
    batch_count: Arc<Mutex<u64>>,
) {
    let start_time = SystemTime::now();

    // create snapshots directory with time-based unique identifier
    let cur_time = start_time
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let path_string = "snapshots-".to_string() + &cur_time.to_string();
    let snapshots_directory = Path::new(&path_string);
    match fs::create_dir(snapshots_directory) {
        Ok(_) => (),
        Err(_) => panic!("Could not create snapshots directory"),
    }

    const CHECK_FOR_COMPLETION_INTERVAL_SECS: u64 = 30;
    let sleep_loop_count = snapshot_frequency_secs / CHECK_FOR_COMPLETION_INTERVAL_SECS;
    // resize syntactically requires something to overwrite elements with in case of resizing up
    let default_pass_msg: PassMsg = (LetterCombination::from(ALL_A_FREQUENCIES), 0);
    let mut snapshot_number: u32 = 0;

    let mut progress_statistics = ProgressStatistics::new(start_time);
    let mut is_last_snapshot = false;

    loop {
        // take snapshots on the specified interval, waking periodically to check for overall completion
        for _ in 0..sleep_loop_count {
            if *execution_completed.lock().unwrap() {
                is_last_snapshot = true;
                break;
            }

            thread::sleep(time::Duration::from_secs(
                CHECK_FOR_COMPLETION_INTERVAL_SECS,
            ));
        }

        // stop the other threads for the snapshot
        // reset indicators that the snapshot was complete
        *workers_snapshot_complete.0.lock().unwrap() = false;
        *generator_snapshot_complete.0.lock().unwrap() = false;
        // trigger the snapshot
        println!("Taking snapshot.");
        stop_for_snapshot.store(true, MemoryOrdering::SeqCst);

        // wait for the other threads to have stopped
        // workers stop after the generator by constrution, so only need to check that
        // we need not (and should not) do this for the last snapshot
        if !is_last_snapshot {
            let (ref workers_stopped_lock, ref workers_stopped_condvar) = *workers_stopped;
            let mut snapshot_completed_check = workers_stopped_lock.lock().unwrap();
            while !*snapshot_completed_check {
                snapshot_completed_check = workers_stopped_condvar
                    .wait(snapshot_completed_check)
                    .unwrap();
            }
        }

        // check that a snapshot invariant (empty queues at completion) has been satisfied
        if global_queue.len() != 0 {
            panic!(
                "Global queue was not empty ({} combination(s)) at end of snapshot",
                global_queue.len()
            );
        }

        // write the snapshot to disk
        // aggregate worker vectors for a single write
        let mut passing_results: Vec<PassMsg> = Vec::new();
        for mutex in worker_pass_vectors.iter() {
            let worker_vec = mutex.lock().unwrap();
            passing_results.extend(worker_vec.iter());
        }

        let batch_pass_count = passing_results.len();

        // write the passing results to the disk
        let encoded_passing_results = bincode::serialize(&passing_results).unwrap();
        let snapshot_name = snapshot_number.to_string();
        let mut generator_snapshot_path = snapshots_directory.to_path_buf();
        generator_snapshot_path.push(snapshot_name);

        // intentionally panic if the filesystem operation fails
        fs::write(generator_snapshot_path, encoded_passing_results).unwrap();

        // clear out the (giant) temporary vector
        passing_results.resize(0, default_pass_msg);

        // write the next combination to disk (the snapshot consists of the results up to this
        // point and the next thing to be considered)
        // it's important that this occurs after the previous write so that we do not erroneously
        // skip vectors if we fail in between these steps (we would wind up with duplicates that we could
        // harmlessly deduplicate).
        let next_combination = *generator_next_combination.lock().unwrap();
        let encoded_next = bincode::serialize(&next_combination).unwrap();
        let snapshot_name = snapshot_number.to_string() + "NEXT";
        let mut generator_snapshot_path = snapshots_directory.to_path_buf();
        generator_snapshot_path.push(snapshot_name);

        // intentionally panic if the filesystem operation fails
        fs::write(generator_snapshot_path, encoded_next).unwrap();
        println!("Finished writing snapshot to disk.");

        // write statistics (we don't need hold the other threads for this)
        let batch_evaluated_count = *batch_count.lock().unwrap();
        progress_statistics.update_with_batch(
            batch_pass_count as u64,
            batch_evaluated_count,
            &next_combination,
            true,
        );

        if is_last_snapshot {
            return;
        }

        // reset for next snapshot
        stop_for_snapshot.store(false, MemoryOrdering::SeqCst);
        snapshot_number += 1;

        // re-start the generator thread first to limit context switch
        // thrashing while it builds up a buffer in the global queue
        // reset from this snapshot
        *generator_stopped.write().unwrap() = false;
        // signal generator to resume
        *generator_snapshot_complete.0.lock().unwrap() = true;
        generator_snapshot_complete.1.notify_all();
        thread::sleep(SPIN_UP_WAIT);

        // re-start the worker threads
        // reset from this snapshot
        *num_running_workers.lock().unwrap() = num_workers;
        *workers_stopped.0.lock().unwrap() = false;
        // signal workers to resume
        *workers_snapshot_complete.0.lock().unwrap() = true;
        workers_snapshot_complete.1.notify_all();
    }
}

// thread needs a lot of state passed in. not bundling into a struct as there is a singular generator thread
#[allow(clippy::too_many_arguments)]
/// push combinations into the global queue
fn generate_combinations(
    combinations: impl Iterator<Item = LetterCombination>,
    queue: Arc<Injector<LetterCombination>>,
    max_target_queue_size: usize,
    all_combinations_generated: Arc<RwLock<bool>>,
    stop_for_snapshot: &AtomicBool,
    generator_thread_stopped: Arc<RwLock<bool>>,
    snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
    next_combination: Arc<Mutex<LetterCombination>>,
    batch_count: Arc<Mutex<u64>>,
) {
    let mut local_batch_count: u64 = 0;

    for combination in combinations {
        // block the generator thread if the queue is ludicrously big.
        let mut stopped = false;
        while local_batch_count % (max_target_queue_size as u64) == 0
            && queue.len() >= max_target_queue_size
        {
            if !stopped {
                println!("Generator ran ahead, sleeping for now.");
                stopped = true;
            }

            // jump out of here to take the snapshot if need be (check at interval)
            if stop_for_snapshot.load(MemoryOrdering::SeqCst) {
                break;
            }
            thread::sleep(time::Duration::from_secs(5));
        }

        // can consider checking every x iteratons only if the atomic cas
        // winds up being too expensive (guessing that it can be parallelized / branch
        // predicted efficiently enough)
        if stop_for_snapshot.load(MemoryOrdering::SeqCst) {
            // write state that the snapshot thread needs
            *next_combination.lock().unwrap() = combination;
            *batch_count.lock().unwrap() = local_batch_count;

            // notify worker threads that we have stopped generating new combinations
            // and intentionally drop the guard asap to try to avoid blocking readers
            // n.b. if the generator thread were to be the last thread to stop,
            // (very unlikely given the amount of work that the worker threads have to do)
            // then the flag would prevent the last worker thread from stopping until
            // this thread has stopped, thus ensuring that the queues are emptied.
            {
                let mut generation_stopped = generator_thread_stopped.write().unwrap();
                *generation_stopped = true;
            }

            println!(
                "Generator stopped for snapshot with {} items remaining in global queue.",
                queue.len()
            );

            // block until the snapshot is complete
            let (ref lock, ref condvar) = *snapshot_complete;
            let mut snapshot_completed_check = lock.lock().unwrap();
            while !*snapshot_completed_check {
                snapshot_completed_check = condvar.wait(snapshot_completed_check).unwrap();
            }

            println!("Generator thread resuming after snapshot.");

            // reset batch
            local_batch_count = 0;
        }

        if stopped {
            println!("Generator thread resuming after running ahead.");
        }

        queue.push(combination);
        local_batch_count += 1;
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
    let pass_vector = worker_information.pass_vector;
    let all_combinations_generated = worker_information.all_combinations_generated;
    let stop_for_snapshot = worker_information.stop_for_snapshot;
    let generator_thread_stopped = worker_information.generator_thread_stopped;
    let workers_stopped = worker_information.workers_stopped;
    let num_running_workers = worker_information.num_running_workers;
    let snapshot_complete = worker_information.snapshot_complete;

    let mut passed_local = Vec::new();

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
                if score >= target {
                    passed_local.push((letters, score));
                }
            }
            None => {
                // completed
                if *all_combinations_generated.read().unwrap() {
                    return;
                }

                // check if we should dump for a snapshot
                if stop_for_snapshot.load(MemoryOrdering::SeqCst) {
                    let mut is_last_worker = false;
                    let mut can_stop_for_snapshot = false;
                    {
                        let mut running_worker_count = num_running_workers.lock().unwrap();
                        // this is the last worker thread stopping - verify that the generator has already stopped
                        if *running_worker_count == 1 {
                            is_last_worker = true;
                            let generator_stopped = generator_thread_stopped.read().unwrap();
                            if *generator_stopped {
                                can_stop_for_snapshot = true;
                            }
                            // don't allow stopping if we were to be the last worker thread to stop
                            // but the global thread has not stopped yet (busy wait)
                        }
                        // not the last worker thread stopping, so can freely stop and write snapshot
                        else {
                            *running_worker_count -= 1;
                            can_stop_for_snapshot = true;
                        }

                        // drop stopped_worker_count mutex
                    }

                    if can_stop_for_snapshot {
                        // move the local passed vector to the shared vector
                        *pass_vector.lock().unwrap() = passed_local;
                        passed_local = Vec::new();

                        if is_last_worker {
                            // notify the snapshot thread that all workers have stopped.
                            let mut workers_stopped_predicate = workers_stopped.0.lock().unwrap();
                            *workers_stopped_predicate = true;
                            let workers_stopped_cvar = &workers_stopped.1;
                            workers_stopped_cvar.notify_all();
                        }

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
                // workers are running ahead of the global thread - block for a while to limit context switch thrashing
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
// this is an upper bound on the score of any board that is a  permutation of these letters
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::letter::translate_word;
    use std::collections::HashSet;
    use trie_rs::TrieBuilder;

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
        assert_eq!(get_combination_score(&trie, all_a_lc), all_a_points);

        // should score AAA, AZA, AZC (can take multiple branches from a node) but not AZAZ (exhaust letters)
        const THREES_POINTS: u32 = 3 * POINTS[3];
        const THREES_FREQS: [u8; ALPHABET_LENGTH] = [
            3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        let threes_lc = LetterCombination::new(THREES_FREQS);
        assert_eq!(get_combination_score(&trie, threes_lc), THREES_POINTS);

        // should score AZA, AZC, and AZACB (use C at different positions, continue past non-scoring nodes)
        const AZ_POINTS: u32 = 2 * POINTS[3] + POINTS[5];
        const AZ_FREQS: [u8; ALPHABET_LENGTH] = [
            2, 1, 1, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ];
        let az_lc = LetterCombination::new(AZ_FREQS);
        assert_eq!(get_combination_score(&trie, az_lc), AZ_POINTS);
    }
}
