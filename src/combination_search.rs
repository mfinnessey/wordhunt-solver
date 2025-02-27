use crate::letter::Letter;
use crate::letter_combination::LetterCombination;
use crate::utilities::ALL_A_FREQUENCIES;
use combination_generator::SequentialLetterCombinationGenerator;
use generator::generate_combinations;
use snapshot::take_snapshots;
use std::{thread, time};
use trie_rs::Trie;
use worker::{evaluate_combinations, WorkerInformation};

use crossbeam::thread::ScopedJoinHandle;
use crossbeam_deque::{Injector, Stealer, Worker};
use crossbeam_utils::thread as crossbeam_thread;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex, RwLock};

pub mod bounding_functions;
mod combination_generator;
mod generator;
mod progress_statistics;
mod snapshot;
mod worker;

// TODO benchmark this if need be
/// the maximum batch size that a worker thread will pull from the global queue at once
const PULL_LIMIT: usize = 10_000;

/// time to wait between starting the generator thread and the worker threads to limit
/// thrashing from context-switches on an under-filled global queue.
const SPIN_UP_WAIT: time::Duration = time::Duration::from_millis(200);

/// data generated for a passing combination
type PassMsg = (LetterCombination, u32);

/// evaluates all combinations of letters using a given wordlist using a given metric
pub struct CombinationSearch<'a, CombinationGenerator: Iterator<Item = LetterCombination>> {
    word_list: &'a Trie<Letter>,
    combinations: CombinationGenerator,
    /// retain the score instead of simply a pass/fail bool so that we can eliminate combinations
    /// based off of lower bounds from stage 2
    metric: fn(&Trie<Letter>, LetterCombination) -> u32,
    target: u32,
    num_worker_threads: usize,
}

impl<'a, Generator: Iterator<Item = LetterCombination> + std::marker::Send>
    CombinationSearch<'a, Generator>
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
impl<'a> CombinationSearch<'a, SequentialLetterCombinationGenerator> {
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
