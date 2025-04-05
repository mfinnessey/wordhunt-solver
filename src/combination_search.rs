use crate::letter::Letter;
use crate::letter_combination::LetterCombination;
use crate::utilities::ALL_A_FREQUENCIES;
use combination_generator::SequentialLetterCombinationGenerator;
use generator::generate_combinations;
use snapshot::take_snapshots;
use std::path::PathBuf;
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
    pub fn evaluate_combinations(self, snapshot_frequency: u64) -> PathBuf {
        let mut snapshot_path: PathBuf = "INVALID".into();

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
        let num_active_workers: Arc<Mutex<usize>> = Arc::new(Mutex::new(self.num_worker_threads));
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

        crossbeam_thread::scope(|s| {
            // spawn the generator thread
            let max_target_queue_size = 2 * self.num_worker_threads * PULL_LIMIT;
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
                    max_target_queue_size,
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
                    Arc::clone(&num_active_workers),
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
            let snapshot_handle = s.spawn(move |_| {
                take_snapshots(
                    snapshot_frequency,
                    stop_for_snapshot,
                    Arc::clone(workers_stopped),
                    Arc::clone(generator_thread_stopped),
                    Arc::clone(&workers_snapshot_complete),
                    Arc::clone(generator_snapshot_complete),
                    Arc::clone(all_combinations_generated),
                    pass_vectors,
                    Arc::clone(&num_running_workers),
                    Arc::clone(&num_active_workers),
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

            match snapshot_handle.join() {
                Ok(path) => snapshot_path = path,
                Err(_e) => panic!("snapshot thread paniced!"),
            }
        })
        .unwrap(); // unwrap as we would just panic anyways

        snapshot_path
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

#[cfg(test)]
mod tests {
    use crate::utilities::test_utilities::TestCleanup;
    use crate::utilities::{ALPHABET_LENGTH, TILE_COUNT};
    use rand::random_range;
    use snapshot::aggregate_snapshots_from_directory;
    use std::cmp::min;
    use std::collections::HashMap;
    use std::hint::black_box;

    use super::*;

    #[test]
    #[ignore = "a very intensive \"unit\" test that is really a stage 1 search integration test with a fake generator"]
    fn test_combination_search() {
        // very rapid snapshots to stress-test the infrastucture
        test_fake_combination_search(
            3_000_000,
            std::thread::available_parallelism().unwrap().get(),
            1,
            8,
        );

        // a generic large test to (mostly) behave as in the real run
        test_fake_combination_search(
            6_000_000,
            std::thread::available_parallelism().unwrap().get(),
            60,
            8,
        );
    }

    /// test combination search with the specified parameters using a fake combination generator that counts
    /// the frequency of the letter 'A'
    fn test_fake_combination_search(
        combination_count: u32,
        num_workers: usize,
        snapshot_frequency_secs: u64,
        target_a_count: u32,
    ) {
        let dummy_trie = trie_rs::TrieBuilder::new().build();
        let dummy_lc = LetterCombination::new(ALL_A_FREQUENCIES);
        let fake_combination_generator = FakeLetterCombinationGenerator::new(combination_count);
        println!(
            "running test_combination_search with {} workers",
            num_workers
        );

        let fcs: CombinationSearch<FakeLetterCombinationGenerator> = CombinationSearch::fake_new(
            &dummy_trie,
            dummy_lc,
            count_letter_a,
            target_a_count,
            num_workers,
            combination_count,
        );

        let snapshot_path = fcs.evaluate_combinations(snapshot_frequency_secs);
        let _test_cleanup = TestCleanup::new(snapshot_path.clone());

        let expected: HashMap<PassMsg, usize> = fake_combination_generator
            .filter_map(|x| {
                let actual_count = count_letter_a(&dummy_trie, x);
                if actual_count >= target_a_count {
                    Some((x, actual_count))
                } else {
                    None
                }
            })
            .fold(HashMap::new(), |mut map, val| {
                map.entry(val).and_modify(|frq| *frq += 1).or_insert(1);
                map
            });

        let actual: HashMap<PassMsg, usize> = aggregate_snapshots_from_directory(snapshot_path)
            .unwrap()
            .iter()
            .fold(HashMap::new(), |mut map, val| {
                map.entry(*val).and_modify(|frq| *frq += 1).or_insert(1);
                map
            });

        assert_eq!(actual, expected);
    }

    /// count the number of a's in the combination as an arbitrary, verifiable metric
    fn count_letter_a(_word_list: &Trie<Letter>, combination: LetterCombination) -> u32 {
        // use the last u8's  high for the range to make
        // the signature still match. a fun lil testing hack!
        let high: u32 = (min(1, combination[25]) as u32) * 1_000;

        // simulate doing some real work
        for _ in 1..random_range(100..high) {
            black_box(())
        }
        combination[0].into()
    }

    /// fake combination generation for testing with bounded size
    /// stores all combinations with at least as many specified a's.
    struct FakeLetterCombinationGenerator {
        a_count: u8,
        combination_count: u32,
        target_combination_count: u32,
    }

    impl FakeLetterCombinationGenerator {
        fn new(target_combination_count: u32) -> Self {
            Self {
                a_count: 0,
                combination_count: 0,
                target_combination_count,
            }
        }
    }

    impl Iterator for FakeLetterCombinationGenerator {
        type Item = LetterCombination;
        fn next(&mut self) -> Option<Self::Item> {
            if self.combination_count >= self.target_combination_count {
                return None;
            }

            let non_a_count: u8 = (TILE_COUNT as u8) - self.a_count;
            let freqs: [u8; ALPHABET_LENGTH] = [
                self.a_count,
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
                non_a_count,
            ];
            let lc = LetterCombination::new(freqs);

            // increment
            self.a_count = (self.a_count + 1) % (TILE_COUNT as u8);
            self.combination_count += 1;

            Some(lc)
        }
    }

    /// implementation for raw stage 1 search
    impl<'a> CombinationSearch<'a, FakeLetterCombinationGenerator> {
        // it's a fake - don't need to use everything
        #[allow(unused_variables)]
        pub fn fake_new(
            word_list: &'a Trie<Letter>,
            starting_frequencies: LetterCombination,
            metric: fn(&Trie<Letter>, LetterCombination) -> u32,
            target: u32,
            num_worker_threads: usize,
            target_combination_count: u32,
        ) -> Self {
            Self {
                word_list,
                combinations: FakeLetterCombinationGenerator::new(target_combination_count),
                metric,
                target,
                num_worker_threads,
            }
        }
    }
}
