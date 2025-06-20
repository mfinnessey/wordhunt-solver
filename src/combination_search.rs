use crate::combination_search::progress_information::ProgressInformation;
use crate::letter_combination::LetterCombination;
use crate::utilities::{ALL_A_FREQUENCIES, ALPHABET_LENGTH, BATCH_SIZE};

use generator::generate_combinations;
use snapshot::take_snapshots;
use std::path::PathBuf;
use std::time::SystemTime;
use std::{thread, time};
use worker::{evaluate_combinations, SharedWorkerInformation};

use crossbeam::thread::ScopedJoinHandle;
use crossbeam_deque::{Injector, Stealer, Worker};
use crossbeam_utils::thread as crossbeam_thread;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex, RwLock};

pub mod bounding_functions;
pub mod combination_generator;
mod generator;
pub mod progress_information;
mod snapshot;
mod worker;

/// the maximum batch size that a worker thread will pull from the global queue at once
/// this isn't scientific, but seems reasonable enough
const PULL_LIMIT: usize = 10_000;

/// time to wait between starting the generator thread and the worker threads to limit
/// thrashing from context-switches on an under-filled global queue.
const SPIN_UP_WAIT: time::Duration = time::Duration::from_millis(200);

/// data generated for a passing combination
pub type PassMsg = (LetterCombination, u32);
/// metric to evaluate a slice of LetterCombinations against a wordlist
/// expressed as a slice of words expressed as their letter frequencies and their score
pub type EvaluationMetric =
    fn(&[([u8; ALPHABET_LENGTH], u8)], &[LetterCombination; BATCH_SIZE]) -> [u32; BATCH_SIZE];

/// evaluates all combinations of letters using a given wordlist using a given metric
pub struct CombinationSearch<'a> {
    word_list: &'a Vec<([u8; ALPHABET_LENGTH], u8)>,
    /// retain the score instead of simply a pass/fail bool so that we can eliminate combinations
    /// based off of lower bounds from stage 2
    metric: EvaluationMetric,
    target: u32,
    num_worker_threads: usize,
    /// set by some outside mechanism (generally a signal handler) to cleanly terminate the
    /// combination search before its natural completion
    terminator: Arc<Mutex<bool>>,
}

impl CombinationSearch<'_> {
    /// spawn threads to create combination evaluation architecture
    pub fn evaluate_combinations<Generator, GeneratorCreatorClosure>(
        self,
        snapshot_frequency: u64,
        incoming_progress_information: Option<ProgressInformation>,
        combinations_generator_creator: GeneratorCreatorClosure,
    ) -> PathBuf
    where
        Generator: Iterator<Item = LetterCombination> + std::marker::Send,
        GeneratorCreatorClosure: Fn(LetterCombination) -> Generator,
    {
        let progress_information = match incoming_progress_information {
            Some(information) => information,
            // need to create the progress information (automatically creates the snapshots directory)
            None => ProgressInformation::new(
                SystemTime::now(),
                LetterCombination::new(ALL_A_FREQUENCIES),
            ),
        };

        println!(
            "beginning evaluation from combination {}",
            progress_information
                .get_next_combination()
                .expect("PROGRESS INFORMATION CONTAINS NO NEXT COMBINATION")
        );

        let initial_letter_combination = progress_information.get_next_combination();
        let snapshots_directory = progress_information.get_snapshots_directory().to_owned();

        let global_queue = Injector::new();

        let mut workers: Vec<Worker<LetterCombination>> = Vec::new();
        let mut stealers: Vec<Stealer<LetterCombination>> = Vec::new();
        let mut stopped_mutexes: Vec<Mutex<()>> = Vec::new();
        let mut pass_vectors: Vec<Arc<Mutex<Option<Vec<PassMsg>>>>> = Vec::new();

        for _ in 0..self.num_worker_threads {
            let w = Worker::new_fifo();
            let stealer = w.stealer();
            let pass_vector = Arc::new(Mutex::new(None));

            workers.push(w);
            stealers.push(stealer);
            stopped_mutexes.push(Mutex::new(()));
            pass_vectors.push(pass_vector);
        }

        let next_combination =
            Arc::new(Mutex::new(Some(LetterCombination::new(ALL_A_FREQUENCIES))));
        let batch_count = Arc::new(Mutex::new(0));

        let all_combinations_generated = Arc::new(RwLock::new(false));
        let stop_for_snapshot = AtomicBool::new(false);
        let generator_thread_stopped = Arc::new(RwLock::new(false));
        let aborting_early = Arc::new(RwLock::new(false));

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

        let stealers_vec = Arc::new(stealers);
        let global_queue = Arc::new(global_queue);

        let shared_worker_information = SharedWorkerInformation::new(
            self.word_list,
            self.metric,
            self.target,
            global_queue.clone(),
            stealers_vec,
            all_combinations_generated.clone(),
            &stop_for_snapshot,
            generator_thread_stopped.clone(),
            workers_stopped.clone(),
            num_running_workers.clone(),
            num_active_workers.clone(),
            workers_snapshot_complete.clone(),
            aborting_early.clone(),
        );

        crossbeam_thread::scope(|s| {
            // spawn the generator thread
            let generator = combinations_generator_creator(
                initial_letter_combination.expect("no initial letter combination"),
            );
            let global_queue_for_generator = global_queue.clone();
            let max_target_queue_size =
                <usize as TryInto<u64>>::try_into(2 * self.num_worker_threads * PULL_LIMIT)
                    .expect("max_target_queue_size will not fit in a usize");

            let all_combinations_generated = &all_combinations_generated;
            let stop_for_snapshot = &stop_for_snapshot;
            let generator_thread_stopped = &generator_thread_stopped;
            let generator_snapshot_complete = &generator_snapshot_complete;
            let aborting_early = &aborting_early;
            let next_combination = &next_combination;
            let batch_count = &batch_count;
            let generator_handle = s.spawn(move |_| {
                generate_combinations(
                    generator,
                    global_queue_for_generator,
                    max_target_queue_size,
                    all_combinations_generated.clone(),
                    stop_for_snapshot,
                    generator_thread_stopped.clone(),
                    generator_snapshot_complete.clone(),
                    aborting_early.clone(),
                    next_combination.clone(),
                    batch_count.clone(),
                )
            });

            thread::sleep(SPIN_UP_WAIT);

            // spawn worker threads
            let mut worker_threads: Vec<ScopedJoinHandle<'_, ()>> = Vec::new();
            println!("spawning {} worker threads.", self.num_worker_threads);
            for (thread_num, pass_vector) in pass_vectors
                .iter()
                .enumerate()
                .take(self.num_worker_threads)
            {
                let thread_worker = workers.pop().expect("missing element in workers vector");
                let shared_worker_information = shared_worker_information.clone();
                let pass_vector = pass_vector.clone();
                let handle = s
                    .builder()
                    .name(thread_num.to_string())
                    .spawn(move |_| {
                        evaluate_combinations(shared_worker_information, thread_worker, pass_vector)
                    })
                    .expect("failed to spawn worker thread");
                worker_threads.push(handle);
            }

            // spawn snapshot thread
            let snapshot_handle = s.spawn(move |_| {
                take_snapshots(
                    progress_information,
                    snapshot_frequency,
                    stop_for_snapshot,
                    workers_stopped.clone(),
                    generator_thread_stopped.clone(),
                    workers_snapshot_complete.clone(),
                    generator_snapshot_complete.clone(),
                    all_combinations_generated.clone(),
                    pass_vectors,
                    num_running_workers.clone(),
                    num_active_workers.clone(),
                    global_queue,
                    next_combination.clone(),
                    batch_count.clone(),
                    self.terminator,
                    aborting_early.clone(),
                )
            });

            // join threads
            if generator_handle.join().is_err() {
                panic!("generator thread paniced!")
            }

            for worker_thread_handle in worker_threads {
                if worker_thread_handle.join().is_err() {
                    panic!("worker thread paniced!")
                }
            }

            if snapshot_handle.join().is_err() {
                panic!("snapshot thread paniced!");
            }
        })
        .expect("combination search failed to spawn or terminate threads cleanly");

        snapshots_directory
    }
}

/// implementation for raw stage 1 search
impl<'a> CombinationSearch<'a> {
    pub fn new(
        word_list: &'a Vec<([u8; ALPHABET_LENGTH], u8)>,
        metric: EvaluationMetric,
        target: u32,
        num_worker_threads: usize,
        terminator: Arc<Mutex<bool>>,
    ) -> Self {
        Self {
            word_list,
            metric,
            target,
            num_worker_threads,
            terminator,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::utilities::snapshot_utilities::{
        aggregate_snapshots_from_directory, read_next_progress_information_from_directory,
    };
    use crate::utilities::test_utilities::TestCleanup;
    use crate::utilities::{ALPHABET_LENGTH, TILE_COUNT};
    use rand::random_range;
    use std::cmp::max;
    use std::collections::HashMap;
    use std::hint::black_box;

    use super::*;

    #[test]
    #[ignore = "a very intensive \"unit\" test that is really a stage 1 search integration test with a fake generator"]
    fn test_combination_search() {
        // a generic large test to (mostly) behave as in the real run
        test_fake_combination_search(
            30_000_000,
            std::thread::available_parallelism().unwrap().get(),
            30,
            8,
        );
    }

    #[test]
    #[ignore = "a long integration test of successfully resuming from a snapshot"]
    fn test_resume_from_snapshot() {
        // test strategy is to interrupt a run and then
        // resume another run from that snapshot and compare it against
        // the expected result.
        // this creates a multithreaded test case within the test infra (lovely!)
        // this test strategy is not deterministic, but what is with multithreading...

        let num_workers = std::thread::available_parallelism().unwrap().get();
        // n.b. this is going to be especially machine-dependent / non-deterministic, but isn't (mostly) everything
        // in terms of multithreading?
        const COMBINATION_COUNT: u64 = 30_000_000;
        const TARGET_A_COUNT: u32 = 10;

        let dummy_vec = Vec::new();
        let unused_fake_combination_generator = FakeLetterCombinationGenerator::new(
            LetterCombination::new(ALL_A_FREQUENCIES),
            COMBINATION_COUNT,
        );
        println!(
            "running test_combination_search with {} workers",
            num_workers
        );

        let terminator = Arc::new(Mutex::new(false));
        let fcs: CombinationSearch = CombinationSearch::new(
            &dummy_vec,
            count_letter_a_batch,
            TARGET_A_COUNT,
            num_workers,
            terminator.clone(),
        );

        // spawn a thread in the background to set terminator to true before any snapshots have been taken
        {
            let terminator = terminator.clone();
            thread::spawn(move || set_terminator_in_secs(terminator, &45));
        }

        let fake_combination_generator_closure =
            |lc: LetterCombination| FakeLetterCombinationGenerator::new(lc, COMBINATION_COUNT);
        let snapshot_path_1 =
            fcs.evaluate_combinations(600, None, fake_combination_generator_closure);
        let _test_cleanup = TestCleanup::new(&snapshot_path_1);

        let (max_snapshot_num_1, progress_info_1) =
            read_next_progress_information_from_directory(&snapshot_path_1).unwrap();
        println!(
            "portion 1 - read the last snapshot (num {}) from snapshot directory",
            max_snapshot_num_1
        );
        println!("evaluated_count {}", *progress_info_1.get_evaluated_count());
        let remaining_combination_count_1 =
            COMBINATION_COUNT - *progress_info_1.get_evaluated_count();
        assert!(
            remaining_combination_count_1 != 0,
            "all combinations were evaluated in the first search"
        );
        // verify that the termination actually happened (as opposed to some other weird edge case that should've paniced)
        assert!(
            !*terminator.lock().unwrap(),
            "termination was unsuccessful for search 1"
        );

        // second part of search
        let fcs2 = CombinationSearch::new(
            &dummy_vec,
            count_letter_a_batch,
            TARGET_A_COUNT,
            num_workers,
            terminator.clone(),
        );

        // pass in the existing progress here
        let fake_combination_generator_2 = |lc: LetterCombination| {
            FakeLetterCombinationGenerator::new(lc, remaining_combination_count_1)
        };

        // spawn a thread in the background to set terminator to true after a snapshot has been taken normally
        {
            let terminator = terminator.clone();
            thread::spawn(move || set_terminator_in_secs(terminator, &80));
        }

        let snapshot_path_2 = fcs2.evaluate_combinations(
            60,
            Some(progress_info_1.clone()),
            fake_combination_generator_2,
        );

        // resuming from a snapshot should write to the same directory
        assert_eq!(snapshot_path_1, snapshot_path_2);
        let (max_snapshot_num_2, progress_info_2) =
            read_next_progress_information_from_directory(&snapshot_path_2).unwrap();
        println!(
            "portion 2 - read the last snapshot (num {}) from snapshot directory",
            max_snapshot_num_2
        );
        println!("evaluated_count {}", *progress_info_2.get_evaluated_count());
        let remaining_combination_count_2 =
            COMBINATION_COUNT - *progress_info_2.get_evaluated_count();
        assert!(
            remaining_combination_count_2 != 0,
            "all combinations were evaluated in the second search"
        );
        // verify that the termination actually happened (as opposed to some other weird edge case that should've paniced)
        assert!(!*terminator.lock().unwrap(), "termination was unsuccessful");

        // final part of search
        let fcs3 = CombinationSearch::new(
            &dummy_vec,
            count_letter_a_batch,
            TARGET_A_COUNT,
            num_workers,
            terminator.clone(),
        );

        // pass in the existing progress here
        let fake_combination_generator_3 = |lc: LetterCombination| {
            FakeLetterCombinationGenerator::new(lc, remaining_combination_count_2)
        };
        let snapshot_path_3 =
            fcs3.evaluate_combinations(30, Some(progress_info_2), fake_combination_generator_3);

        // resuming from a snapshot should write to the same directory
        assert_eq!(snapshot_path_2, snapshot_path_3);
        // we should have generated all combinations here
        let (_, progress_info_3) =
            read_next_progress_information_from_directory(&snapshot_path_3).unwrap();
        assert_eq!(progress_info_3.get_evaluated_count(), &COMBINATION_COUNT);

        // verify that the expected and actual contents are as expected
        let expected: HashMap<PassMsg, usize> = unused_fake_combination_generator
            .filter_map(|x| {
                let actual_count = count_letter_a(&x);
                if actual_count >= TARGET_A_COUNT {
                    Some((x, actual_count))
                } else {
                    None
                }
            })
            .fold(HashMap::new(), |mut map, val| {
                map.entry(val).and_modify(|frq| *frq += 1).or_insert(1);
                map
            });

        let actual: HashMap<PassMsg, usize> = aggregate_snapshots_from_directory(snapshot_path_2)
            .unwrap()
            .iter()
            .fold(HashMap::new(), |mut map, val| {
                map.entry(*val).and_modify(|frq| *frq += 1).or_insert(1);
                map
            });

        assert_eq!(actual, expected);
    }

    /// set the terminator after (no less than) the specified amount of time
    fn set_terminator_in_secs(terminator: Arc<Mutex<bool>>, secs: &u64) {
        thread::sleep(time::Duration::from_secs(*secs));
        *terminator.lock().unwrap() = true;
    }

    /// test combination search with the specified parameters using a fake combination generator that counts
    /// the frequency of the letter 'A'
    fn test_fake_combination_search(
        combination_count: u64,
        num_workers: usize,
        snapshot_frequency_secs: u64,
        target_a_count: u32,
    ) {
        let dummy_vec = Vec::new();
        let fake_combination_generator_closure =
            |lc: LetterCombination| FakeLetterCombinationGenerator::new(lc, combination_count);
        let fake_combination_generator_unused = FakeLetterCombinationGenerator::new(
            LetterCombination::new(ALL_A_FREQUENCIES),
            combination_count,
        );
        println!(
            "running test_combination_search with {} workers",
            num_workers
        );

        let fcs: CombinationSearch = CombinationSearch::new(
            &dummy_vec,
            count_letter_a_batch,
            target_a_count,
            num_workers,
            Arc::new(Mutex::new(false)),
        );

        let snapshot_path = fcs.evaluate_combinations(
            snapshot_frequency_secs,
            None,
            fake_combination_generator_closure,
        );
        let _test_cleanup = TestCleanup::new(snapshot_path.clone());

        let expected: HashMap<PassMsg, usize> = fake_combination_generator_unused
            .filter_map(|x| {
                let actual_count = count_letter_a(&x);
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

    /// count the number of a's in the combinations as an arbitrary, verifiable metric
    fn count_letter_a_batch(
        _word_list: &[([u8; ALPHABET_LENGTH], u8)],
        combinations: &[LetterCombination; BATCH_SIZE],
    ) -> [u32; BATCH_SIZE] {
        // use the last u8's  high for the range to make
        // the signature still match. a fun lil testning hack!

        for combination in combinations {
            let high: u32 = (max(1, combination[25]) as u32) * 1_000;
            // simulate doing some real work
            for _ in 1..random_range(100..high) {
                black_box(())
            }
        }

        combinations
            .iter()
            .map(|combination| combination[0] as u32)
            .collect::<Vec<u32>>()
            .try_into()
            .unwrap()
    }

    /// quick version for verification
    fn count_letter_a(combination: &LetterCombination) -> u32 {
        combination[0].into()
    }

    /// fake combination generation for testing with bounded size
    /// stores all combinations with at least as many specified a's.
    #[derive(Clone)]
    struct FakeLetterCombinationGenerator {
        a_count: u8,
        combination_count: u64,
        target_combination_count: u64,
    }

    impl FakeLetterCombinationGenerator {
        fn new(
            initial_letter_combination: LetterCombination,
            target_combination_count: u64,
        ) -> Self {
            let initial_a_count = initial_letter_combination[0];
            Self {
                a_count: initial_a_count,
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
}
