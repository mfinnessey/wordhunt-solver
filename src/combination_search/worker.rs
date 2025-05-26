use super::{PassMsg, PULL_LIMIT};
use crate::combination_search::EvaluationMetric;
use crate::letter_combination::LetterCombination;
use crate::utilities::{ALL_A_FREQUENCIES, ALPHABET_LENGTH, BATCH_SIZE};
use std::{iter, thread, time};

use crossbeam_deque::{Injector, Stealer, Worker};
use std::sync::atomic::{AtomicBool, Ordering as MemoryOrdering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

/// the shared information that a worker thread is provided with to process combinations
/// in conjunction with the overall program.
#[derive(Clone)]
pub struct SharedWorkerInformation<'a> {
    word_list_with_scores: &'a Vec<([u8; ALPHABET_LENGTH], u8)>,
    metric: EvaluationMetric,
    target: u32,
    /// shared queue accessors
    global: Arc<Injector<LetterCombination>>,
    stealers: Arc<Vec<Stealer<LetterCombination>>>,
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
    /// the number of workers that are active (i.e. haven't returned) - used to cover the pathological case where
    /// 1) a snapshot is triggered between generation completion and run completion
    /// 2) workers are split between termination and snapshotting
    /// 3) workers are not rescheduled to complete before the next snapshot
    /// 4) the snapshot thread blocks on a condvar that will never be signaled as the full complement of workers will never
    ///    decrement num_running_workers again for a worker to be the "last" worker.
    num_active_workers: Arc<Mutex<usize>>,
    /// set by the snapshot thread to notify the worker threads that the snapshot has been completed
    snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
    /// set if aborting early (e.g. for ctrl-c)
    aborting_early: Arc<RwLock<bool>>,
}
impl<'a> SharedWorkerInformation<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        word_list: &'a Vec<([u8; ALPHABET_LENGTH], u8)>,
        metric: EvaluationMetric,
        target: u32,
        global: Arc<Injector<LetterCombination>>,
        stealers: Arc<Vec<Stealer<LetterCombination>>>,
        all_combinations_generated: Arc<RwLock<bool>>,
        stop_for_snapshot: &'a AtomicBool,
        generator_thread_stopped: Arc<RwLock<bool>>,
        workers_stopped: Arc<(Mutex<bool>, Condvar)>,
        num_running_workers: Arc<Mutex<usize>>,
        num_active_workers: Arc<Mutex<usize>>,
        snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
        aborting_early: Arc<RwLock<bool>>,
    ) -> Self {
        Self {
            word_list_with_scores: word_list,
            metric,
            target,
            global,
            stealers,
            all_combinations_generated,
            stop_for_snapshot,
            generator_thread_stopped,
            workers_stopped,
            num_running_workers,
            num_active_workers,
            snapshot_complete,
            aborting_early,
        }
    }
}

pub fn evaluate_combinations(
    worker_information: SharedWorkerInformation,
    local: Worker<LetterCombination>,
    pass_vector: Arc<Mutex<Option<Vec<PassMsg>>>>,
) {
    // unpack convenience struct
    let word_list = worker_information.word_list_with_scores;
    let metric = worker_information.metric;
    let target = worker_information.target;
    let global = worker_information.global;
    let stealers = worker_information.stealers;
    let all_combinations_generated = worker_information.all_combinations_generated;
    let stop_for_snapshot = worker_information.stop_for_snapshot;
    let generator_thread_stopped = worker_information.generator_thread_stopped;
    let workers_stopped = worker_information.workers_stopped;
    let num_running_workers = worker_information.num_running_workers;
    let num_active_workers = worker_information.num_active_workers;
    let snapshot_complete = worker_information.snapshot_complete;
    let aborting_early = worker_information.aborting_early;

    let mut passed_local = Vec::new();
    // initialize to make rust happy
    let mut batch_local = [LetterCombination::new(ALL_A_FREQUENCIES); BATCH_SIZE];
    let mut batch_count = 0;

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
                // evaluate a batch at a time
                batch_local[batch_count] = letters;
                batch_count += 1;

                if batch_count == BATCH_SIZE {
                    batch_count = 0;
                    evaluate_batch(
                        metric,
                        target,
                        word_list,
                        &batch_local,
                        BATCH_SIZE,
                        &mut passed_local,
                    );
                }
            }
            None => {
                // nothing remaining period - we're done!
                if *all_combinations_generated.read().expect("generator or snapshot thread paniced while holding all_combinations_generated rwlock") {
                    // evaluate anything that's in the batch
                    evaluate_batch(
                        metric,
                        target,
                        word_list,
                        &batch_local,
                        batch_count,
                        &mut passed_local,
                    );

                    // move the local passed vector to the shared vector
                    *pass_vector.lock().unwrap_or_else(|_| panic!("snapshot thread paniced while holding pass_vector mutex for worker thread {}", thread::current().name().unwrap_or("unnamed"))) = Some(passed_local);

                    // it's possible for all_combinations_generated to be set after some workers have stopped
                    // for a snapshot but not all, so we need to be compatible with the normal start / stop
                    // mechanism
                    {
                        let mut running_worker_count = num_running_workers.lock().expect("snapshot or worker thread paniced while holding num_running_workers mutex");
                        // decrement active workers before running workers to ensure that any subsequent restart
                        // doesn't rely on this thread restarting (as the decrement to the running workers MUST occur
                        // before such a restart attempt by construction, but we could be the second to last thread to
                        // stop here with the last thread stopping in the "normal" snapshot case)
                        *num_active_workers.lock().expect("snapshot or worker thread paniced while holding num_active_workers mutex") -= 1;
                        *running_worker_count -= 1;
			// the last worker to stop notifies the snapshot thread
                        if *running_worker_count == 0 {
                            notify_snapshot_thread_all_workers_stopped(&workers_stopped);
                        }

                        println!(
                            "worker thread {} stopped with {} running {} active",
                            thread::current().name().unwrap_or("unnamed"),
                            *running_worker_count,
                            *num_active_workers.lock().expect("snapshot or worker thread paniced while holding num_active_workers mutex")
                        );
                    }

                    // we need not wait to be restarted - we're done!

                    // check invariant (local queue empty at thread termination)
                    if !local.is_empty() {
                        panic!("local queue on worker thread {} was not empty ({} combination(s)) at termination.",
			       thread::current().name().unwrap_or("unnamed"), local.len())
                    }

                    println!(
                        "worker thread {} completed execution",
                        thread::current().name().unwrap_or("unnamed")
                    );

                    return;
                }

                // there's stuff remaining, but the queues are dry - check if we should dump for a snapshot
                if stop_for_snapshot.load(MemoryOrdering::SeqCst) {
                    // workers can only stop for a snaphot after the generator is stopped
                    if *generator_thread_stopped.read().expect("generator or snapshot thread paniced while holding generator_stopped mutex") {
                        // process anything that's in the current batch
                        evaluate_batch(
                            metric,
                            target,
                            word_list,
                            &batch_local,
                            batch_count,
                            &mut passed_local,
                        );
                        batch_count = 0;

                        // move the local passed vector to the shared vector
                        let passed_count = passed_local.len();
                        *pass_vector.lock().unwrap_or_else(|_| panic!("snapshot thread paniced while holding pass_vector mutex for worker thread {}", thread::current().name().unwrap_or("unnamed"))) = Some(passed_local);
                        passed_local = Vec::new();

                        // check snapshot invariant (queues empty)
                        if !local.is_empty() {
                            panic!("local queue on worker thread {} was not empty ({} combination(s)) when stopping for snapshot.",
			    thread::current().name().unwrap_or("unnamed"), local.len())
                        }

                        {
                            let mut running_worker_count = num_running_workers.lock().expect("generator or worker thread paniced while holding num_running_workers mutex");
                            *running_worker_count -= 1;
                            // the last worker to stop notifies the snapshot thread
                            if *running_worker_count == 0 {
                                notify_snapshot_thread_all_workers_stopped(&workers_stopped);
                                println!(
                                    "worker thread {} stopped for snapshot (last thread) with {} passed",
                                    thread::current().name().unwrap_or("unnamed"),
				    passed_count
                                );
                            } else {
                                println!(
                                    "worker thread {} stopped for snapshot with {} passed",
                                    thread::current().name().unwrap_or("unnamed"),
                                    passed_count
                                );
                            }
                        }

                        // block until the global thread has completed the snapshot
                        let mut snapshot_complete_predicate = snapshot_complete.0.lock().expect("snapshot or worker thread paniced while holding worker snapshot_complete mutex");
                        let snapshot_complete_condvar = &snapshot_complete.1;
                        while !*snapshot_complete_predicate {
                            snapshot_complete_predicate = snapshot_complete_condvar
                                .wait(snapshot_complete_predicate)
                                .expect("snapshot or worker thread paniced while holding worker snapshot_complete mutex");
                        }

                        if *aborting_early.read().expect("generator or snapshot thread paniced while holding aborting_early mutex") {
                            println!(
                                "worker thread {} aborting early",
                                thread::current().name().unwrap_or("unnamed")
                            );
                            return;
                        }

                        *num_running_workers.lock().expect("snapshot or worker thread paniced while holding num_running_workers mutex") += 1;
                        println!(
                            "worker thread {} restarted after snapshot",
                            thread::current().name().unwrap_or("unnamed")
                        );
                    }
                }
                // queues are dry, but no snapshot has been triggered - workers are running ahead of the generator
                else {
                    println!("all queues are dry but not all letter combinations are exhausted - sleeping.");
                    thread::sleep(time::Duration::from_secs(1));
                }
            }
        }
    }
}

fn notify_snapshot_thread_all_workers_stopped(workers_stopped: &Arc<(Mutex<bool>, Condvar)>) {
    // notify the snapshot thread that all workers have stopped.
    let mut workers_stopped_predicate = workers_stopped
        .0
        .lock()
        .expect("snapshot thread paniced while holding workers_stopped mutex");
    *workers_stopped_predicate = true;
    let workers_stopped_cvar = &workers_stopped.1;
    workers_stopped_cvar.notify_all();
}

fn evaluate_batch(
    metric: EvaluationMetric,
    target: u32,
    words: &[([u8; ALPHABET_LENGTH], u8)],
    letter_combinations: &[LetterCombination; BATCH_SIZE],
    batch_count: usize,
    passed_local: &mut Vec<PassMsg>,
) {
    let scores = metric(words, letter_combinations);
    // trim off any bogus letter combinations (from a partially full batch)
    // from consideration for movement to the pass vector
    for (letter_combination, score) in letter_combinations[..batch_count]
        .iter()
        .zip(scores[..batch_count].iter())
    {
        if *score >= target {
            passed_local.push((*letter_combination, *score));
        }
    }
}
