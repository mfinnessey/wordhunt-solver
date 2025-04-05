use super::{PassMsg, PULL_LIMIT};
use crate::letter::Letter;
use crate::letter_combination::LetterCombination;
use std::{iter, thread, time};
use trie_rs::Trie;

use crossbeam_deque::{Injector, Stealer, Worker};
use std::sync::atomic::{AtomicBool, Ordering as MemoryOrdering};
use std::sync::{Arc, Condvar, Mutex, RwLock};

/// the information that a worker thread is provided with to process combinations
/// in conjunction with the overall program.
pub struct WorkerInformation<'a> {
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
    /// the number of workers that are active (i.e. haven't returned) - used to cover the pathological case where
    /// 1) a snapshot is triggered between generation completion and run completion 2) workers are split between
    /// termination and snapshotting 3) workers are not rescheduled to complete before the next snapshot 4) the
    /// snapshot thread blocks on a condvar that will never be signaled as the full complement of workers will never
    /// decrement num_running_workers again for a worker to be the "last" worker.
    num_active_workers: Arc<Mutex<usize>>,
    /// set by the snapshot thread to notify the worker threads that the snapshot has been completed
    snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
}

impl<'a> WorkerInformation<'a> {
    // this object's constructor is designed to simplify the signature of evaluate_combinations
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
        num_active_workers: Arc<Mutex<usize>>,
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
            num_active_workers,
            snapshot_complete,
        }
    }
}

pub fn evaluate_combinations(worker_information: WorkerInformation) {
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
    let num_active_workers = worker_information.num_active_workers;
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
                // nothing remaining period - we're done!
                if *all_combinations_generated.read().unwrap() {
                    // move the local passed vector to the shared vector
                    *pass_vector.lock().unwrap() = passed_local;

                    // it's possible for all_combinations_generated to be set after some workers have stopped
                    // for a snapshot but not all, so we need to be compatible with the normal start / stop
                    // mechanism
                    {
                        let mut running_worker_count = num_running_workers.lock().unwrap();
                        // decrement active workers before running workers to ensure that any subsequent restart
                        // doesn't rely on this thread restarting (as the decrement to the running workers MUST occur
                        // before such a restart attempt by construction, but we could be the second to last thread to
			// stop here with the last thread stopping in the "normal" snapshot case)
                        *num_active_workers.lock().unwrap() -= 1;
                        *running_worker_count -= 1;
                        if *running_worker_count == 1 {
                            notify_snapshot_thread_all_workers_stopped(&workers_stopped);
                        }
                    }

                    // we need not wait to be restarted - we're done!

                    // check invariant (local queue empty at thread termination)
                    if !local.is_empty() {
                        panic!("Local queue on thread {} was not empty ({} combination(s)) at end of snapshot.",
			       thread::current().name().unwrap_or("unnamed"), local.len())
                    }

                    println!(
                        "thread {} completed execution",
                        thread::current().name().unwrap_or("unnamed")
                    );

                    return;
                }

                // there's stuff remaining, but the queues are dry - check if we should dump for a snapshot
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
                            can_stop_for_snapshot = true;
                        }

                        // drop stopped_worker_count mutex
                    }

                    if can_stop_for_snapshot {
                        // move the local passed vector to the shared vector
                        *pass_vector.lock().unwrap() = passed_local;
                        passed_local = Vec::new();

                        // check snapshot invariant (queues empty)
                        if !local.is_empty() {
                            panic!("Local queue on thread {} was not empty ({} combination(s)) at end of snapshot.",
			    thread::current().name().unwrap_or("unnamed"), local.len())
                        }

                        *num_running_workers.lock().unwrap() -= 1;
                        println!(
                            "thread {} stopped for snapshot",
                            thread::current().name().unwrap_or("unnamed")
                        );

                        if is_last_worker {
                            notify_snapshot_thread_all_workers_stopped(&workers_stopped);
                        }

                        // block until the global thread has completed the snapshot
                        let mut snapshot_complete_predicate = snapshot_complete.0.lock().unwrap();
                        let snapshot_complete_condvar = &snapshot_complete.1;
                        while !*snapshot_complete_predicate {
                            snapshot_complete_predicate = snapshot_complete_condvar
                                .wait(snapshot_complete_predicate)
                                .unwrap();
                        }

                        *num_running_workers.lock().unwrap() += 1;
                        println!(
                            "thread {} restarted after snapshot",
                            thread::current().name().unwrap_or("unnamed")
                        );
                    }
                }
                // queues are dry, but no snapshot has been triggered - workers are running ahead of the generator
                else {
                    println!("All queues are dry but not all letter combinations are exhausted. Sleeping.");
                    thread::sleep(time::Duration::from_secs(1));
                }
            }
        }
    }
}

fn notify_snapshot_thread_all_workers_stopped(workers_stopped: &Arc<(Mutex<bool>, Condvar)>) {
    // notify the snapshot thread that all workers have stopped.
    let mut workers_stopped_predicate = workers_stopped.0.lock().unwrap();
    *workers_stopped_predicate = true;
    let workers_stopped_cvar = &workers_stopped.1;
    workers_stopped_cvar.notify_all();
}
