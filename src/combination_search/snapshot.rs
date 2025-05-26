use super::progress_information::ProgressInformation;
use super::{PassMsg, SPIN_UP_WAIT};
use crate::letter_combination::LetterCombination;
use crossbeam_deque::Injector;
use std::cmp::max;
use std::sync::atomic::{AtomicBool, Ordering as MemoryOrdering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::{fs, thread, time};

// no way around passing everything in, and no sense in a convenience struct
// given that it's a singular thread
#[allow(clippy::too_many_arguments)]
/// periodically take snapshots
pub fn take_snapshots(
    mut progress_information: ProgressInformation,
    snapshot_frequency_secs: u64,
    stop_for_snapshot: &AtomicBool,
    workers_stopped: Arc<(Mutex<bool>, Condvar)>,
    generator_stopped: Arc<RwLock<bool>>,
    workers_snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
    generator_snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
    all_combinations_generated: Arc<RwLock<bool>>,
    worker_pass_vectors: Vec<Arc<Mutex<Option<Vec<PassMsg>>>>>,
    num_running_workers: Arc<Mutex<usize>>,
    num_active_workers: Arc<Mutex<usize>>,
    global_queue: Arc<Injector<LetterCombination>>,
    generator_next_combination: Arc<Mutex<Option<LetterCombination>>>,
    batch_count: Arc<Mutex<u64>>,
    early_abort: Arc<Mutex<bool>>,
    abort_other_threads: Arc<RwLock<bool>>,
) {
    const CHECK_FOR_COMPLETION_INTERVAL_SECS: u64 = 30;
    assert!(
        snapshot_frequency_secs >= CHECK_FOR_COMPLETION_INTERVAL_SECS,
        "SNAPSHOT INTERVAL TOO SHORT"
    );
    // ensure that we always sleep at least once before a snapshot (can happen with very small snapshot intervals)
    // functionally enforces a minimum sleep_interval of 30 seconds
    let sleep_loop_count = max(
        1,
        snapshot_frequency_secs / CHECK_FOR_COMPLETION_INTERVAL_SECS,
    );

    let mut is_last_snapshot = false;
    let mut is_aborting_early = false;

    loop {
        progress_information.mark_batch_start();

        // take snapshots on the specified interval, waking periodically to check for overall completion
        for _ in 0..sleep_loop_count {
            thread::sleep(time::Duration::from_secs(
                CHECK_FOR_COMPLETION_INTERVAL_SECS,
            ));

            // n.b. this could technically be against the starting number of workers,
            // but I have another check after all workers are stopped anyways, which would
            // handle the case in which this is some positive integer less than the starting
            // number of workers
            if *num_active_workers
                .lock()
                .expect("snapshot thread paniced while holding num_active_workres mutex")
                == 0
            {
                println!("snapshot thread detected workers completed - taking last snapshot");
                is_last_snapshot = true;
                break;
            }

            if *early_abort
                .lock()
                .expect("generator or worker thread paniced while holding early_abort mutex")
            {
                is_aborting_early = true;
                println!("snapshot thread detected abort request");
                break;
            }
        }

        // reset indicators that the snapshot was complete, after we verify
        // that all threads (worker and generator) have resumed from the snapshot.
        // generator indicates this by setting generator_snapshot_complete (its stop indicator)
        // to false (or alternatively the generator has completed)
        // the workers indicate this by incrementing the number of running workers.
        // note that this check is particularly key as not all workers may have been scheduled
        // yet after a snapshot is completed
        while (*generator_snapshot_complete
            .0
            .lock()
            .expect("generator thread paniced while holding generator_snapshot_complete mutex")
            && !*all_combinations_generated.read().expect(
                "generator or worker thread paniced while holding all_combinations_generated mutex",
            ))
            || *num_running_workers
                .lock()
                .expect("worker thread paniced while holding num_running_workers mutex")
                != *num_active_workers
                    .lock()
                    .expect("worker thread paniced while holding num_active_workers mutex")
        {
            println!(
                "num running {} num active {}",
                *num_running_workers
                    .lock()
                    .expect("worker thread paniced while holding num_running_workers mutex"),
                *num_active_workers
                    .lock()
                    .expect("worker thread paniced while holding num_active_workers mutex")
            );

            thread::sleep(time::Duration::from_millis(100));
        }
        *workers_snapshot_complete
            .0
            .lock()
            .expect("worker thread paniced while holding workers_snapshot_complete mutex") = false;

        // trigger the snapshot
        progress_information.bump_snapshot_number();
        println!(
            "taking snapshot {}.",
            progress_information.get_snapshot_number()
        );
        stop_for_snapshot.store(true, MemoryOrdering::SeqCst);

        // wait for the other threads to have stopped
        // workers stop after the generator by constrution, so only need to check that
        // we need not (and should not) if we know that execution has already completed
        if !is_last_snapshot {
            println!("waiting for other threads to stop");
            let (ref workers_stopped_lock, ref workers_stopped_condvar) = *workers_stopped;
            let mut snapshot_completed_check = workers_stopped_lock
                .lock()
                .expect("worker thread paniced while holding workers_stopped mutex");
            while !*snapshot_completed_check {
                let result = workers_stopped_condvar
                    .wait_timeout(snapshot_completed_check, time::Duration::from_secs(60))
                    .expect("worker thread paniced while holding workers_stopped mutex");

                // this loop is racy with the last worker thread stopping.
                // prevent the case where the worker stops before we're waiting
                // on the condvar and we never get signaled :/
                if result.1.timed_out()
                    && *num_active_workers
                        .lock()
                        .expect("worker thread paniced while holding num_active_workers mutex")
                        == 0
                {
                    is_last_snapshot = true;
                    break;
                }

                snapshot_completed_check = result.0;
            }
        }

        // check that a snapshot invariant (empty queues at completion) has been satisfied
        if !global_queue.is_empty() {
            panic!(
                "global queue was not empty ({} combination(s)) at end of snapshot",
                global_queue.len()
            );
        }

        // recheck if any workers have stopped (workers might have finished out the last combinations during
        // this snapshot)
        if *num_active_workers
            .lock()
            .expect("worker thread paniced while holding num_active_workers mutex")
            == 0
        {
            println!("snapshot thread detected workers completed - last snapshot data gathered");
            is_last_snapshot = true;
        }

        // write the snapshot to disk in pieces to avoid memory pressure
        // (combining oom'd this with 64 GB memory on fedora 42)
        let mut batch_pass_count = 0;
        for (thread_num, mutex) in worker_pass_vectors.iter().enumerate() {
            let mut worker_vec_guard = mutex.lock().unwrap_or_else(|_| {
                panic!(
                    "worker thread {} paniced while holding its pass_vector mutex",
                    thread_num
                )
            });
            match *worker_vec_guard {
                Some(ref populated_vec) => {
                    batch_pass_count += populated_vec.len();

                    // compress the results to save disk space
                    let encoded_passing_results = bincode::serialize(&populated_vec)
                        .expect("failed to serialize passing results");
                    // for snapshot 2 on worker thread 5 would be 2W5
                    let snapshot_name = progress_information.get_snapshot_number().to_string()
                        + "W"
                        + &thread_num.to_string();
                    let snapshots_directory = progress_information.get_snapshots_directory();
                    let mut snapshot_path = snapshots_directory.to_path_buf();
                    snapshot_path.push(snapshot_name);

                    match fs::write(&snapshot_path, encoded_passing_results) {
                        Ok(_) => (),
                        Err(e) => panic!(
			    "write of serialized pass vectors to disk at {} failed due to error: {}",
			    snapshot_path.display(),
			    e
			),
                    }

                    // reset to force the worker to repopulate
                    *worker_vec_guard = None;
                }
                None => {
                    // it's an invariant violation for a worker thread to not populate its pass vector
                    // before the final worker indicates that the snapshot is complete
                    if !is_last_snapshot {
                        panic!(
                            "worker thread {} failed to populate its pass vector",
                            thread_num
                        );
                    }
                    // the last snapshot would be the exception - it's possible that some workers terminate
                    // during the penultimate snapshot, leaving some unpopulated vector(s) around for the
                    // final snapshot
                }
            }
        }

        // write progress information to disk (the snapshot consists of the results up to this
        // point, the next combination to be considered, and statistics about what we've processed so far)
        // it's important that this occurs after the previous write so that we do not erroneously
        // skip vectors if we fail in between these steps
        let next_combination = *generator_next_combination
            .lock()
            .expect("generator thread paniced while holding next_combination mutex");
        let batch_evaluated_count = *batch_count
            .lock()
            .expect("generator thread paniced while holding batch_count mutex");
        progress_information.update_with_batch(
            batch_pass_count as u64,
            batch_evaluated_count,
            next_combination.as_ref(),
            true,
        );

        // note that it doesn't matter if we're aborting here - the effect is the same
        // we're exiting anyways
        if is_last_snapshot {
            println!("snapshot thread terminating on last snapshot");
            return;
        }

        // if we are aborting early, then don't restart the other threads
        // as normal but rather signal them that they should abort too
        if is_aborting_early {
            println!("snapshot thread aborting other threads");
            *abort_other_threads
                .write()
                .expect("snapshot or worker thread paniced while holding abort_early mutex") = true;

            *generator_snapshot_complete
                .0
                .lock()
                .expect("generator paniced while holding generator_snapshot_complete mutex") = true;
            generator_snapshot_complete.1.notify_all();

            *workers_snapshot_complete
                .0
                .lock()
                .expect("worker thread paniced while holding workers_snapshot_complete mutex") =
                true;
            workers_snapshot_complete.1.notify_all();

            // indicate abortion recognized back to the initiator (currently for testing)
            // note that while state is persisted at this point, abortion is NOT complete
            // (that occurs after the CombinationSearch joins the worker and generator
            // threads)
            *early_abort
                .lock()
                .expect("signal handler thread paniced while holding early_abort mutex") = false;
            return;
        }

        // reset for next snapshot
        stop_for_snapshot.store(false, MemoryOrdering::SeqCst);

        // re-start the generator thread first to limit context switch
        // thrashing while it builds up a buffer in the global queue
        // reset from this snapshot
        *generator_stopped
            .write()
            .expect("generator thread paniced while holding generator_stopped mutex") = false;
        // signal generator to resume
        *generator_snapshot_complete
            .0
            .lock()
            .expect("generator thread paniced while holding generator_snapshot_complete mutex") =
            true;
        generator_snapshot_complete.1.notify_all();
        thread::sleep(SPIN_UP_WAIT);

        // re-start the worker threads
        // reset from this snapshot
        *workers_stopped
            .0
            .lock()
            .expect("worker thread paniced while holding workers_stopped mutex") = false;
        // signal workers to resume
        *workers_snapshot_complete
            .0
            .lock()
            .expect("worker thread paniced while holding workers_snapshot_complete mutex") = true;
        workers_snapshot_complete.1.notify_all();
    }
}
