use super::{progress_statistics::ProgressStatistics, PassMsg, SPIN_UP_WAIT};
use crate::letter_combination::LetterCombination;
use crate::utilities::ALL_A_FREQUENCIES;
use crossbeam_deque::Injector;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering as MemoryOrdering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::SystemTime;
use std::{fs, thread, time};

// no way around passing everything in, and no sense in a convenience struct
// given that it's a singular thread
#[allow(clippy::too_many_arguments)]
/// periodically take snapshots
pub fn take_snapshots(
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
