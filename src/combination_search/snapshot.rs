use super::{progress_statistics::ProgressStatistics, PassMsg, SPIN_UP_WAIT};
use crate::letter_combination::LetterCombination;
use crate::utilities::ALL_A_FREQUENCIES;
use crossbeam_deque::Injector;
use regex::Regex;
use std::collections::HashSet;
use std::fs::{read, read_dir};
use std::path::{Path, PathBuf};
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
) -> PathBuf {
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
        let mut snapshot_path = snapshots_directory.to_path_buf();
        snapshot_path.push(snapshot_name);

        // intentionally panic if the filesystem operation fails
        fs::write(snapshot_path, encoded_passing_results).unwrap();

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

        // write statistics
        let batch_evaluated_count = *batch_count.lock().unwrap();
        progress_statistics.update_with_batch(
            batch_pass_count as u64,
            batch_evaluated_count,
            &next_combination,
            true,
        );

        if is_last_snapshot {
            return snapshots_directory.into();
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

/// aggregates the snapshots from a directory into a vector
/// does not respect the order of the snapshots
pub fn aggregate_snapshots_from_directory<P: AsRef<Path>>(
    directory: P,
) -> Result<Vec<PassMsg>, String> {
    let snapshot_file_regex = Regex::new(r"^*/[0-9]+$").unwrap();

    let mut seen_snapshot_numbers = HashSet::new();
    let mut pass_msgs: Vec<PassMsg> = Vec::new();

    let files = read_dir(directory).map_err(|_e| "Could not read directory!")?;
    for file in files {
        let file = file.map_err(|_e| "Error reading file returned in directory")?;
        let file_path = file.path();
        // ignore sub-directories (which shouldn't exist by construction, but belt & suspenders)
        if !file_path.is_file() {
            continue;
        }
        // ignore non-snapshot files
        let file_path_as_string = file_path.clone().into_os_string().into_string().unwrap();
        if !snapshot_file_regex.is_match(&file_path_as_string) {
            continue;
        }

        if let Some(file_name) = file_path.file_name() {
            if let Ok(snapshot_number) = file_name.to_str().unwrap().parse() {
                seen_snapshot_numbers.insert(snapshot_number);
            } else {
                return Err("Regex-validated file name did not parse as a number".to_string());
            }
        } else {
            return Err("File has no name!".to_string());
        }

        let data = read(file_path).map_err(|_e| "Error reading file data")?;
        let deser = &mut bincode::deserialize(&data).map_err(|e| e.to_string())?;
        pass_msgs.append(deser);
    }

    // ensure that we're not missing any snapshots
    let max_snapshot_num: u32 = *seen_snapshot_numbers
        .iter()
        .max()
        .ok_or("READ NO SNAPSHOTS")?;
    let mut missing_snapshot_numbers = Vec::new();
    for i in 0..max_snapshot_num {
        if !seen_snapshot_numbers.contains(&i) {
            missing_snapshot_numbers.push(i);
        }
    }

    if missing_snapshot_numbers.is_empty() {
        println!("Read all {} 0-indexed snapshots", max_snapshot_num);
        Ok(pass_msgs)
    } else {
        Err(format!(
            "Missing snapshot numbers {:?}",
            missing_snapshot_numbers
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::test_utilities::TestCleanup;
    use std::fs::create_dir;

    const TEMP_DIR: &str = "unittest";

    #[test]
    fn test_aggregate_snapshots_from_directory() {
        let test_dir = TEMP_DIR.to_owned() + "test_aggregate_snapshots_from_directory";
        let _cleanup = TestCleanup::new(test_dir.clone());
        let mut v: Vec<PassMsg> = Vec::new();
        let mut frequencies = ALL_A_FREQUENCIES;
        for i in 0..6 {
            v.push((LetterCombination::new(frequencies), i));
            frequencies[0] -= 1;
            frequencies[1] += 1;
        }

        // dump vector in heterogeneous but consecutive pieces a la snapshots

        create_dir(test_dir.clone()).unwrap();

        let vec0: Vec<PassMsg> = vec![v[0]];
        let encoded0 = bincode::serialize(&vec0).unwrap();
        fs::write(test_dir.clone() + "/0", encoded0).unwrap();

        let vec1 = &v[1..4].to_vec();
        let encoded1 = bincode::serialize(&vec1).unwrap();
        fs::write(test_dir.clone() + "/1", encoded1).unwrap();

        let vec2 = &v[4..6].to_vec();
        let encoded2 = bincode::serialize(&vec2).unwrap();
        fs::write(test_dir.clone() + "/2", encoded2).unwrap();

        // throw in a bogus file for grins
        fs::write(test_dir.clone() + "/foo", "foo").unwrap();

        assert_eq!(aggregate_snapshots_from_directory(test_dir).unwrap(), v);
    }

    #[test]
    fn test_aggregate_snapshots_from_directory_missing_snapshot() {
        let test_dir =
            TEMP_DIR.to_owned() + "test_aggregate_snapshots_from_directory_missing_snapshot";
        let _cleanup = TestCleanup::new(test_dir.clone());
        let mut v: Vec<PassMsg> = Vec::new();
        let mut frequencies = ALL_A_FREQUENCIES;
        for i in 0..6 {
            v.push((LetterCombination::new(frequencies), i));
            frequencies[0] -= 1;
            frequencies[1] += 1;
        }

        // dump vector in heterogeneous but consecutive pieces a la snapshots
        create_dir(test_dir.clone()).unwrap();

        let vec0: Vec<PassMsg> = vec![v[0]];
        let encoded0 = bincode::serialize(&vec0).unwrap();
        fs::write(test_dir.clone() + "/0", encoded0).unwrap();

        // oops, we forgot to dump vector 1

        let vec2 = &v[4..6].to_vec();
        let encoded2 = bincode::serialize(&vec2).unwrap();
        fs::write(test_dir.to_owned() + "/2", encoded2).unwrap();

        // throw in a bogus file for grins
        fs::write(test_dir.clone() + "/foo", "foo").unwrap();

        assert_eq!(
            aggregate_snapshots_from_directory(test_dir).unwrap_err(),
            "Missing snapshot numbers [1]"
        );
    }

    #[test]
    fn test_aggregate_snapshots_from_directory_bogus_snapshot() {
        let test_dir =
            TEMP_DIR.to_owned() + "test_aggregate_snapshots_from_directory_bogus_snapshot";
        let _cleanup = TestCleanup::new(test_dir.clone());
        let mut v: Vec<PassMsg> = Vec::new();
        let mut frequencies = ALL_A_FREQUENCIES;
        for i in 0..6 {
            v.push((LetterCombination::new(frequencies), i));
            frequencies[0] -= 1;
            frequencies[1] += 1;
        }

        // dump vector in heterogeneous but consecutive pieces a la snapshots

        create_dir(test_dir.clone()).unwrap();

        let vec0: Vec<PassMsg> = vec![v[0]];
        let encoded0 = bincode::serialize(&vec0).unwrap();
        fs::write(test_dir.clone() + "/0", encoded0).unwrap();

        // oops, we dumped garbage
        let encoded1 = "foo!";
        fs::write(test_dir.clone() + "/1", encoded1).unwrap();

        let vec2 = &v[4..6].to_vec();
        let encoded2 = bincode::serialize(&vec2).unwrap();
        fs::write(test_dir.clone() + "/2", encoded2).unwrap();

        // throw in a bogus file for grins
        fs::write(test_dir.clone() + "/foo", "foo").unwrap();

        // only check for errors as we could reasonably expect to get multiple kinds of errors
        // (i.e. not enough data, doesn't decode, etc.)
        // this is a rough sanity check anyways as arbitrary data could in theory map
        // to some garbage vector unfortunately (no checksumming)
        assert!(aggregate_snapshots_from_directory(test_dir).is_err());
    }
}
