use super::progress_information::{ProgressInformation, PROGRESS_SNAPSHOT_IDENTIFIER};
use super::{PassMsg, SPIN_UP_WAIT};
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
// TODO dump stuff to a log file (e.g. snapshot numbers, etc.)
#[allow(clippy::too_many_arguments)]
/// periodically take snapshots
pub fn take_snapshots(
    snapshot_frequency_secs: u64,
    stop_for_snapshot: &AtomicBool,
    workers_stopped: Arc<(Mutex<bool>, Condvar)>,
    generator_stopped: Arc<RwLock<bool>>,
    workers_snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
    generator_snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
    all_combinations_generated: Arc<RwLock<bool>>,
    worker_pass_vectors: Vec<Arc<Mutex<Vec<PassMsg>>>>,
    num_running_workers: Arc<Mutex<usize>>,
    num_active_workers: Arc<Mutex<usize>>,
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
    let first_combination = LetterCombination::from(ALL_A_FREQUENCIES);
    // resize syntactically requires something to overwrite elements with in case of resizing up
    let default_pass_msg: PassMsg = (first_combination, 0);
    let mut snapshot_number: u32 = 0;

    let mut progress_statistics = ProgressInformation::new(start_time, first_combination);
    let mut is_last_snapshot = false;

    loop {
        // take snapshots on the specified interval, waking periodically to check for overall completion
        for _ in 0..sleep_loop_count {
            thread::sleep(time::Duration::from_secs(
                CHECK_FOR_COMPLETION_INTERVAL_SECS,
            ));

            if *num_active_workers.lock().unwrap() == 0 {
                is_last_snapshot = true;
                break;
            }
        }

        if *num_active_workers.lock().unwrap() == 0 {
            is_last_snapshot = true;
        }

        // reset indicators that the snapshot was complete, after we verify
        // that all threads (worker and generator) have resumed from the snapshot.
        // generator indicates this by setting generator_snapshot_complete (its stop indicator)
        // to false (or alternatively the generator has completed)
        // the workers indicate this by incrementing the number of running workers.
        // note that this check is particularly key as not all workers may have been scheduled
        // yet after a snapshot is completed
        while (*generator_snapshot_complete.0.lock().unwrap()
            && !*all_combinations_generated.read().unwrap())
            || *num_running_workers.lock().unwrap() != *num_active_workers.lock().unwrap()
        {
            thread::sleep(time::Duration::from_secs(
                CHECK_FOR_COMPLETION_INTERVAL_SECS,
            ));
        }
        *workers_snapshot_complete.0.lock().unwrap() = false;

        // trigger the snapshot
        println!("Taking snapshot {}.", snapshot_number);
        stop_for_snapshot.store(true, MemoryOrdering::SeqCst);

        // wait for the other threads to have stopped
        // workers stop after the generator by constrution, so only need to check that
        // we need not (and should not) if we know that execution has already completed
        if !is_last_snapshot {
            println!("waiting for other threads to stop");
            let (ref workers_stopped_lock, ref workers_stopped_condvar) = *workers_stopped;
            let mut snapshot_completed_check = workers_stopped_lock.lock().unwrap();
            while !*snapshot_completed_check {
                let result = workers_stopped_condvar
                    .wait_timeout(snapshot_completed_check, time::Duration::from_secs(60))
                    .unwrap();

                // this loop is racy with the last worker thread stopping.
                // prevent the case where the worker stops before we're waiting
                // on the condvar and we never get signaled :/
                if result.1.timed_out() && *num_active_workers.lock().unwrap() == 0 {
                    is_last_snapshot = true;
                    break;
                }

                snapshot_completed_check = result.0;
            }
        }

        // check that a snapshot invariant (empty queues at completion) has been satisfied
        if !global_queue.is_empty() {
            panic!(
                "Global queue was not empty ({} combination(s)) at end of snapshot",
                global_queue.len()
            );
        }

        // write the snapshot to disk
        // aggregate worker vectors for a single write
        let mut passing_results: Vec<PassMsg> = Vec::new();
        for mutex in worker_pass_vectors.iter() {
            let mut worker_vec = mutex.lock().unwrap();
            passing_results.extend(worker_vec.iter());
            // reset to an empty vector to correctly handle the pathological case where the last snapshot
            // is triggered with some workers stopping and others terminating to avoid duplicating
            // the last snapshot on the final "cleanup" snapshot triggered when no workers are active
            *worker_vec = Vec::new();
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

        // write progress information to disk (the snapshot consists of the results up to this
        // point, the next combination to be considered, and statistics about what we've processed so far)
        // it's important that this occurs after the previous write so that we do not erroneously
        // skip vectors if we fail in between these steps
        let next_combination = *generator_next_combination.lock().unwrap();
        let batch_evaluated_count = *batch_count.lock().unwrap();
        progress_statistics.update_with_batch(
            batch_pass_count as u64,
            batch_evaluated_count,
            &next_combination,
            snapshot_number,
            snapshots_directory.to_path_buf(),
            true,
        );

        if is_last_snapshot {
            println!("snapshot thread terminating on last snapshot");
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

/// read the next combination saved to a snapshot directory
pub fn read_next_progress_information_from_directory<P: AsRef<Path>>(
    directory: P,
) -> Result<ProgressInformation, String> {
    // find all next combination files
    let snapshot_file_regex = Regex::new(r"^*/[0-9]+PROGRESS").unwrap();
    const PROGRESS_LENGTH: usize = 8;
    let mut max_snapshot_num: u32 = 0;
    let mut max_snapshot_file_path = None;

    let files = read_dir(directory).map_err(|_e| "Could not read directory!")?;

    for file in files {
        let file = file.map_err(|_e| "Error reading file returned in directory")?;
        let file_path = file.path();
        // ignore sub-directories (which shouldn't exist by construction, but belt & suspenders)
        if !file_path.is_file() {
            continue;
        }
        // ignore non-next combination files
        let file_path_as_string = file_path.clone().into_os_string().into_string().unwrap();
        if !snapshot_file_regex.is_match(&file_path_as_string) {
            continue;
        }

        if let Some(file_name) = file_path.file_name() {
            let file_name_str = file_name.to_str().unwrap();

            let num_slice = &file_name_str[..file_name_str.len() - PROGRESS_LENGTH];
            // no point in not unwrapping given it's validated by the regex
            let snapshot_num: u32 = num_slice.parse().unwrap();

            // retain the path to the maximal snapshot file that we've fouund so far (including the first)
            // the is_none check ensures that max_snapshot_num will be initialized with real data by
            // construction before its first reading
            if max_snapshot_file_path.is_none() || snapshot_num > max_snapshot_num {
                max_snapshot_num = snapshot_num;
                max_snapshot_file_path = Some(file_path);
            }
        } else {
            return Err("File has no name!".to_string());
        }
    }

    match max_snapshot_file_path {
        Some(path) => {
            let data = read(path).map_err(|_e| "Error reading file data")?;
            let deser =
                bincode::deserialize::<ProgressInformation>(&data).map_err(|e| e.to_string())?;
            Ok(deser)
        }
        None => Err("No next combination files found in directory".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::test_utilities::TestCleanup;
    use crate::utilities::TILE_COUNT;
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

    #[test]
    fn test_read_next_progress_information_from_directory() {
        let test_dir = TEMP_DIR.to_owned() + "test_read_next_progress_information_from_directory";
        let _cleanup = TestCleanup::new(test_dir.clone());

        create_dir(test_dir.clone()).unwrap();

        // dump a bunch of progress information files
        const NUM_PROGRESS_INFORMATION_FILES: usize = 131;
        let mut frequencies = ALL_A_FREQUENCIES;
        let mut to_idx = 1;
        let mut lc = LetterCombination::new(frequencies);
        let mut progress_info = ProgressInformation::new(SystemTime::now(), lc);
        let mut batch_pass_count = 5;
        let mut batch_evaluated_count = 10;
        for i in 0..NUM_PROGRESS_INFORMATION_FILES {
            lc = LetterCombination::new(frequencies);

            // TODO update the progress statistics here...
            progress_info.update_with_batch(
                batch_pass_count,
                batch_evaluated_count,
                &lc,
                i.try_into().unwrap(),
                test_dir.clone().into(),
                true,
            );

            // update information
            frequencies[to_idx - 1] -= 1;
            frequencies[to_idx] += 1;
            if frequencies[to_idx] as usize == TILE_COUNT {
                to_idx += 1;
            }

            batch_pass_count += (i % 10) as u64;
            batch_evaluated_count += (i % 10) as u64;
        }

        // throw in a bogus file for grins
        fs::write(test_dir.clone() + "/foo", "foo").unwrap();

        assert_eq!(
            read_next_progress_information_from_directory(&test_dir).unwrap(),
            progress_info
        );

        // test that we still get the same result if we're missing some files
        for i in 0..NUM_PROGRESS_INFORMATION_FILES {
            if i % 2 != 0 {
                fs::remove_file(
                    test_dir.clone() + "/" + &i.to_string() + PROGRESS_SNAPSHOT_IDENTIFIER,
                )
                .unwrap();
            }
        }
        assert_eq!(
            read_next_progress_information_from_directory(&test_dir).unwrap(),
            progress_info
        );
    }
}
