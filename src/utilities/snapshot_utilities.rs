use crate::combination_search::progress_information::{
    ProgressInformation, PROGRESS_SNAPSHOT_IDENTIFIER,
};
use crate::combination_search::PassMsg;
use regex::Regex;
use std::cmp::max;
use std::collections::HashMap;
use std::fs::{read, read_dir};
use std::path::Path;

/// aggregates the snapshots from a directory into a vector
/// does not respect the order of the snapshots
pub fn aggregate_snapshots_from_directory<P: AsRef<Path>>(
    directory: P,
) -> Result<Vec<PassMsg>, String> {
    let snapshot_file_regex = match Regex::new(r"^*/([0-9])+W([0-9])+$") {
        Ok(regex) => regex,
        Err(e) => panic!("failed to create snapshot_file regex due to error: {e}"),
    };

    let mut seen_snapshots: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut pass_msgs: Vec<PassMsg> = Vec::new();

    let mut max_snapshot_num = 0;
    let mut max_worker_num = 0;
    let files = read_dir(directory).map_err(|_e| "Could not read directory!")?;
    for file in files {
        let file = file.map_err(|_e| "Error reading file returned in directory")?;
        let file_path = file.path();
        // ignore sub-directories (which shouldn't exist by construction, but belt & suspenders)
        if !file_path.is_file() {
            continue;
        }
        // ignore non-snapshot files
        let file_path_as_string = file_path
            .clone()
            .into_os_string()
            .into_string()
            .unwrap_or_else(|_| {
                panic!(
                    "failed to convert file path {} into a utf-8 string",
                    file_path.display()
                )
            });

        match snapshot_file_regex.captures(&file_path_as_string) {
            Some(captures) => {
                let snapshot_num_str = match captures.get(1) {
                    Some(snapshot_num_match) => snapshot_num_match.as_str(),
                    None => continue,
                };
                let snapshot_num = snapshot_num_str
                    .parse()
                    .expect("regex-validated snapshot number failed to parse");
                let worker_num_str = match captures.get(2) {
                    Some(worker_num_match) => worker_num_match.as_str(),
                    None => continue,
                };
                let worker_num = worker_num_str
                    .parse()
                    .expect("regex-validated worker number failed to parse");

                max_snapshot_num = max(snapshot_num, max_snapshot_num);
                max_worker_num = max(worker_num, max_worker_num);

                if let Some(worker_snapshot_numbers) = seen_snapshots.get_mut(&snapshot_num) {
                    worker_snapshot_numbers.push(worker_num);
                } else {
                    seen_snapshots.insert(snapshot_num, vec![worker_num]);
                }
            }
            None => continue,
        }

        let data = read(file_path).map_err(|_e| "Error reading file data")?;
        let deser = &mut bincode::deserialize(&data).map_err(|e| e.to_string())?;
        pass_msgs.append(deser);
    }

    if max_snapshot_num == 0 {
        return Err("READ NO SNAPSHOTS".to_string());
    }

    // ensure that we're not missing any snapshots
    let mut missing_snapshot_numbers = Vec::new();
    let mut missing_snapshot_worker_pairs = Vec::new();
    // snapshots are 1-indexed
    for snapshot_num in 1..max_snapshot_num {
        if let Some(worker_snapshot_numbers) = seen_snapshots.get(&snapshot_num) {
            if snapshot_num != max_snapshot_num {
                for worker_num in 0..max_worker_num {
                    if !worker_snapshot_numbers.contains(&worker_num) {
                        missing_snapshot_worker_pairs.push((snapshot_num, worker_num));
                    }
                }
            }
        } else {
            missing_snapshot_numbers.push(snapshot_num);
        }
    }

    if missing_snapshot_numbers.is_empty() && missing_snapshot_worker_pairs.is_empty() {
        println!("read all {} 0-indexed snapshots", max_snapshot_num);
        Ok(pass_msgs)
    } else {
        Err(format!(
            "missing snapshot numbers {:?} and (snapshot, worker) pairs {:?}",
            missing_snapshot_numbers, missing_snapshot_worker_pairs
        ))
    }
}

/// read the next combination saved to a snapshot directory
pub fn read_next_progress_information_from_directory<P: AsRef<Path>>(
    directory: P,
) -> Result<(u32, ProgressInformation), String> {
    // find all next combination files
    let progress_information_regex = match Regex::new(r"^*/[0-9]+PROGRESS") {
        Ok(regex) => regex,
        Err(e) => panic!(
            "failed to create progrss_information_regex due to error: {}",
            e
        ),
    };

    let progress_length = PROGRESS_SNAPSHOT_IDENTIFIER.len();
    let mut max_progress_information_num: u32 = 0;
    let mut max_progress_information_file_path = None;

    let files = read_dir(directory).map_err(|_e| "Could not read directory!")?;

    for file in files {
        let file = file.map_err(|_e| "Error reading file returned in directory")?;
        let file_path = file.path();
        // ignore sub-directories (which shouldn't exist by construction, but belt & suspenders)
        if !file_path.is_file() {
            continue;
        }
        // ignore non-next combination files
        let file_path_as_string = file_path
            .clone()
            .into_os_string()
            .into_string()
            .unwrap_or_else(|_| {
                panic!(
                    "failed to convert filepath {} into a utf-8 string",
                    file_path.display()
                )
            });
        if !progress_information_regex.is_match(&file_path_as_string) {
            continue;
        }

        if let Some(file_name) = file_path.file_name() {
            let file_name_str = file_name
                .to_str()
                .expect("failed to convert progress information filename into a utf-8 string");

            let num_slice = &file_name_str[..file_name_str.len() - progress_length];
            let progress_information_num: u32 = num_slice.parse().unwrap_or_else(|_| {
                panic!(
                    "regex-validated number {} failed to parse as a number",
                    num_slice
                )
            });

            // retain the path to the maximal snapshot file that we've fouund so far (including the first)
            // the is_none check ensures that max_snapshot_num will be initialized with real data by
            // construction before its first reading
            if max_progress_information_file_path.is_none()
                || progress_information_num > max_progress_information_num
            {
                max_progress_information_num = progress_information_num;
                max_progress_information_file_path = Some(file_path);
            }
        } else {
            return Err("File has no name!".to_string());
        }
    }

    match max_progress_information_file_path {
        Some(path) => {
            let data = read(path).map_err(|_e| "Error reading file data")?;
            let deser =
                bincode::deserialize::<ProgressInformation>(&data).map_err(|e| e.to_string())?;
            Ok((max_progress_information_num, deser))
        }
        None => Err("No next combination files found in directory".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::letter_combination::LetterCombination;
    use crate::utilities::test_utilities::TestCleanup;
    use crate::utilities::{ALL_A_FREQUENCIES, TILE_COUNT};
    use std::collections::HashSet;
    use std::fs;
    use std::fs::create_dir;
    use std::time::SystemTime;

    const TEMP_DIR: &str = "unittest";

    #[test]
    fn test_aggregate_snapshots_from_directory() {
        let test_dir = TEMP_DIR.to_owned() + "test_aggregate_snapshots_from_directory";
        let _cleanup = TestCleanup::new(test_dir.clone());
        let mut v0: Vec<PassMsg> = Vec::new();
        let mut v1: Vec<PassMsg> = Vec::new();
        let mut v2: Vec<PassMsg> = Vec::new();
        let mut frequencies0 = ALL_A_FREQUENCIES;
        let mut frequencies1 = ALL_A_FREQUENCIES;
        let mut frequencies2 = ALL_A_FREQUENCIES;
        for i in 0..6 {
            v0.push((LetterCombination::new(frequencies0), i));
            v1.push((LetterCombination::new(frequencies1), i));
            v2.push((LetterCombination::new(frequencies2), i));
            frequencies0[0] -= 1;
            frequencies0[1] += 1;
            frequencies1[0] -= 1;
            frequencies1[1] += 1;
            frequencies2[0] -= 1;
            frequencies2[2] += 1;
        }

        // dump vectors in heterogeneous pieces
        create_dir(test_dir.clone()).unwrap();

        let vec1w0: Vec<PassMsg> = vec![v0[0]];
        let encoded1w0 = bincode::serialize(&vec1w0).unwrap();
        fs::write(test_dir.clone() + "/1W0", encoded1w0).unwrap();
        let vec1w1: Vec<PassMsg> = vec![v1[0]];
        let encoded1w1 = bincode::serialize(&vec1w1).unwrap();
        fs::write(test_dir.clone() + "/1W1", encoded1w1).unwrap();
        let encoded1w2 = bincode::serialize(&v2[0..2]).unwrap();
        fs::write(test_dir.clone() + "/1W2", encoded1w2).unwrap();

        let vec2w0 = &v0[1..4].to_vec();
        let encoded2w0 = bincode::serialize(&vec2w0).unwrap();
        fs::write(test_dir.clone() + "/2W0", encoded2w0).unwrap();
        let vec2w1 = &v1[1..3].to_vec();
        let encoded2w1 = bincode::serialize(&vec2w1).unwrap();
        fs::write(test_dir.clone() + "/2W1", encoded2w1).unwrap();
        let vec2w2 = &v2[2..4].to_vec();
        let encoded2w2 = bincode::serialize(&vec2w2).unwrap();
        fs::write(test_dir.clone() + "/2W2", encoded2w2).unwrap();

        let vec3w0 = &v0[4..6].to_vec();
        let encoded3w0 = bincode::serialize(&vec3w0).unwrap();
        fs::write(test_dir.clone() + "/3W0", encoded3w0).unwrap();
        let vec3w1 = &v1[3..6].to_vec();
        let encoded3w1 = bincode::serialize(&vec3w1).unwrap();
        fs::write(test_dir.clone() + "/3W1", encoded3w1).unwrap();
        let vec3w2 = &v2[4..6].to_vec();
        let encoded3w2 = bincode::serialize(&vec3w2).unwrap();
        fs::write(test_dir.clone() + "/3W2", encoded3w2).unwrap();

        // throw in a bogus file for grins
        fs::write(test_dir.clone() + "/foo", "foo").unwrap();

        let mut expected: HashSet<PassMsg> = HashSet::from_iter(v0);
        expected.extend(v1);
        expected.extend(v2);

        let actual = HashSet::from_iter(aggregate_snapshots_from_directory(test_dir).unwrap());

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_aggregate_snapshots_from_directory_not_all_last_worker() {
        let test_dir =
            TEMP_DIR.to_owned() + "test_aggregate_snapshots_from_directory_not_all_last_worker";
        let _cleanup = TestCleanup::new(test_dir.clone());
        let mut v0: Vec<PassMsg> = Vec::new();
        let mut v1: Vec<PassMsg> = Vec::new();
        let mut v2: Vec<PassMsg> = Vec::new();
        let mut frequencies0 = ALL_A_FREQUENCIES;
        let mut frequencies1 = ALL_A_FREQUENCIES;
        let mut frequencies2 = ALL_A_FREQUENCIES;
        for i in 0..6 {
            v0.push((LetterCombination::new(frequencies0), i));
            v1.push((LetterCombination::new(frequencies1), i));
            v2.push((LetterCombination::new(frequencies2), i));
            frequencies0[0] -= 1;
            frequencies0[1] += 1;
            frequencies1[0] -= 1;
            frequencies1[1] += 1;
            frequencies2[0] -= 1;
            frequencies2[2] += 1;
        }

        // dump vectors in heterogeneous pieces
        create_dir(test_dir.clone()).unwrap();

        let vec1w0: Vec<PassMsg> = vec![v0[0]];
        let encoded1w0 = bincode::serialize(&vec1w0).unwrap();
        fs::write(test_dir.clone() + "/1W0", encoded1w0).unwrap();
        let vec1w1: Vec<PassMsg> = vec![v1[0]];
        let encoded1w1 = bincode::serialize(&vec1w1).unwrap();
        fs::write(test_dir.clone() + "/1W1", encoded1w1).unwrap();
        let encoded1w2 = bincode::serialize(&v2[0..2]).unwrap();
        fs::write(test_dir.clone() + "/1W2", encoded1w2).unwrap();

        let vec2w0 = &v0[1..4].to_vec();
        let encoded2w0 = bincode::serialize(&vec2w0).unwrap();
        fs::write(test_dir.clone() + "/2W0", encoded2w0).unwrap();
        let vec2w1 = &v1[1..3].to_vec();
        let encoded2w1 = bincode::serialize(&vec2w1).unwrap();
        fs::write(test_dir.clone() + "/2W1", encoded2w1).unwrap();
        let vec2w2 = &v2[2..6].to_vec();
        let encoded2w2 = bincode::serialize(&vec2w2).unwrap();
        fs::write(test_dir.clone() + "/2W2", encoded2w2).unwrap();

        let vec3w0 = &v0[4..6].to_vec();
        let encoded3w0 = bincode::serialize(&vec3w0).unwrap();
        fs::write(test_dir.clone() + "/3W0", encoded3w0).unwrap();
        let vec3w1 = &v1[3..6].to_vec();
        let encoded3w1 = bincode::serialize(&vec3w1).unwrap();
        fs::write(test_dir.clone() + "/3W1", encoded3w1).unwrap();
        // worker 2 terminated early - this is ok

        // throw in a bogus file for grins
        fs::write(test_dir.clone() + "/foo", "foo").unwrap();

        let mut expected: HashSet<PassMsg> = HashSet::from_iter(v0);
        expected.extend(v1);
        expected.extend(v2);

        let actual = HashSet::from_iter(aggregate_snapshots_from_directory(test_dir).unwrap());

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_aggregate_snapshots_from_directory_missing_snapshot() {
        let test_dir =
            TEMP_DIR.to_owned() + "test_aggregate_snapshots_from_directory_missing_snapshot";
        let _cleanup = TestCleanup::new(test_dir.clone());
        let mut v0: Vec<PassMsg> = Vec::new();
        let mut v1: Vec<PassMsg> = Vec::new();
        let mut v2: Vec<PassMsg> = Vec::new();
        let mut frequencies0 = ALL_A_FREQUENCIES;
        let mut frequencies1 = ALL_A_FREQUENCIES;
        let mut frequencies2 = ALL_A_FREQUENCIES;
        for i in 0..6 {
            v0.push((LetterCombination::new(frequencies0), i));
            v1.push((LetterCombination::new(frequencies1), i));
            v2.push((LetterCombination::new(frequencies2), i));
            frequencies0[0] -= 1;
            frequencies0[1] += 1;
            frequencies1[0] -= 1;
            frequencies1[1] += 1;
            frequencies2[0] -= 1;
            frequencies2[2] += 1;
        }

        // dump vectors in heterogeneous pieces
        create_dir(test_dir.clone()).unwrap();

        let vec1w0: Vec<PassMsg> = vec![v0[0]];
        let encoded1w0 = bincode::serialize(&vec1w0).unwrap();
        fs::write(test_dir.clone() + "/1W0", encoded1w0).unwrap();
        let vec1w1: Vec<PassMsg> = vec![v1[0]];
        let encoded1w1 = bincode::serialize(&vec1w1).unwrap();
        fs::write(test_dir.clone() + "/1W1", encoded1w1).unwrap();
        let encoded1w2 = bincode::serialize(&v2[0..2]).unwrap();
        fs::write(test_dir.clone() + "/1W2", encoded1w2).unwrap();

        // oops, we forgot to write snapshot 2

        let vec3w0 = &v0[4..6].to_vec();
        let encoded3w0 = bincode::serialize(&vec3w0).unwrap();
        fs::write(test_dir.clone() + "/3W0", encoded3w0).unwrap();
        let vec3w1 = &v1[3..6].to_vec();
        let encoded3w1 = bincode::serialize(&vec3w1).unwrap();
        fs::write(test_dir.clone() + "/3W1", encoded3w1).unwrap();
        let vec3w2 = &v2[4..6].to_vec();
        let encoded3w2 = bincode::serialize(&vec3w2).unwrap();
        fs::write(test_dir.clone() + "/3W2", encoded3w2).unwrap();

        // throw in a bogus file for grins
        fs::write(test_dir.clone() + "/foo", "foo").unwrap();

        assert_eq!(
            aggregate_snapshots_from_directory(test_dir).unwrap_err(),
            "missing snapshot numbers [2] and (snapshot, worker) pairs []"
        );
    }

    #[test]
    fn test_aggregate_snapshots_from_directory_missing_worker() {
        let test_dir =
            TEMP_DIR.to_owned() + "test_aggregate_snapshots_from_directory_missing_worker";
        let _cleanup = TestCleanup::new(test_dir.clone());
        let mut v0: Vec<PassMsg> = Vec::new();
        let mut v1: Vec<PassMsg> = Vec::new();
        let mut v2: Vec<PassMsg> = Vec::new();
        let mut frequencies0 = ALL_A_FREQUENCIES;
        let mut frequencies1 = ALL_A_FREQUENCIES;
        let mut frequencies2 = ALL_A_FREQUENCIES;
        for i in 0..6 {
            v0.push((LetterCombination::new(frequencies0), i));
            v1.push((LetterCombination::new(frequencies1), i));
            v2.push((LetterCombination::new(frequencies2), i));
            frequencies0[0] -= 1;
            frequencies0[1] += 1;
            frequencies1[0] -= 1;
            frequencies1[1] += 1;
            frequencies2[0] -= 1;
            frequencies2[2] += 1;
        }

        // dump vectors in heterogeneous pieces
        create_dir(test_dir.clone()).unwrap();

        let vec1w0: Vec<PassMsg> = vec![v0[0]];
        let encoded1w0 = bincode::serialize(&vec1w0).unwrap();
        fs::write(test_dir.clone() + "/1W0", encoded1w0).unwrap();
        // oops - what happened to worker 1 here??
        let encoded1w2 = bincode::serialize(&v2[0..2]).unwrap();
        fs::write(test_dir.clone() + "/1W2", encoded1w2).unwrap();

        let vec2w0 = &v0[1..4].to_vec();
        let encoded2w0 = bincode::serialize(&vec2w0).unwrap();
        fs::write(test_dir.clone() + "/2W0", encoded2w0).unwrap();
        let vec2w1 = &v1[1..3].to_vec();
        let encoded2w1 = bincode::serialize(&vec2w1).unwrap();
        fs::write(test_dir.clone() + "/2W1", encoded2w1).unwrap();
        let vec2w2 = &v2[2..4].to_vec();
        let encoded2w2 = bincode::serialize(&vec2w2).unwrap();
        fs::write(test_dir.clone() + "/2W2", encoded2w2).unwrap();

        let vec3w0 = &v0[4..6].to_vec();
        let encoded3w0 = bincode::serialize(&vec3w0).unwrap();
        fs::write(test_dir.clone() + "/3W0", encoded3w0).unwrap();
        let vec3w1 = &v1[3..6].to_vec();
        let encoded3w1 = bincode::serialize(&vec3w1).unwrap();
        fs::write(test_dir.clone() + "/3W1", encoded3w1).unwrap();
        let vec3w2 = &v2[4..6].to_vec();
        let encoded3w2 = bincode::serialize(&vec3w2).unwrap();
        fs::write(test_dir.clone() + "/3W2", encoded3w2).unwrap();

        // throw in a bogus file for grins
        fs::write(test_dir.clone() + "/foo", "foo").unwrap();

        assert_eq!(
            aggregate_snapshots_from_directory(test_dir).unwrap_err(),
            "missing snapshot numbers [] and (snapshot, worker) pairs [(1, 1)]"
        );
    }

    #[test]
    fn test_aggregate_snapshots_from_directory_missing_snapshot_and_worker() {
        let test_dir = TEMP_DIR.to_owned()
            + "test_aggregate_snapshots_from_directory_missing_snapshot_and_worker";
        let _cleanup = TestCleanup::new(test_dir.clone());
        let mut v0: Vec<PassMsg> = Vec::new();
        let mut v1: Vec<PassMsg> = Vec::new();
        let mut v2: Vec<PassMsg> = Vec::new();
        let mut frequencies0 = ALL_A_FREQUENCIES;
        let mut frequencies1 = ALL_A_FREQUENCIES;
        let mut frequencies2 = ALL_A_FREQUENCIES;
        for i in 0..6 {
            v0.push((LetterCombination::new(frequencies0), i));
            v1.push((LetterCombination::new(frequencies1), i));
            v2.push((LetterCombination::new(frequencies2), i));
            frequencies0[0] -= 1;
            frequencies0[1] += 1;
            frequencies1[0] -= 1;
            frequencies1[1] += 1;
            frequencies2[0] -= 1;
            frequencies2[2] += 1;
        }

        // dump vectors in heterogeneous pieces
        create_dir(test_dir.clone()).unwrap();

        let vec1w0: Vec<PassMsg> = vec![v0[0]];
        let encoded1w0 = bincode::serialize(&vec1w0).unwrap();
        fs::write(test_dir.clone() + "/1W0", encoded1w0).unwrap();
        // oops - what happened to worker 1 here??
        let encoded1w2 = bincode::serialize(&v2[0..2]).unwrap();
        fs::write(test_dir.clone() + "/1W2", encoded1w2).unwrap();

        // oops - what happened to snapshot 2?

        let vec3w0 = &v0[4..6].to_vec();
        let encoded3w0 = bincode::serialize(&vec3w0).unwrap();
        fs::write(test_dir.clone() + "/3W0", encoded3w0).unwrap();
        let vec3w1 = &v1[3..6].to_vec();
        let encoded3w1 = bincode::serialize(&vec3w1).unwrap();
        fs::write(test_dir.clone() + "/3W1", encoded3w1).unwrap();
        let vec3w2 = &v2[4..6].to_vec();
        let encoded3w2 = bincode::serialize(&vec3w2).unwrap();
        fs::write(test_dir.clone() + "/3W2", encoded3w2).unwrap();

        // throw in a bogus file for grins
        fs::write(test_dir.clone() + "/foo", "foo").unwrap();

        assert_eq!(
            aggregate_snapshots_from_directory(test_dir).unwrap_err(),
            "missing snapshot numbers [2] and (snapshot, worker) pairs [(1, 1)]"
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

        let vec1: Vec<PassMsg> = vec![v[0]];
        let encoded1 = bincode::serialize(&vec1).unwrap();
        fs::write(test_dir.clone() + "/1W0", encoded1).unwrap();

        // oops, we dumped garbage
        let encoded2 = "foo!";
        fs::write(test_dir.clone() + "/2W0", encoded2).unwrap();

        let vec3 = &v[4..6].to_vec();
        let encoded3 = bincode::serialize(&vec3).unwrap();
        fs::write(test_dir.clone() + "/3W0", encoded3).unwrap();

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
        // dump a bunch of progress information files
        const NUM_PROGRESS_INFORMATION_FILES: usize = 131;
        let mut frequencies = ALL_A_FREQUENCIES;
        let mut to_idx = 1;
        let mut lc = LetterCombination::new(frequencies);
        let mut progress_info = ProgressInformation::new(SystemTime::now(), lc);
        let _cleanup = TestCleanup::new(progress_info.get_snapshots_directory().clone());
        let mut batch_pass_count = 5;
        let mut batch_evaluated_count = 10;
        for i in 0..NUM_PROGRESS_INFORMATION_FILES {
            lc = LetterCombination::new(frequencies);

            progress_info.bump_snapshot_number();
            progress_info.update_with_batch(
                batch_pass_count,
                batch_evaluated_count,
                Some(&lc),
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
        let mut bogus_file_path = progress_info.get_snapshots_directory().clone();
        bogus_file_path.push("foo");
        println!("{}", bogus_file_path.display());
        fs::write(bogus_file_path, "foo").unwrap();

        let expected = ((NUM_PROGRESS_INFORMATION_FILES) as u32, &progress_info);
        let actual =
            read_next_progress_information_from_directory(progress_info.get_snapshots_directory())
                .unwrap();
        assert_eq!((actual.0, &actual.1), expected);

        // test that we still get the same result if we're missing some files
        for i in 0..NUM_PROGRESS_INFORMATION_FILES {
            if i % 2 != 0 {
                let mut snapshot_file_path = progress_info.get_snapshots_directory().clone();
                snapshot_file_path.push(i.to_string() + PROGRESS_SNAPSHOT_IDENTIFIER);
                fs::remove_file(snapshot_file_path).unwrap();
            }
        }
        let actual =
            read_next_progress_information_from_directory(progress_info.get_snapshots_directory())
                .unwrap();
        assert_eq!((actual.0, &actual.1), expected);
    }
}
