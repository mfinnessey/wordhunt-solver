use crate::combination_search::progress_information::{
    ProgressInformation, PROGRESS_SNAPSHOT_IDENTIFIER,
};
use crate::combination_search::PassMsg;
use regex::Regex;
use std::collections::HashSet;
use std::fs::{read, read_dir};
use std::path::Path;

/// aggregates the snapshots from a directory into a vector
/// does not respect the order of the snapshots
pub fn aggregate_snapshots_from_directory<P: AsRef<Path>>(
    directory: P,
) -> Result<Vec<PassMsg>, String> {
    let snapshot_file_regex = match Regex::new(r"^*/[0-9]+$") {
        Ok(regex) => regex,
        Err(e) => panic!("failed to create snapshot_file regex due to error: {e}"),
    };

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
        if !snapshot_file_regex.is_match(&file_path_as_string) {
            continue;
        }

        if let Some(file_name) = file_path.file_name() {
            if let Ok(snapshot_number) = file_name
                .to_str()
                .expect("failed to convert snapshot filename to a utf-8 string")
                .parse()
            {
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
    // snapshots are 1-indexed
    for i in 1..max_snapshot_num {
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
    use std::fs;
    use std::fs::create_dir;
    use std::time::SystemTime;

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

        let vec1: Vec<PassMsg> = vec![v[0]];
        let encoded1 = bincode::serialize(&vec1).unwrap();
        fs::write(test_dir.clone() + "/1", encoded1).unwrap();

        let vec2 = &v[1..4].to_vec();
        let encoded2 = bincode::serialize(&vec2).unwrap();
        fs::write(test_dir.clone() + "/2", encoded2).unwrap();

        let vec3 = &v[4..6].to_vec();
        let encoded3 = bincode::serialize(&vec3).unwrap();
        fs::write(test_dir.clone() + "/3", encoded3).unwrap();

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

        let vec1: Vec<PassMsg> = vec![v[0]];
        let encoded1 = bincode::serialize(&vec1).unwrap();
        fs::write(test_dir.clone() + "/1", encoded1).unwrap();

        // oops, we forgot to dump vector 2

        let vec3 = &v[4..6].to_vec();
        let encoded3 = bincode::serialize(&vec3).unwrap();
        fs::write(test_dir.to_owned() + "/3", encoded3).unwrap();

        // throw in a bogus file for grins
        fs::write(test_dir.clone() + "/foo", "foo").unwrap();

        assert_eq!(
            aggregate_snapshots_from_directory(test_dir).unwrap_err(),
            "Missing snapshot numbers [2]"
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
        fs::write(test_dir.clone() + "/1", encoded1).unwrap();

        // oops, we dumped garbage
        let encoded2 = "foo!";
        fs::write(test_dir.clone() + "/2", encoded2).unwrap();

        let vec3 = &v[4..6].to_vec();
        let encoded3 = bincode::serialize(&vec3).unwrap();
        fs::write(test_dir.clone() + "/3", encoded3).unwrap();

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
	let actual = read_next_progress_information_from_directory(progress_info.get_snapshots_directory()).unwrap();
        assert_eq!(
            (actual.0, &actual.1),
            expected
        );

        // test that we still get the same result if we're missing some files
        for i in 0..NUM_PROGRESS_INFORMATION_FILES {
            if i % 2 != 0 {
		let mut snapshot_file_path = progress_info.get_snapshots_directory().clone();
		snapshot_file_path.push(i.to_string() + PROGRESS_SNAPSHOT_IDENTIFIER);
                fs::remove_file(
                    snapshot_file_path
                )
                .unwrap();
            }
        }
	let actual = read_next_progress_information_from_directory(progress_info.get_snapshots_directory()).unwrap();
        assert_eq!(
            (actual.0, &actual.1),
            expected
        );
    }
}
