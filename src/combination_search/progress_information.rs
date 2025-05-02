use crate::letter_combination::LetterCombination;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

pub const PROGRESS_SNAPSHOT_IDENTIFIER: &str = "PROGRESS";
#[derive(Serialize, Deserialize, Eq, PartialEq, Debug, Clone)]
pub struct ProgressInformation {
    start_time: SystemTime,
    last_update_time: SystemTime,
    total_elapsed_time: Duration,
    pass_count: u64,
    evaluated_count: u64,
    next_combination: Option<LetterCombination>,
    snapshot_num: u32,
    snapshots_directory: PathBuf,
}

impl ProgressInformation {
    pub fn new(start_time: SystemTime, next_combination: LetterCombination) -> Self {
        // create the snapshots directory
        let secs_since_unix_epoch = start_time
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string();
        let path_string = "snapshots-".to_string() + &secs_since_unix_epoch;
        let snapshots_directory = Path::new(&path_string);
        match fs::create_dir(snapshots_directory) {
            Ok(_) => (),
            Err(_) => panic!("could not create snapshots directory"),
        }

        Self {
            start_time,
            last_update_time: start_time,
            pass_count: 0,
            total_elapsed_time: Duration::ZERO,
            evaluated_count: 0,
            next_combination: Some(next_combination),
            snapshot_num: 0,
            snapshots_directory: snapshots_directory.to_path_buf(),
        }
    }

    pub fn mark_batch_start(&mut self) {
        self.last_update_time = SystemTime::now();
    }

    pub fn update_with_batch(
        &mut self,
        batch_pass_count: u64,
        batch_evaluated_count: u64,
        next_combination: Option<&LetterCombination>,
        mut snapshots_directory: PathBuf,
        print_statistics: bool,
    ) {
        // update state
        let now = SystemTime::now();
        let batch_elapsed_time = match now.duration_since(self.last_update_time) {
            Ok(time) => time,
            // statistics aren't required to be accurate
            Err(_) => Duration::ZERO,
        };
        self.last_update_time = now;
        self.total_elapsed_time += batch_elapsed_time;
        self.pass_count += batch_pass_count;
        self.evaluated_count += batch_evaluated_count;
        self.next_combination = next_combination.copied();

        // dump state to disk
        let encoded_progress_information = bincode::serialize(&self).unwrap();
        let progress_information_name =
            self.snapshot_num.to_string() + PROGRESS_SNAPSHOT_IDENTIFIER;
        snapshots_directory.push(progress_information_name);
        fs::write(snapshots_directory, encoded_progress_information).unwrap();

        // bail out of other calculations if we're not printing statistics for whatever reason
        if !print_statistics {
            return;
        }

        let batch_fail_count = batch_evaluated_count - batch_pass_count;
        let batch_pass_rate: f64 =
	// prevent divide by zero
	    if batch_evaluated_count != 0 {
		(batch_pass_count as f64) / (batch_evaluated_count as f64)
	    }
	else {
	    0.0
	};

        let overall_fail_count: u64 = self.evaluated_count - self.pass_count;
        let overall_pass_rate: f64 =
	// prevent divide by zero
	    if self.evaluated_count != 0 {
		(self.pass_count as f64) / (self.evaluated_count as f64)
	    }
	else {
	    0.0
	};

        const SECS_PER_HOUR: f64 = 3600.0;

        let batch_elapsed_time_hours = (batch_elapsed_time.as_secs_f64()) / SECS_PER_HOUR;
        let batch_combinations_per_hour = (batch_evaluated_count as f64) / batch_elapsed_time_hours;

        let overall_elapsed_time_hours = (self.total_elapsed_time.as_secs_f64()) / SECS_PER_HOUR;
        let overall_combinations_per_hour =
            (self.evaluated_count as f64) / (overall_elapsed_time_hours);

        // TODO compute this...
        const TOTAL_COMBINATIONS_COUNT: u64 = 103_077_446_706;
        let remaining_combinations_count = TOTAL_COMBINATIONS_COUNT - self.evaluated_count;
        let remaining_combinations_percentage: f64 =
            100.0 * (remaining_combinations_count as f64) / (TOTAL_COMBINATIONS_COUNT as f64);
        let estimated_time_remaining_hours =
            (remaining_combinations_count as f64) / overall_combinations_per_hour;

        println!("*****");
        println!(
            "Batch: Total {} | Pass {} | Fail {} | Pass Rate {:.2} | Combinations Per Hour {}",
            batch_evaluated_count,
            batch_pass_count,
            batch_fail_count,
            batch_pass_rate,
            batch_combinations_per_hour
        );
        println!(
            "Overall: Total {} | Pass {} | Fail {} | Pass Rate {:.2} | Combinations Per Hour {}",
            self.evaluated_count,
            self.pass_count,
            overall_fail_count,
            overall_pass_rate,
            overall_combinations_per_hour
        );
        println!(
            "Remaining: {} ({:.2}%) of Total {})",
            remaining_combinations_count,
            remaining_combinations_percentage,
            TOTAL_COMBINATIONS_COUNT
        );
        println!(
            "Ran for {:.1} hours. Estimate {:.1} hours remaining",
            overall_elapsed_time_hours, estimated_time_remaining_hours
        );
        match next_combination {
            Some(combination) => println!("next combination is: {}.", combination),
            None => println!("no next combination (all combinations evaluated)"),
        }

        println!("*****");
    }

    pub fn get_next_combination(&self) -> Option<LetterCombination> {
        self.next_combination
    }

    pub fn get_start_time(&self) -> &SystemTime {
        &self.start_time
    }

    pub fn get_evaluated_count(&self) -> &u64 {
        &self.evaluated_count
    }

    pub fn get_snapshots_directory(&self) -> &PathBuf {
        &self.snapshots_directory
    }

    pub fn get_snapshot_number(&self) -> u32 {
        self.snapshot_num
    }

    pub fn bump_snapshot_number(&mut self) {
        self.snapshot_num += 1;
    }
}
