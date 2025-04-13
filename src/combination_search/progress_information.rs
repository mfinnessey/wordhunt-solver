use crate::letter_combination::LetterCombination;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

pub const PROGRESS_SNAPSHOT_IDENTIFIER: &str = "PROGRESS";
#[derive(Serialize, Deserialize, Eq, PartialEq, Debug)]
pub struct ProgressInformation {
    last_update_time: SystemTime,
    total_elapsed_time: Duration,
    pass_count: u64,
    evaluated_count: u64,
    next_combination: LetterCombination,
    snapshot_num: u32,
}

impl ProgressInformation {
    pub fn new(start_time: SystemTime, next_combination: LetterCombination) -> Self {
        Self {
            last_update_time: start_time,
            pass_count: 0,
            total_elapsed_time: Duration::ZERO,
            evaluated_count: 0,
            next_combination,
            snapshot_num: 0,
        }
    }

    pub fn update_with_batch(
        &mut self,
        batch_pass_count: u64,
        batch_evaluated_count: u64,
        next_combination: &LetterCombination,
        snapshot_num: u32,
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
        self.next_combination = *next_combination;
        self.snapshot_num = snapshot_num;

        // dump state to disk
        let encoded_progress_information = bincode::serialize(&self).unwrap();
        let progress_information_name = snapshot_num.to_string() + PROGRESS_SNAPSHOT_IDENTIFIER;
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
        println!("Next combination is: {}.", next_combination);
        println!("*****");
    }
}
