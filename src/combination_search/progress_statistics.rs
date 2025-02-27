use crate::letter_combination::LetterCombination;
use std::time::SystemTime;

pub struct ProgressStatistics {
    start_time: SystemTime,
    pass_count: u64,
    evaluated_count: u64,
    batch_number: u64,
}

impl ProgressStatistics {
    pub fn new(start_time: SystemTime) -> Self {
        Self {
            start_time,
            pass_count: 0,
            evaluated_count: 0,
            batch_number: 0,
        }
    }

    pub fn update_with_batch(
        &mut self,
        batch_pass_count: u64,
        batch_evaluated_count: u64,
        next_combination: &LetterCombination,
        print_statistics: bool,
    ) {
        // update state
        self.pass_count += batch_pass_count;
        self.evaluated_count += batch_evaluated_count;
        self.batch_number += 1;

        if !print_statistics {
            return;
        }

        // print statistics
        const TOTAL_COMBINATIONS_COUNT: u64 = 103_077_446_706;

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

        let remaining_combinations_count = TOTAL_COMBINATIONS_COUNT - self.evaluated_count;
        let remaining_combinations_percentage: f64 =
            100.0 * (remaining_combinations_count as f64) / (TOTAL_COMBINATIONS_COUNT as f64);

        let elapsed_time = SystemTime::now().duration_since(self.start_time);
        let elapsed_time_hours = match elapsed_time {
            Ok(time) => (time.as_secs_f64()) / 3600.0,
            Err(_) => 0.0,
        };

        // ratio of remaining work to completed work applied to elapsed time
        let hours_per_combination = elapsed_time_hours / (self.evaluated_count as f64);
        let estimated_time_remaining_hours =
            (hours_per_combination * TOTAL_COMBINATIONS_COUNT as f64) - elapsed_time_hours;

        println!("*****");
        println!("Wrote snapshot to disk.");
        println!(
            "Batch: Total {} | Pass {} | Fail {} | Pass Rate {:.2}",
            batch_evaluated_count, batch_pass_count, batch_fail_count, batch_pass_rate
        );
        println!(
            "Overall: Total {} | Pass {} | Fail {} | Pass Rate {:.2}",
            self.evaluated_count, self.pass_count, overall_fail_count, overall_pass_rate
        );
        println!(
            "Remaining: {} ({:.2}%) of Total {})",
            remaining_combinations_count,
            remaining_combinations_percentage,
            TOTAL_COMBINATIONS_COUNT
        );
        println!(
            "Ran for {:.1} hours. Estimate {:.1} hours remaining",
            elapsed_time_hours, estimated_time_remaining_hours
        );
        println!("Next combination is: {}.", next_combination);
        println!("*****");
    }
}
