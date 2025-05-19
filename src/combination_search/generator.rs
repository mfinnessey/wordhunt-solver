use crate::letter_combination::LetterCombination;
use crossbeam_deque::Injector;
use std::sync::atomic::{AtomicBool, Ordering as MemoryOrdering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::{thread, time};

// thread needs a lot of state passed in. not bundling into a struct as there is a singular generator thread
#[allow(clippy::too_many_arguments)]
/// push combinations into the global queue
pub fn generate_combinations(
    combinations: impl Iterator<Item = LetterCombination>,
    queue: Arc<Injector<LetterCombination>>,
    max_target_queue_size: usize,
    all_combinations_generated: Arc<RwLock<bool>>,
    stop_for_snapshot: &AtomicBool,
    generator_thread_stopped: Arc<RwLock<bool>>,
    snapshot_complete: Arc<(Mutex<bool>, Condvar)>,
    aborting_early: Arc<RwLock<bool>>,
    next_combination: Arc<Mutex<Option<LetterCombination>>>,
    batch_count: Arc<Mutex<u64>>,
) {
    let mut local_batch_count: u64 = 0;

    for combination in combinations {
        // block the generator thread if the queue is ludicrously big.
        let mut stopped = false;
        while local_batch_count % (max_target_queue_size as u64) == 0
            && queue.len() >= max_target_queue_size
        {
            if !stopped {
                println!("Generator ran ahead, sleeping for now.");
                stopped = true;
            }

            // jump out of here to take the snapshot if need be (check at interval)
            if stop_for_snapshot.load(MemoryOrdering::SeqCst) {
                break;
            }
            thread::sleep(time::Duration::from_secs(5));
        }

        // can consider checking every x iteratons only if the atomic cas
        // winds up being too expensive (guessing that it can be parallelized / branch
        // predicted efficiently enough)
        if stop_for_snapshot.load(MemoryOrdering::SeqCst) {
            // write state that the snapshot thread needs
            *next_combination
                .lock()
                .expect("snapshot thread paniced while holding next_combination mutex") =
                Some(combination);
            *batch_count
                .lock()
                .expect("snapshot thread paniced while holding batch_count mutex") =
                local_batch_count;

            // notify worker threads that we have stopped generating new combinations
            // and intentionally drop the guard asap to try to avoid blocking readers
            // n.b. if the generator thread were to be the last thread to stop,
            // (very unlikely given the amount of work that the worker threads have to do)
            // then the flag would prevent the last worker thread from stopping until
            // this thread has stopped, thus ensuring that the queues are emptied.
            {
                let mut generation_stopped = generator_thread_stopped.write().expect(
                    "snapshot or worker thread paniced while holding generation_stopped rwlock",
                );
                *generation_stopped = true;
            }

            println!(
                "Generator stopped for snapshot with {} items remaining in global queue.",
                queue.len()
            );

            // block until the snapshot is complete
            let (ref lock, ref condvar) = *snapshot_complete;
            let mut snapshot_completed_check = lock
                .lock()
                .expect("snapshot thread paniced while holding generator snapshot complete mutex");
            while !*snapshot_completed_check {
                snapshot_completed_check = condvar.wait(snapshot_completed_check).expect(
                    "snapshot thread paniced while holding generator snapshot complete mutex",
                );
            }

            if *aborting_early
                .read()
                .expect("snapshot or worker thread paniced while holding generator abortion rwlock")
            {
                println!("generator thread detected early abort");
                return;
            }

            *snapshot_completed_check = false;
            println!("Generator thread resuming after snapshot.");

            // reset batch
            local_batch_count = 0;
        }

        if stopped {
            println!("Generator thread resuming after running ahead.");
        }

        queue.push(combination);
        local_batch_count += 1;
    }

    *batch_count
        .lock()
        .expect("snapshot thread paniced while holding batch_count mutex") = local_batch_count;
    *next_combination
        .lock()
        .expect("snapshot thread paniced while holding next_combination mutex") = None;
    *generator_thread_stopped.write().expect(
        "snapshot or worker thread paniced while holding generator_thread_stopped rwlock",
    ) = true;
    *all_combinations_generated.write().expect(
        "snapshot or worker thread paniced while holding all_combinations_generated rwlock ",
    ) = true;
    println!("generated all combinations");
}
