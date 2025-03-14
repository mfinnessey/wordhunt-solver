use std::fs::remove_dir_all;
use std::path::Path;

/// convenience struct to ensure that temporary test files get cleaned up
/// even after a panic
pub struct TestCleanup<P: AsRef<Path>> {
    // individual test dirs to enable test cases to run in parallel
    test_dir: Option<P>,
}

impl<P: AsRef<Path>> TestCleanup<P> {
    pub fn new(test_dir: P) -> Self {
        Self {
            test_dir: Some(test_dir),
        }
    }
}

impl<P: AsRef<Path>> Drop for TestCleanup<P> {
    fn drop(&mut self) {
        if let Some(dir) = self.test_dir.take() {
            remove_dir_all(dir).unwrap();
        }
    }
}
