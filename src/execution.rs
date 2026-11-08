//! Explicit, reusable worker pools for independent evaluation work.
//!
//! Rayon’s process-global pool is intentionally unused: callers choose the
//! budget, and a run or outer experiment holds its pool for its lifetime.

use anyhow::Result;
#[cfg(feature = "parallel")]
use std::sync::Arc;

#[cfg(feature = "parallel")]
use std::{
    collections::VecDeque,
    sync::{Mutex, OnceLock},
};
#[cfg(feature = "parallel")]
const MAX_IDLE_POOLS: usize = 4;
#[cfg(feature = "parallel")]
type PoolCache = Mutex<VecDeque<(usize, Arc<rayon::ThreadPool>)>>;
#[cfg(feature = "parallel")]
static CACHE: OnceLock<PoolCache> = OnceLock::new();

/// Thread budget for evaluation, holding a reusable Rayon pool when more than one
/// thread is requested. Cloning shares the pool.
#[derive(Clone, Debug)]
pub struct ExecutionContext {
    threads: usize,
    #[cfg(feature = "parallel")]
    pool: Option<Arc<std::sync::OnceLock<Arc<rayon::ThreadPool>>>>,
}

impl ExecutionContext {
    /// Create a context for `threads` workers. Errors if `threads` is zero, or above
    /// one when built without the `parallel` feature. The pool is created lazily.
    pub fn new(threads: usize) -> Result<Self> {
        anyhow::ensure!(threads > 0, "execution thread count must be positive");
        #[cfg(feature = "parallel")]
        {
            let pool = if threads > 1 {
                Some(Arc::new(std::sync::OnceLock::new()))
            } else {
                None
            };
            Ok(Self { threads, pool })
        }
        #[cfg(not(feature = "parallel"))]
        {
            anyhow::ensure!(
                threads == 1,
                "threads > 1 require rebuilding with --features parallel"
            );
            Ok(Self { threads })
        }
    }

    /// Configured thread count.
    pub fn threads(&self) -> usize {
        self.threads
    }
    /// Whether more than one thread is configured.
    pub fn is_parallel(&self) -> bool {
        self.threads > 1
    }

    /// Run `f` inside this context's pool (or directly on the caller for one thread).
    /// Errors only if the pool cannot be built.
    #[cfg(feature = "parallel")]
    pub fn install<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match &self.pool {
            Some(slot) => {
                if let Some(pool) = slot.get() {
                    Ok(pool.install(f))
                } else {
                    let pool = cached_pool(self.threads)?;
                    let _ = slot.set(pool);
                    Ok(slot
                        .get()
                        .expect("pool set by this or a concurrent install")
                        .install(f))
                }
            }
            None => Ok(f()),
        }
    }
}

#[cfg(feature = "parallel")]
impl Drop for ExecutionContext {
    fn drop(&mut self) {
        let Some(slot) = self.pool.take() else {
            return;
        };
        let had_pool = slot.get().is_some();
        drop(slot);
        if had_pool {
            trim_cached_idle();
        }
    }
}

#[cfg(feature = "parallel")]
fn cached_pool(threads: usize) -> Result<Arc<rayon::ThreadPool>> {
    let cache = CACHE.get_or_init(|| Mutex::new(VecDeque::new()));
    let mut pools = cache.lock().expect("execution pool cache poisoned");
    if let Some(index) = pools.iter().position(|(count, _)| *count == threads) {
        let item = pools.remove(index).expect("pool index exists");
        let pool = Arc::clone(&item.1);
        pools.push_front(item);
        let evicted = trim_idle(&mut pools);
        drop(pools);
        drop(evicted);
        return Ok(pool);
    }
    let evicted = trim_idle(&mut pools);
    // Construction is setup work. Holding the cache lock prevents concurrent
    // cold requests from creating duplicate pools for the same count.
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?,
    );
    pools.push_front((threads, Arc::clone(&pool)));
    drop(pools);
    drop(evicted);
    Ok(pool)
}

#[cfg(feature = "parallel")]
fn trim_idle(pools: &mut VecDeque<(usize, Arc<rayon::ThreadPool>)>) -> Vec<Arc<rayon::ThreadPool>> {
    let mut idle = pools
        .iter()
        .filter(|(_, pool)| Arc::strong_count(pool) == 1)
        .count();
    let mut removed = Vec::new();
    while idle > MAX_IDLE_POOLS {
        let index = pools
            .iter()
            .rposition(|(_, pool)| Arc::strong_count(pool) == 1)
            .expect("idle count is positive");
        removed.push(pools.remove(index).expect("idle index exists").1);
        idle -= 1;
    }
    removed
}

#[cfg(feature = "parallel")]
fn trim_cached_idle() {
    let cache = CACHE.get_or_init(|| Mutex::new(VecDeque::new()));
    let removed = trim_idle(&mut cache.lock().expect("execution pool cache poisoned"));
    drop(removed);
}

#[cfg(feature = "parallel")]
#[cfg(test)]
fn cached_pool_count() -> usize {
    CACHE
        .get_or_init(|| Mutex::new(VecDeque::new()))
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, pool)| Arc::strong_count(pool) == 1)
        .count()
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;
    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn contexts_reuse_pools_and_serial_context_has_none() {
        let _guard = TEST_LOCK.lock().unwrap();
        let a = ExecutionContext::new(3).unwrap();
        let b = ExecutionContext::new(3).unwrap();
        a.install(|| ()).unwrap();
        b.install(|| ()).unwrap();
        assert!(Arc::ptr_eq(
            a.pool.as_ref().unwrap().get().unwrap(),
            b.pool.as_ref().unwrap().get().unwrap()
        ));
        let serial = ExecutionContext::new(1).unwrap();
        assert!(serial.pool.is_none());
        let caller = std::thread::current().id();
        assert_eq!(
            serial.install(|| std::thread::current().id()).unwrap(),
            caller
        );
    }

    #[test]
    fn pool_cache_is_bounded() {
        let _guard = TEST_LOCK.lock().unwrap();
        let held = ExecutionContext::new(3).unwrap();
        held.install(|| ()).unwrap();
        for count in 5..12 {
            let context = ExecutionContext::new(count).unwrap();
            context.install(|| ()).unwrap();
        }
        assert!(cached_pool_count() <= MAX_IDLE_POOLS);
        let held_again = ExecutionContext::new(3).unwrap();
        held_again.install(|| ()).unwrap();
        assert!(Arc::ptr_eq(
            held.pool.as_ref().unwrap().get().unwrap(),
            held_again.pool.as_ref().unwrap().get().unwrap()
        ));
    }
}
