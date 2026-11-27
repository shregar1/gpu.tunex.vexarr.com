//! Per-kernel GPU time profiler.
//!
//! When enabled (e.g. via `GPUX_PROFILE=1`), every dispatch issued through
//! `CommandQueue::enqueue` is wrapped with `synchronize()` calls and CPU-timed,
//! then accumulated by kernel function name.
//!
//! This gives a serialized-execution view of where wall time goes — much slower
//! than a normal run, but precise enough to identify which kernels dominate.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

#[derive(Default, Debug, Clone, Copy)]
pub struct KernelStats {
    pub total: Duration,
    pub calls: u64,
}

pub struct KernelProfiler {
    enabled: AtomicBool,
    stats: Mutex<HashMap<String, KernelStats>>,
}

impl KernelProfiler {
    fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            stats: Mutex::new(HashMap::new()),
        }
    }

    pub fn global() -> &'static Self {
        static P: OnceLock<KernelProfiler> = OnceLock::new();
        P.get_or_init(KernelProfiler::new)
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    pub fn reset(&self) {
        self.stats.lock().clear();
    }

    pub fn record(&self, kernel: &str, elapsed: Duration) {
        let mut stats = self.stats.lock();
        let entry = stats.entry(kernel.to_string()).or_default();
        entry.total += elapsed;
        entry.calls += 1;
    }

    pub fn snapshot(&self) -> Vec<(String, KernelStats)> {
        let stats = self.stats.lock();
        let mut entries: Vec<(String, KernelStats)> = stats
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        entries.sort_by_key(|(_, s)| std::cmp::Reverse(s.total));
        entries
    }

    pub fn print_report(&self, title: &str) {
        let snap = self.snapshot();
        let total: Duration = snap.iter().map(|(_, s)| s.total).sum();
        eprintln!("");
        eprintln!("=== {} ===", title);
        eprintln!(
            "  {:<40}  {:>8}  {:>10}  {:>9}  {:>6}",
            "kernel", "calls", "total (ms)", "per (µs)", "% time"
        );
        eprintln!("  {}", "-".repeat(82));
        for (name, s) in &snap {
            let pct = if total.as_nanos() > 0 {
                100.0 * s.total.as_secs_f64() / total.as_secs_f64()
            } else {
                0.0
            };
            let per_us = if s.calls > 0 {
                s.total.as_secs_f64() * 1e6 / s.calls as f64
            } else {
                0.0
            };
            eprintln!(
                "  {:<40}  {:>8}  {:>10.3}  {:>9.1}  {:>5.1}%",
                name,
                s.calls,
                s.total.as_secs_f64() * 1000.0,
                per_us,
                pct
            );
        }
        eprintln!("  {}", "-".repeat(82));
        eprintln!(
            "  {:<40}  {:>8}  {:>10.3}",
            "TOTAL",
            snap.iter().map(|(_, s)| s.calls).sum::<u64>(),
            total.as_secs_f64() * 1000.0,
        );
    }
}
