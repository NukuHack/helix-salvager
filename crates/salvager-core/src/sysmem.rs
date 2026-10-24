//! System / process memory and disk monitoring for huge-archive recovery.
//!
//! Provides:
//! - [`MemMonitor`] — cheap, rate-limited snapshots of free RAM, free swap /
//!   pagefile, this process's own resident memory and free disk space
//!   (container cgroup limits are honoured on Linux).
//! - [`Governor`] — turns snapshots into a decision: how large the read
//!   buffer may be *right now*, and whether the machine is under so much
//!   pressure that work should pause.
//!
//! The huge-file engine never relies on swap: it sizes its working set from
//! *available physical RAM* so the OS never has to page it out.

use std::path::Path;
use std::time::{Duration, Instant};
use sysinfo::{Disks, Pid, ProcessesToUpdate, System};

pub const MIB: u64 = 1024 * 1024;
pub const GIB: u64 = 1024 * MIB;

/// A point-in-time view of memory.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemSnapshot {
    /// Total physical RAM (or cgroup limit if lower).
    pub total_ram: u64,
    /// RAM that can be handed out without swapping (free + reclaimable cache).
    pub available_ram: u64,
    /// Free swap / pagefile space.
    pub free_swap: u64,
    /// Total swap / pagefile space.
    pub total_swap: u64,
    /// Resident memory of this process.
    pub self_rss: u64,
}

/// Rate-limited memory / disk sampler.
pub struct MemMonitor {
    sys: System,
    pid: Option<Pid>,
    last: MemSnapshot,
    last_at: Option<Instant>,
    min_interval: Duration,
}

impl Default for MemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl MemMonitor {
    pub fn new() -> Self {
        Self {
            sys: System::new(),
            pid: sysinfo::get_current_pid().ok(),
            last: MemSnapshot::default(),
            last_at: None,
            min_interval: Duration::from_millis(500),
        }
    }

    /// Take a fresh sample (ignores the rate limit).
    pub fn sample(&mut self) -> MemSnapshot {
        self.sys.refresh_memory();
        let mut total = self.sys.total_memory();
        let mut avail = self.sys.available_memory();
        if avail == 0 {
            avail = self.sys.free_memory();
        }

        // Containers: honour cgroup limits when they are tighter than the host.
        if let Some(cg) = self.sys.cgroup_limits() {
            if cg.total_memory > 0 && cg.total_memory < total {
                total = cg.total_memory;
                avail = avail.min(cg.free_memory);
            }
        }

        let mut rss = 0;
        if let Some(pid) = self.pid {
            self.sys
                .refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
            if let Some(p) = self.sys.process(pid) {
                rss = p.memory();
            }
        }

        self.last = MemSnapshot {
            total_ram: total,
            available_ram: avail,
            free_swap: self.sys.free_swap(),
            total_swap: self.sys.total_swap(),
            self_rss: rss,
        };
        self.last_at = Some(Instant::now());
        self.last
    }

    /// Sample at most every ~500 ms; otherwise return the cached value.
    pub fn snapshot(&mut self) -> MemSnapshot {
        match self.last_at {
            Some(t) if t.elapsed() < self.min_interval => self.last,
            _ => self.sample(),
        }
    }

    /// Free space (bytes) on the volume that holds `path`.
    /// Returns `None` if it cannot be determined.
    pub fn free_disk_space(&self, path: &Path) -> Option<u64> {
        let abs = std::fs::canonicalize(path)
            .or_else(|_| {
                // Path may not exist yet — canonicalise the nearest existing parent.
                let mut p = path.to_path_buf();
                while !p.exists() {
                    if !p.pop() {
                        break;
                    }
                }
                std::fs::canonicalize(p)
            })
            .ok()?;
        let disks = Disks::new_with_refreshed_list();
        disks
            .list()
            .iter()
            .filter(|d| abs.starts_with(d.mount_point()))
            .max_by_key(|d| d.mount_point().as_os_str().len())
            .map(|d| d.available_space())
    }
}

/// What the governor wants the engine to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pressure {
    /// Plenty of memory.
    Ok,
    /// Memory is getting tight — buffers were shrunk.
    Low,
    /// Even minimum-size buffers are risky — pause / checkpoint.
    Critical,
}

/// Adaptive buffer sizing.
#[derive(Debug, Clone)]
pub struct Governor {
    /// Hard budget for this process (bytes).
    pub budget: u64,
    /// Largest buffer we would like to use.
    pub max_buf: usize,
    /// Smallest buffer we will go down to.
    pub min_buf: usize,
    /// Current buffer size.
    pub cur_buf: usize,
    pub shrink_events: u64,
}

impl Governor {
    /// Decide the budget from a snapshot and an optional user cap.
    ///
    /// * auto budget = half of the currently *available* RAM
    ///   (never more than 8 GiB, never less than 64 MiB)
    /// * a user cap can only lower it
    pub fn new(snap: &MemSnapshot, user_limit: Option<u64>) -> Self {
        let auto = (snap.available_ram / 2).clamp(64 * MIB, 8 * GIB);
        let budget = match user_limit {
            // Even an explicit cap may not exceed 80 % of what is free now.
            Some(l) => l.min((snap.available_ram / 10 * 8).max(64 * MIB)).max(32 * MIB),
            None => auto,
        };
        // Peak usage ≈ 2 × buffer (buffer + in-memory file for refinement)
        // + dedup table, so give the buffer a third of the budget.
        let max_buf = ((budget / 3).clamp(2 * MIB, 512 * MIB) / MIB * MIB) as usize;
        Self {
            budget,
            max_buf,
            min_buf: (MIB as usize).min(max_buf),
            cur_buf: max_buf,
            shrink_events: 0,
        }
    }

    /// Re-evaluate the buffer size against current conditions.
    pub fn evaluate(&mut self, snap: &MemSnapshot) -> Pressure {
        // Low-water mark: keep the machine responsive.
        let low = (snap.total_ram / 20).max(256 * MIB);
        let critical = (snap.total_ram / 50).max(96 * MIB);

        let over_budget = snap.self_rss > self.budget + self.budget / 2;
        let avail = snap.available_ram;

        if avail < critical || over_budget {
            if self.cur_buf > self.min_buf {
                self.cur_buf = (self.cur_buf / 2).max(self.min_buf);
                self.shrink_events += 1;
                return Pressure::Low;
            }
            // Already at the floor.
            return if avail < critical {
                Pressure::Critical
            } else {
                Pressure::Low
            };
        }
        if avail < low {
            if self.cur_buf > self.min_buf {
                self.cur_buf = (self.cur_buf / 2).max(self.min_buf);
                self.shrink_events += 1;
            }
            return Pressure::Low;
        }
        // Plenty of room: grow back toward max, one doubling at a time, and
        // only if there is clear headroom after the growth.
        if self.cur_buf < self.max_buf {
            let next = (self.cur_buf * 2).min(self.max_buf);
            let extra = (next - self.cur_buf) as u64 * 2;
            if avail > low + extra {
                self.cur_buf = next;
            }
        }
        Pressure::Ok
    }
}

/// Human-friendly byte formatting.
pub fn fmt_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{} B", n)
    } else {
        format!("{:.1} {}", v, UNITS[u])
    }
}

/// Parse sizes like `512M`, `4G`, `1.5GiB`, `2048` (MiB if bare number).
pub fn parse_size(s: &str) -> Option<u64> {
    let t = s.trim().to_ascii_lowercase();
    let t = t.trim_end_matches('b').trim_end_matches('i');
    let (num, mult) = if let Some(n) = t.strip_suffix('k') {
        (n, 1024.0)
    } else if let Some(n) = t.strip_suffix('m') {
        (n, MIB as f64)
    } else if let Some(n) = t.strip_suffix('g') {
        (n, GIB as f64)
    } else if let Some(n) = t.strip_suffix('t') {
        (n, (GIB * 1024) as f64)
    } else {
        (t, MIB as f64) // bare number = MiB
    };
    num.trim().parse::<f64>().ok().map(|v| (v * mult) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(total: u64, avail: u64, rss: u64) -> MemSnapshot {
        MemSnapshot {
            total_ram: total,
            available_ram: avail,
            free_swap: 0,
            total_swap: 0,
            self_rss: rss,
        }
    }

    #[test]
    fn budget_is_half_of_available() {
        let g = Governor::new(&snap(16 * GIB, 8 * GIB, 0), None);
        assert_eq!(g.budget, 4 * GIB);
        assert!(g.max_buf as u64 <= 512 * MIB);
    }

    #[test]
    fn user_cap_lowers_budget() {
        let g = Governor::new(&snap(16 * GIB, 8 * GIB, 0), Some(256 * MIB));
        assert_eq!(g.budget, 256 * MIB);
        assert!(g.max_buf as u64 <= 256 * MIB / 3 + MIB);
    }

    #[test]
    fn user_cap_cannot_exceed_free_ram() {
        let g = Governor::new(&snap(8 * GIB, GIB, 0), Some(64 * GIB));
        assert!(g.budget <= GIB);
    }

    #[test]
    fn shrinks_then_goes_critical() {
        let total = 4 * GIB;
        let mut g = Governor::new(&snap(total, 2 * GIB, 0), None);
        let start = g.cur_buf;
        let tight = snap(total, 50 * MIB, 0);
        assert_eq!(g.evaluate(&tight), Pressure::Low);
        assert!(g.cur_buf < start);
        let mut last = Pressure::Low;
        for _ in 0..20 {
            last = g.evaluate(&tight);
        }
        assert_eq!(last, Pressure::Critical);
        assert_eq!(g.cur_buf, g.min_buf);
    }

    #[test]
    fn grows_back_when_memory_returns() {
        let total = 8 * GIB;
        let mut g = Governor::new(&snap(total, 4 * GIB, 0), None);
        let max = g.max_buf;
        for _ in 0..10 {
            g.evaluate(&snap(total, 100 * MIB, 0));
        }
        assert_eq!(g.cur_buf, g.min_buf);
        for _ in 0..20 {
            g.evaluate(&snap(total, 6 * GIB, 0));
        }
        assert_eq!(g.cur_buf, max);
    }

    #[test]
    fn own_rss_over_budget_shrinks() {
        let total = 16 * GIB;
        let mut g = Governor::new(&snap(total, 8 * GIB, 0), None);
        let before = g.cur_buf;
        g.evaluate(&snap(total, 8 * GIB, g.budget * 2));
        assert!(g.cur_buf < before);
    }

    #[test]
    fn size_parsing() {
        assert_eq!(parse_size("512M"), Some(512 * MIB));
        assert_eq!(parse_size("4G"), Some(4 * GIB));
        assert_eq!(parse_size("1.5GiB"), Some(GIB + GIB / 2));
        assert_eq!(parse_size("2048"), Some(2048 * MIB));
        assert_eq!(parse_size("junk"), None);
    }

    #[test]
    fn monitor_reports_something() {
        let mut m = MemMonitor::new();
        let s = m.sample();
        assert!(s.total_ram > 0);
        assert!(s.self_rss > 0);
    }
}
