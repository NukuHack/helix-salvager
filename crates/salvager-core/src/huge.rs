//! Huge-archive recovery — works on inputs far larger than RAM **and** swap.
//!
//! The classic [`SalvageEngine::salvage`](crate::SalvageEngine::salvage) path
//! needs the whole input (and every recovered file) in memory. This module
//! never does. Its memory use is bounded by a *budget* derived from the
//! machine's currently-available RAM, and it is re-checked while running:
//!
//! * **Streaming input** — the file is read through a fixed-size buffer
//!   (no `mmap`, so nothing is ever charged to the pagefile).
//! * **Streaming output** — every recovered file is written to disk as soon
//!   as it is complete; only small metadata stays in RAM. Big files are
//!   spilled to disk while they are still being carved.
//! * **Adaptive buffer** — [`Governor`] watches free RAM, swap and this
//!   process's own RSS and shrinks / regrows the buffer; under critical
//!   pressure it checkpoints and pauses instead of thrashing.
//! * **Checkpoint / resume** — progress is persisted atomically in
//!   `<out>/.helix-state/`; a crash, Ctrl-C, full disk or low-memory pause
//!   can be continued with `resume = true`.
//!
//! Modes (auto-detected):
//!
//! | Input | Strategy |
//! |---|---|
//! | ZIP with a readable central directory | entry-by-entry streaming extraction |
//! | ZIP with a damaged / missing directory (e.g. truncated download) | local-header scan, deflate streamed per entry, partial entries kept |
//! | tar | streaming entry extraction |
//! | gzip / bzip2 / xz | streamed decompression → tar extraction, or → carving |
//! | anything else (7z, RAR, disk images, raw blobs) | streaming signature carver |

use crate::salvager::{CarvedFileType, SalvageEngine, SalvagedFile};
use crate::sysmem::{fmt_bytes, Governor, MemMonitor, MemSnapshot, Pressure, MIB};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Bytes kept at the end of the buffer between iterations so signatures that
/// straddle two reads are still seen.
const KEEP: usize = 1024;
/// Largest in-memory file we hold for end-marker refinement + validation.
const MEM_FILE_CAP: usize = 64 * 1024 * 1024;
const STATE_DIR: &str = ".helix-state";

// ──────────────────────────────────────────────────────────
//  Public types
// ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HugeConfig {
    /// Cap on this process's working memory (bytes). `None` = automatic
    /// (half of currently available RAM). Can only *lower* the automatic value.
    pub mem_limit: Option<u64>,
    /// Continue a previous run found in the output directory.
    pub resume: bool,
    /// Minimum seconds between checkpoint writes.
    pub checkpoint_secs: u64,
    /// Ignore carved fragments smaller than this.
    pub min_file_size: usize,
    /// Stop (with a checkpoint) when free disk space drops below this.
    pub disk_reserve: u64,
    /// How long to wait for memory to recover under critical pressure
    /// before stopping with a checkpoint.
    pub pause_wait_secs: u64,
    /// Skip structured extraction and always use the signature carver.
    pub force_carve: bool,
}

impl Default for HugeConfig {
    fn default() -> Self {
        Self {
            mem_limit: None,
            resume: false,
            checkpoint_secs: 10,
            min_file_size: 32,
            disk_reserve: 512 * MIB,
            pause_wait_secs: 120,
            force_carve: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HugeProgress {
    pub mode: String,
    pub position: u64,
    pub total: u64,
    pub files: u64,
    pub bytes_out: u64,
    pub buffer: usize,
    pub budget: u64,
    pub mem: MemSnapshot,
    pub pressure: Pressure,
    pub note: Option<String>,
}

pub type HugeProgressCb<'a> = Option<&'a dyn Fn(&HugeProgress)>;

#[derive(Debug, Clone, Serialize)]
pub struct HugeReport {
    pub input_size: u64,
    pub mode: String,
    pub completed: bool,
    pub bytes_scanned: u64,
    pub files_recovered: u64,
    pub bytes_recovered: u64,
    pub duplicates_skipped: u64,
    pub partial_files: u64,
    pub unsupported_entries: u64,
    pub unreadable_regions: u64,
    pub unreadable_bytes: u64,
    pub resumed_from: Option<u64>,
    pub memory_budget: u64,
    pub buffer_max: usize,
    pub buffer_min_used: usize,
    pub buffer_shrinks: u64,
    pub dedup_table_full: bool,
    pub notes: Vec<String>,
    pub scan_time_secs: f64,
    pub output_dir: String,
    pub manifest: String,
}

#[derive(Debug)]
pub enum HugeError {
    Io(io::Error),
    /// Output directory has an unfinished run and `resume` was not requested,
    /// or the checkpoint does not match the input.
    Checkpoint(String),
    /// Free disk space fell below the reserve. A checkpoint was saved.
    DiskLow { free: u64, reserve: u64 },
    /// Memory stayed critically low. A checkpoint was saved.
    MemoryPressure,
}

impl std::fmt::Display for HugeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Checkpoint(m) => write!(f, "{}", m),
            Self::DiskLow { free, reserve } => write!(
                f,
                "free disk space ({}) fell below the reserve ({}) — progress saved, free some space and resume",
                fmt_bytes(*free),
                fmt_bytes(*reserve)
            ),
            Self::MemoryPressure => write!(
                f,
                "system memory stayed critically low — progress saved, close other programs and resume"
            ),
        }
    }
}

impl std::error::Error for HugeError {}

impl From<io::Error> for HugeError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

type HResult<T> = Result<T, HugeError>;

// ──────────────────────────────────────────────────────────
//  Checkpoint
// ──────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
struct Checkpoint {
    version: u32,
    input_size: u64,
    mode: String,
    completed: bool,
    /// Carve / scan modes: input offset (decompressed offset for compressed
    /// inputs) where the next unfinished file starts.
    resume_offset: u64,
    /// Entry modes: number of archive entries fully handled.
    entries_done: u64,
    next_index: u64,
    manifest_len: u64,
    dedup_len: u64,
    files: u64,
    bytes_out: u64,
    dups: u64,
    partial: u64,
    unsupported: u64,
}

fn write_atomic(path: &Path, data: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)
}

// ──────────────────────────────────────────────────────────
//  Run state
// ──────────────────────────────────────────────────────────

struct Run<'a> {
    engine: &'a SalvageEngine,
    cfg: &'a HugeConfig,
    out: PathBuf,
    state_dir: PathBuf,
    input_size: u64,
    mode: String,
    mon: MemMonitor,
    gov: Governor,
    progress: HugeProgressCb<'a>,
    last_progress: Instant,
    last_disk_check: Instant,
    last_save: Instant,

    manifest: BufWriter<File>,
    dedup_file: BufWriter<File>,
    dedup: HashSet<u128>,
    dedup_cap: usize,
    dedup_overflow: bool,

    cp: Checkpoint,   // live counters
    cand: Checkpoint, // last safe restart point

    pressure: Pressure,
    buf_min_used: usize,
    unreadable_regions: u64,
    unreadable_bytes: u64,
    notes: Vec<String>,
    position: u64,
}

impl<'a> Run<'a> {
    fn spill_path(&self) -> PathBuf {
        self.state_dir.join("spill.tmp")
    }
    fn marker_path(&self) -> PathBuf {
        self.state_dir.join("current_entry")
    }

    fn note(&mut self, s: impl Into<String>) {
        let s = s.into();
        if self.notes.len() < 50 {
            self.notes.push(s);
        }
    }

    // ---- checkpointing ----

    /// Remember the current counters as a safe restart point.
    fn capture(&mut self, resume_offset: u64, entries_done: u64) {
        self.cand = Checkpoint {
            resume_offset,
            entries_done,
            ..self.cp.clone()
        };
    }

    fn save(&mut self, force: bool, completed: bool) -> io::Result<()> {
        if !force && self.last_save.elapsed() < Duration::from_secs(self.cfg.checkpoint_secs) {
            return Ok(());
        }
        self.manifest.flush()?;
        self.manifest.get_ref().sync_data()?;
        self.dedup_file.flush()?;
        self.dedup_file.get_ref().sync_data()?;
        let mut c = self.cand.clone();
        c.completed = completed;
        if completed {
            c = self.cp.clone();
            c.completed = true;
        }
        c.version = 1;
        c.input_size = self.input_size;
        c.mode = self.mode.clone();
        let json = serde_json::to_vec_pretty(&c).map_err(io::Error::other)?;
        write_atomic(&self.state_dir.join("checkpoint.json"), &json)?;
        self.last_save = Instant::now();
        Ok(())
    }

    // ---- guards ----

    fn check_disk(&mut self, force: bool) -> HResult<()> {
        if !force && self.last_disk_check.elapsed() < Duration::from_secs(2) {
            return Ok(());
        }
        self.last_disk_check = Instant::now();
        if let Some(free) = self.mon.free_disk_space(&self.out) {
            if free < self.cfg.disk_reserve {
                let _ = self.save(true, false);
                return Err(HugeError::DiskLow {
                    free,
                    reserve: self.cfg.disk_reserve,
                });
            }
        }
        Ok(())
    }

    fn emit_progress(&mut self, force: bool, note: Option<String>) {
        if !force && self.last_progress.elapsed() < Duration::from_millis(400) {
            return;
        }
        self.last_progress = Instant::now();
        if let Some(cb) = self.progress {
            let mem = self.mon.snapshot();
            cb(&HugeProgress {
                mode: self.mode.clone(),
                position: self.position,
                total: self.input_size,
                files: self.cp.files,
                bytes_out: self.cp.bytes_out,
                buffer: self.gov.cur_buf,
                budget: self.gov.budget,
                mem,
                pressure: self.pressure,
                note,
            });
        }
    }

    /// Wait for memory to recover; on timeout checkpoint and give up.
    fn handle_critical(&mut self) -> HResult<()> {
        let _ = self.save(true, false);
        let deadline = Instant::now() + Duration::from_secs(self.cfg.pause_wait_secs);
        self.emit_progress(
            true,
            Some("memory critically low — paused, waiting for it to recover".into()),
        );
        loop {
            std::thread::sleep(Duration::from_secs(2));
            let snap = self.mon.sample();
            self.pressure = self.gov.evaluate(&snap);
            if self.pressure != Pressure::Critical {
                return Ok(());
            }
            if Instant::now() > deadline {
                return Err(HugeError::MemoryPressure);
            }
            self.emit_progress(
                true,
                Some("memory critically low — paused, waiting for it to recover".into()),
            );
        }
    }

    // ---- output / manifest ----

    fn manifest_line(&mut self, v: serde_json::Value) -> io::Result<()> {
        let mut line = serde_json::to_vec(&v).map_err(io::Error::other)?;
        line.push(b'\n');
        self.manifest.write_all(&line)?;
        self.cp.manifest_len += line.len() as u64;
        Ok(())
    }

    fn seen_before(&mut self, key: u128) -> io::Result<bool> {
        if self.dedup.contains(&key) {
            return Ok(true);
        }
        if self.dedup.len() < self.dedup_cap {
            self.dedup.insert(key);
            self.dedup_file.write_all(&key.to_be_bytes())?;
            self.cp.dedup_len += 16;
        } else {
            self.dedup_overflow = true;
        }
        Ok(false)
    }

    fn carved_rel(&self, idx: u64, ext: &str) -> String {
        let ext = sanitize_ext(ext);
        format!("carved/{:04}/{:08}.{}", idx / 10_000, idx, ext)
    }

    fn emit_carved_mem(&mut self, offset: u64, ft: CarvedFileType, data: &[u8], conf: f64) -> io::Result<()> {
        let mut f = SalvagedFile {
            index: 0,
            name: String::new(),
            file_type: ft.label().to_string(),
            extension: ft.extension().to_string(),
            offset: offset as usize,
            size: data.len(),
            sha256: String::new(),
            confidence: conf,
            data: data.to_vec(),
        };
        crate::validate::validate_and_adjust(std::slice::from_mut(&mut f));
        let digest = Sha256::digest(&f.data);
        let key = u128::from_be_bytes(digest[..16].try_into().unwrap());
        if self.seen_before(key)? {
            self.cp.dups += 1;
            return Ok(());
        }
        let idx = self.cp.next_index;
        let rel = self.carved_rel(idx, ft.extension());
        let abs = self.out.join(&rel);
        if let Some(p) = abs.parent() {
            fs::create_dir_all(p)?;
        }
        fs::write(&abs, &f.data)?;
        self.cp.next_index += 1;
        self.cp.files += 1;
        self.cp.bytes_out += f.data.len() as u64;
        self.manifest_line(serde_json::json!({
            "i": idx, "path": rel, "type": f.file_type, "offset": offset,
            "size": f.data.len(), "sha256": hex(&digest), "confidence": f.confidence,
            "method": "carve", "status": "ok"
        }))
    }

    fn emit_carved_spilled(
        &mut self,
        offset: u64,
        ft: CarvedFileType,
        size: u64,
        digest: &[u8],
    ) -> io::Result<()> {
        let key = u128::from_be_bytes(digest[..16].try_into().unwrap());
        if self.seen_before(key)? {
            self.cp.dups += 1;
            let _ = fs::remove_file(self.spill_path());
            return Ok(());
        }
        let idx = self.cp.next_index;
        let rel = self.carved_rel(idx, ft.extension());
        let abs = self.out.join(&rel);
        if let Some(p) = abs.parent() {
            fs::create_dir_all(p)?;
        }
        fs::rename(self.spill_path(), &abs)?;
        self.cp.next_index += 1;
        self.cp.files += 1;
        self.cp.bytes_out += size;
        self.manifest_line(serde_json::json!({
            "i": idx, "path": rel, "type": ft.label(), "offset": offset,
            "size": size, "sha256": hex(digest), "confidence": 0.55,
            "method": "carve", "status": "ok"
        }))
    }

    // ---- carved-file lifecycle ----

    fn open_push(&mut self, o: &mut OpenFile, bytes: &[u8]) -> HResult<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        if o.spill.is_none() {
            let cap = self.gov.cur_buf.min(MEM_FILE_CAP);
            if o.mem.len() + bytes.len() <= cap {
                o.mem.extend_from_slice(bytes);
                o.written += bytes.len() as u64;
                return Ok(());
            }
            // Too big to hold — spill to disk, keep hashing incrementally.
            self.check_disk(true)?;
            let f = File::create(self.spill_path())?;
            let mut w = BufWriter::with_capacity(MIB as usize, f);
            let mut h = Sha256::new();
            w.write_all(&o.mem)?;
            h.update(&o.mem);
            let first = std::mem::take(&mut o.mem);
            // `written` currently counts the bytes in `first`; track_end expects
            // it to point at the start of the slice being scanned.
            o.written -= first.len() as u64;
            o.track_end(&first);
            o.written += first.len() as u64;
            o.spill = Some(w);
            o.hasher = Some(h);
        }
        o.track_end(bytes);
        o.spill.as_mut().unwrap().write_all(bytes)?;
        o.hasher.as_mut().unwrap().update(bytes);
        o.written += bytes.len() as u64;
        self.check_disk(false)?;
        Ok(())
    }

    fn finish_open(&mut self, mut o: OpenFile) -> HResult<()> {
        let size = o.written - o.start;
        if size < self.cfg.min_file_size as u64 {
            if o.spill.is_some() {
                let _ = fs::remove_file(self.spill_path());
            }
            return Ok(());
        }
        if let Some(mut w) = o.spill.take() {
            w.flush()?;
            drop(w);
            let mut digest = o.hasher.take().unwrap().finalize();
            let mut size = size;
            if let Some(t) = o.trim {
                if t > o.start && t - o.start < size {
                    // Cut the tail past the end marker and re-hash.
                    size = t - o.start;
                    let f = OpenOptions::new().write(true).open(self.spill_path())?;
                    f.set_len(size)?;
                    drop(f);
                    let mut h = Sha256::new();
                    let mut r = BufReader::with_capacity(MIB as usize, File::open(self.spill_path())?);
                    let mut chunk = vec![0u8; MIB as usize];
                    loop {
                        let n = r.read(&mut chunk)?;
                        if n == 0 {
                            break;
                        }
                        h.update(&chunk[..n]);
                    }
                    digest = h.finalize();
                }
            }
            self.emit_carved_spilled(o.start, o.ft, size, &digest)?;
        } else {
            let data = std::mem::take(&mut o.mem);
            let trimmed = self.engine.refine_file_end(&data, o.ft);
            if trimmed.len() < self.cfg.min_file_size {
                return Ok(());
            }
            let conf = if trimmed.len() < data.len() { 0.70 } else { 0.55 };
            self.emit_carved_mem(o.start, o.ft, trimmed, conf)?;
        }
        Ok(())
    }

    // ---- the streaming carver ----

    /// Carve signatures out of `reader`, whose first byte is at global offset
    /// `start`. `total` is the number of bytes in the stream if known.
    fn carve<R: Read>(&mut self, reader: &mut R, start: u64, total: Option<u64>) -> HResult<CarveEnd> {
        let mut base = start;
        let mut accepted_upto = start;
        let mut buf: Vec<u8> = vec![0u8; self.gov.cur_buf + 2 * KEEP];
        let mut len = 0usize;
        let mut open: Option<OpenFile> = None;
        let mut eof = false;
        let mut end = CarveEnd::Eof;

        loop {
            // 1. memory governor
            let snap = self.mon.snapshot();
            self.pressure = self.gov.evaluate(&snap);
            self.buf_min_used = self.buf_min_used.min(self.gov.cur_buf);
            if self.pressure == Pressure::Critical {
                self.handle_critical()?;
            }
            // Resize buffer only when it holds little data (right after a shift).
            let want = self.gov.cur_buf + 2 * KEEP;
            if buf.len() != want && len <= 2 * KEEP {
                let mut nb = vec![0u8; want];
                nb[..len].copy_from_slice(&buf[..len]);
                buf = nb;
            }

            // 2. fill
            while len < buf.len() && !eof {
                match reader.read(&mut buf[len..]) {
                    Ok(0) => eof = true,
                    Ok(n) => len += n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        eof = true;
                        end = CarveEnd::ReadError {
                            pos: base + len as u64,
                            msg: e.to_string(),
                        };
                    }
                }
            }

            // 3. find signatures
            let extra_tail = match total {
                Some(t) => t.saturating_sub(base + len as u64),
                None => 1u64 << 40,
            }
            .min(usize::MAX as u64 / 2) as usize;
            let hits = self.engine.collect_hits(&buf[..len], extra_tail);
            let limit = if eof { len } else { len.saturating_sub(KEEP) };

            for (a, ft) in hits {
                if a >= limit {
                    break;
                }
                let g = base + a as u64;
                if g < accepted_upto {
                    continue;
                }
                if let Some(o) = &open {
                    if g <= o.start + 4 {
                        continue;
                    }
                }
                if let Some(mut o) = open.take() {
                    let w = (o.written - base) as usize;
                    self.open_push(&mut o, &buf[w..a])?;
                    self.finish_open(o)?;
                }
                // Everything before `g` is finalised → safe restart point.
                self.capture(g, 0);
                open = Some(OpenFile::new(g, ft));
            }
            accepted_upto = base + limit as u64;
            self.position = base + len as u64;

            if eof {
                if let Some(mut o) = open.take() {
                    let w = (o.written - base) as usize;
                    self.open_push(&mut o, &buf[w..len])?;
                    self.finish_open(o)?;
                }
                break;
            }

            // 4. flush what is safely behind us and slide the window
            let keep_from = limit.saturating_sub(KEEP / 2);
            if let Some(o) = open.as_mut() {
                let w = (o.written - base) as usize;
                if keep_from > w {
                    let mut taken = std::mem::replace(o, OpenFile::new(0, CarvedFileType::Unknown));
                    self.open_push(&mut taken, &buf[w..keep_from])?;
                    *o = taken;
                }
            }
            buf.copy_within(keep_from..len, 0);
            len -= keep_from;
            base += keep_from as u64;

            self.save(false, false)?;
            self.emit_progress(false, None);
        }

        self.emit_progress(true, None);
        Ok(end)
    }

    // ---- entry-style output (zip / tar) ----

    fn copy_entry(&mut self, name: &str, reader: &mut dyn Read) -> HResult<EntryOut> {
        let rel = sanitize_path(name, self.cp.next_index);
        let mut abs = self.out.join("files").join(&rel);
        // Avoid clobbering a different entry with the same name.
        if abs.exists() {
            let stem = abs.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let ext = abs.extension().map(|s| format!(".{}", s.to_string_lossy())).unwrap_or_default();
            let mut n = 1;
            loop {
                let cand = abs.with_file_name(format!("{}~{}{}", stem, n, ext));
                if !cand.exists() {
                    abs = cand;
                    break;
                }
                n += 1;
            }
        }
        if let Some(p) = abs.parent() {
            fs::create_dir_all(p)?;
        }
        fs::write(self.marker_path(), abs.to_string_lossy().as_bytes())?;
        let file = File::create(&abs)?;
        let mut w = BufWriter::with_capacity(MIB as usize, file);
        let mut chunk = vec![0u8; MIB as usize];
        let mut crc = flate2::Crc::new();
        let mut bytes = 0u64;
        let mut err = None;
        let mut since_check = 0u64;
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    w.write_all(&chunk[..n])?;
                    crc.update(&chunk[..n]);
                    bytes += n as u64;
                    since_check += n as u64;
                    if since_check >= 128 * MIB {
                        since_check = 0;
                        w.flush()?;
                        self.check_disk(true)?;
                        self.save(false, false)?;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    err = Some(e.to_string());
                    break;
                }
            }
        }
        w.flush()?;
        drop(w);
        let rel_out = abs
            .strip_prefix(&self.out)
            .unwrap_or(&abs)
            .to_string_lossy()
            .replace('\\', "/");
        Ok(EntryOut {
            rel: rel_out,
            bytes,
            crc: crc.sum(),
            err,
        })
    }

    fn record_entry(
        &mut self,
        e: &EntryOut,
        orig: &str,
        method: &str,
        status: &str,
        confidence: f64,
    ) -> io::Result<()> {
        let idx = self.cp.next_index;
        self.cp.next_index += 1;
        self.cp.files += 1;
        self.cp.bytes_out += e.bytes;
        if status != "ok" {
            self.cp.partial += 1;
        }
        self.manifest_line(serde_json::json!({
            "i": idx, "path": e.rel, "name": orig, "size": e.bytes,
            "crc32": format!("{:08x}", e.crc), "confidence": confidence,
            "method": method, "status": status,
            "error": e.err
        }))?;
        let _ = fs::remove_file(self.marker_path());
        Ok(())
    }

    // ---- format drivers ----

    fn run_carve_file(&mut self, file: &mut File, start: u64) -> HResult<()> {
        let mut pos = start;
        let size = self.input_size;
        loop {
            file.seek(SeekFrom::Start(pos))?;
            let mut r = BufReader::with_capacity(MIB as usize, &mut *file);
            match self.carve(&mut r, pos, Some(size))? {
                CarveEnd::Eof => return Ok(()),
                CarveEnd::ReadError { pos: p, msg } => {
                    // Damaged media: skip a block and keep going.
                    const SKIP: u64 = 64 * 1024;
                    let next = (p + SKIP).min(size);
                    self.unreadable_regions += 1;
                    self.unreadable_bytes += next - p;
                    self.note(format!("unreadable region at {:#x}: {} — skipped {} KiB", p, msg, SKIP / 1024));
                    if next >= size {
                        return Ok(());
                    }
                    pos = next;
                }
            }
        }
    }

    fn run_compressed_carve<R: Read>(&mut self, reader: &mut R, start: u64) -> HResult<()> {
        if let CarveEnd::ReadError { pos, msg } = self.carve(reader, start, None)? {
            self.note(format!(
                "decompression stopped at decompressed offset {:#x}: {} — everything before it was recovered",
                pos, msg
            ));
        }
        Ok(())
    }

    fn run_tar<R: Read>(&mut self, reader: R, skip: u64) -> HResult<()> {
        let mut ar = tar::Archive::new(reader);
        let entries = match ar.entries() {
            Ok(e) => e,
            Err(e) => {
                self.note(format!("tar: cannot read entries: {}", e));
                return Ok(());
            }
        };
        for (i, ent) in entries.enumerate() {
            let i = i as u64;
            let mut ent = match ent {
                Ok(e) => e,
                Err(e) => {
                    self.note(format!("tar: stopped at entry {}: {} — earlier entries were recovered", i, e));
                    break;
                }
            };
            if i < skip {
                continue;
            }
            self.check_disk(false)?;
            let is_file = ent.header().entry_type().is_file();
            if is_file {
                let name = ent.path().map(|p| p.to_string_lossy().to_string()).unwrap_or_default();
                let expected = ent.header().size().unwrap_or(0);
                let out = self.copy_entry(&name, &mut ent)?;
                let (status, conf) = if out.err.is_some() || out.bytes < expected {
                    ("partial", 0.4)
                } else {
                    ("ok", 0.95)
                };
                self.record_entry(&out, &name, "tar", status, conf)?;
            }
            self.cp.entries_done = i + 1;
            self.capture(0, i + 1);
            self.save(false, false)?;
            self.emit_progress(false, None);
        }
        Ok(())
    }

    fn run_zip_cd(&mut self, file: File, skip: u64) -> HResult<bool> {
        let rd = BufReader::with_capacity(MIB as usize, file);
        let mut zip = match zip::ZipArchive::new(rd) {
            Ok(z) => z,
            Err(_) => return Ok(false),
        };
        let n = zip.len();
        for i in (skip as usize)..n {
            self.check_disk(false)?;
            self.position = i as u64;
            let mut ent = match zip.by_index(i) {
                Ok(e) => e,
                Err(e) => {
                    self.cp.unsupported += 1;
                    self.note(format!("zip entry {}: cannot open ({}) — skipped", i, e));
                    self.cp.entries_done = i as u64 + 1;
                    self.capture(0, i as u64 + 1);
                    continue;
                }
            };
            let name = ent.name().to_string();
            if ent.is_dir() {
                self.cp.entries_done = i as u64 + 1;
                self.capture(0, i as u64 + 1);
                continue;
            }
            let expected = ent.size();
            let out = self.copy_entry(&name, &mut ent)?;
            let (status, conf) = match (&out.err, out.bytes == expected) {
                (None, _) => ("ok", 0.95),
                (Some(_), true) => ("crc_mismatch", 0.7),
                (Some(_), false) => ("partial", 0.4),
            };
            drop(ent);
            self.record_entry(&out, &name, "zip-central-directory", status, conf)?;
            self.cp.entries_done = i as u64 + 1;
            self.capture(0, i as u64 + 1);
            self.save(false, false)?;
            self.emit_progress(false, None);
        }
        Ok(true)
    }

    fn run_zip_scan(&mut self, file: &mut File, start: u64) -> HResult<()> {
        let size = self.input_size;
        let mut pos = start;
        let mut skipped: std::collections::BTreeMap<String, u64> = Default::default();
        while let Some(sig) = find_local_header(file, pos, size)? {
            self.position = sig;
            self.capture(sig, 0);
            self.save(false, false)?;
            self.check_disk(false)?;
            self.emit_progress(false, None);

            file.seek(SeekFrom::Start(sig))?;
            let mut br = BufReader::with_capacity(256 * 1024, &mut *file);
            let mut hdr = [0u8; 30];
            if br.read_exact(&mut hdr).is_err() {
                break; // truncated inside a header
            }
            let flags = u16::from_le_bytes([hdr[6], hdr[7]]);
            let method = u16::from_le_bytes([hdr[8], hdr[9]]);
            let crc_hdr = u32::from_le_bytes([hdr[14], hdr[15], hdr[16], hdr[17]]);
            let csize = u32::from_le_bytes([hdr[18], hdr[19], hdr[20], hdr[21]]) as u64;
            let nlen = u16::from_le_bytes([hdr[26], hdr[27]]) as usize;
            let elen = u16::from_le_bytes([hdr[28], hdr[29]]) as usize;
            let mut name = vec![0u8; nlen];
            let mut extra = vec![0u8; elen];
            if br.read_exact(&mut name).is_err() || br.read_exact(&mut extra).is_err() {
                break;
            }
            let name = String::from_utf8_lossy(&name).to_string();
            let data_start = sig + 30 + nlen as u64 + elen as u64;

            if name.ends_with('/') {
                pos = data_start;
                continue;
            }
            if flags & 1 != 0 {
                self.cp.unsupported += 1;
                *skipped.entry("encrypted".to_string()).or_insert(0) += 1;
                pos = data_start;
                continue;
            }

            let mut cnt = Counting { inner: &mut br, n: 0 };
            // (crc from descriptor, bytes after data to skip, ended-by-header, hit-eof)
            let mut special: Option<(Option<u32>, u64, bool, bool)> = None;
            let outcome = match method {
                8 => {
                    let mut dec = flate2::bufread::DeflateDecoder::new(&mut cnt);
                    Some((self.copy_entry(&name, &mut dec)?, "zip-local-deflate"))
                }
                12 => {
                    let mut dec = bzip2::bufread::BzDecoder::new(&mut cnt);
                    Some((self.copy_entry(&name, &mut dec)?, "zip-local-bzip2"))
                }
                0 if flags & 8 == 0 && csize != 0xFFFF_FFFF => {
                    let mut lim = (&mut cnt).take(csize);
                    Some((self.copy_entry(&name, &mut lim)?, "zip-local-stored"))
                }
                0 => {
                    // Size unknown: stream until the data descriptor.
                    let mut dr = DescReader::new(&mut cnt);
                    let out = self.copy_entry(&name, &mut dr)?;
                    let (crc, dlen) = match dr.desc {
                        Some((c, l)) => (Some(c), l),
                        None => (None, 0),
                    };
                    special = Some((crc, dr.emitted + dlen, dr.ended_by_header, dr.eof && dr.desc.is_none() && !dr.ended_by_header));
                    Some((out, "zip-local-stored-stream"))
                }
                _ => None,
            };

            match outcome {
                None => {
                    self.cp.unsupported += 1;
                    *skipped.entry(format!("compression method {}", method)).or_insert(0) += 1;
                    pos = data_start;
                }
                Some((out, how)) => {
                    let (status, conf, next);
                    if let Some((dcrc, consumed, by_header, hit_eof)) = special {
                        next = data_start + consumed.max(1);
                        (status, conf) = if out.err.is_some() || hit_eof {
                            ("partial", 0.4)
                        } else if let Some(c) = dcrc {
                            if c == out.crc { ("ok", 0.95) } else { ("crc_mismatch", 0.6) }
                        } else if by_header {
                            ("unverified", 0.6)
                        } else {
                            ("partial", 0.4)
                        };
                    } else {
                        // CRC: from header, or from the trailing data descriptor.
                        let expect_crc = if flags & 8 == 0 {
                            Some(crc_hdr)
                        } else {
                            match cnt.fill_buf() {
                                Ok(b) if b.len() >= 8 && b[..4] == [0x50, 0x4B, 0x07, 0x08] => {
                                    Some(u32::from_le_bytes([b[4], b[5], b[6], b[7]]))
                                }
                                Ok(b) if b.len() >= 4 => Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
                                _ => None,
                            }
                        };
                        next = data_start + cnt.n.max(1);
                        (status, conf) = if out.err.is_some() {
                            ("partial", 0.4)
                        } else {
                            match expect_crc {
                                Some(c) if c == out.crc => ("ok", 0.95),
                                Some(_) => ("crc_mismatch", 0.6),
                                None => ("ok", 0.8),
                            }
                        };
                    }
                    self.record_entry(&out, &name, how, status, conf)?;
                    pos = next;
                }
            }
        }
        for (what, n) in skipped {
            self.note(format!("{} ZIP entries skipped ({}) — not supported in this mode", n, what));
        }
        self.position = size;
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────
//  Small helper types
// ──────────────────────────────────────────────────────────

enum CarveEnd {
    Eof,
    ReadError { pos: u64, msg: String },
}

struct EntryOut {
    rel: String,
    bytes: u64,
    crc: u32,
    err: Option<String>,
}

/// A file currently being carved.
struct OpenFile {
    start: u64,
    ft: CarvedFileType,
    /// Global offset up to which bytes have been moved into `mem` / `spill`.
    written: u64,
    mem: Vec<u8>,
    spill: Option<BufWriter<File>>,
    hasher: Option<Sha256>,
    /// Last bytes of the previous push (marker may straddle pushes).
    carry: Vec<u8>,
    /// Global offset where the file should end, per end-marker tracking.
    trim: Option<u64>,
}

impl OpenFile {
    fn new(start: u64, ft: CarvedFileType) -> Self {
        Self {
            start,
            ft,
            written: start,
            mem: Vec::new(),
            spill: None,
            hasher: None,
            carry: Vec::new(),
            trim: None,
        }
    }

    /// Streaming version of `refine_file_end` for spilled files:
    /// JPEG → first FFD9, PNG → first IEND(+CRC), PDF → last %%EOF.
    fn track_end(&mut self, bytes: &[u8]) {
        let (marker, extra, last): (&[u8], u64, bool) = match self.ft {
            CarvedFileType::Jpeg => (&[0xFF, 0xD9], 2, false),
            CarvedFileType::Png => (b"IEND", 8, false),
            CarvedFileType::Pdf => (b"%%EOF", 6, true),
            _ => return,
        };
        if !last && self.trim.is_some() {
            return;
        }
        let m = marker.len();
        let carry_start = self.written - self.carry.len() as u64;
        // Matches that start inside the carry and finish in `bytes`.
        let mut edge = self.carry.clone();
        edge.extend_from_slice(&bytes[..bytes.len().min(m - 1)]);
        let mut found: Vec<u64> = edge
            .windows(m)
            .enumerate()
            .filter(|(i, w)| *w == marker && *i < self.carry.len())
            .map(|(i, _)| carry_start + i as u64)
            .collect();
        let mut i = 0;
        while let Some(p) = bytes[i..].windows(m).position(|w| w == marker) {
            found.push(self.written + (i + p) as u64);
            i += p + 1;
            if !last {
                break;
            }
        }
        found.sort_unstable();
        let pick = if last { found.last() } else { found.first() };
        if let Some(&pos) = pick {
            self.trim = Some(pos + extra.max(m as u64));
        }
        let keep = bytes.len().min(m - 1);
        self.carry.extend_from_slice(&bytes[bytes.len() - keep..]);
        let cl = self.carry.len();
        if cl > m - 1 {
            self.carry.drain(..cl - (m - 1));
        }
    }
}

/// `BufRead` wrapper that counts consumed bytes.
struct Counting<'a, R: BufRead> {
    inner: &'a mut R,
    n: u64,
}
impl<R: BufRead> Read for Counting<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let k = self.inner.read(buf)?;
        self.n += k as u64;
        Ok(k)
    }
}
impl<R: BufRead> BufRead for Counting<'_, R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.inner.fill_buf()
    }
    fn consume(&mut self, amt: usize) {
        self.n += amt as u64;
        self.inner.consume(amt)
    }
}


/// Reads a *stored* ZIP entry whose size is not in the local header (streamed
/// archives such as OneDrive / browser "download as zip" use a trailing data
/// descriptor). Yields the data and stops at the first of:
///   * a data descriptor whose sizes equal the bytes seen so far (exact match,
///     so false positives are practically impossible; 32- and 64-bit forms),
///   * something that looks like the next local / central-directory header
///     (damaged or missing descriptor),
///   * end of input (truncated entry).
struct DescReader<'a, R: Read> {
    inner: &'a mut R,
    buf: Vec<u8>,
    start: usize,
    avail_end: usize,
    scan_pos: usize,
    eof: bool,
    done: bool,
    emitted: u64,
    /// (crc, descriptor length) when a valid descriptor ended the data.
    desc: Option<(u32, u64)>,
    ended_by_header: bool,
}

const DESC_LOOK: usize = 700;

impl<'a, R: Read> DescReader<'a, R> {
    fn new(inner: &'a mut R) -> Self {
        Self {
            inner,
            buf: Vec::new(),
            start: 0,
            avail_end: 0,
            scan_pos: 0,
            eof: false,
            done: false,
            emitted: 0,
            desc: None,
            ended_by_header: false,
        }
    }

    fn plausible_header(b: &[u8], i: usize) -> bool {
        let rest = &b[i..];
        match rest.get(2..4) {
            Some([3, 4]) => {
                if rest.len() < 30 {
                    return false;
                }
                let ver = u16::from_le_bytes([rest[4], rest[5]]);
                let method = u16::from_le_bytes([rest[8], rest[9]]);
                let nlen = u16::from_le_bytes([rest[26], rest[27]]) as usize;
                let elen = u16::from_le_bytes([rest[28], rest[29]]) as usize;
                if ver > 99 || !matches!(method, 0 | 8 | 9 | 12 | 14 | 93 | 95 | 98 | 99) {
                    return false;
                }
                if nlen == 0 || nlen > 512 || elen > 4096 || rest.len() < 30 + nlen {
                    return false;
                }
                match std::str::from_utf8(&rest[30..30 + nlen]) {
                    Ok(n) => !n.chars().any(|c| c.is_control()),
                    Err(_) => false,
                }
            }
            Some([1, 2]) => {
                rest.len() >= 46 && {
                    let nlen = u16::from_le_bytes([rest[28], rest[29]]) as usize;
                    nlen > 0 && nlen <= 512
                }
            }
            _ => false,
        }
    }

    fn fill_and_scan(&mut self) -> io::Result<()> {
        if self.start > 0 {
            self.buf.drain(..self.start);
            self.scan_pos -= self.start;
            self.avail_end -= self.start;
            self.start = 0;
        }
        let target = MIB as usize + DESC_LOOK;
        let mut len = self.buf.len();
        self.buf.resize(target, 0);
        while len < target && !self.eof {
            match self.inner.read(&mut self.buf[len..]) {
                Ok(0) => self.eof = true,
                Ok(n) => len += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => self.eof = true, // damaged media: treat as end of data
            }
        }
        self.buf.truncate(len);
        let limit = if self.eof { len } else { len.saturating_sub(DESC_LOOK) };
        let mut i = self.scan_pos;
        while i < limit {
            let Some(off) = self.buf[i..limit].iter().position(|&b| b == 0x50) else {
                break;
            };
            i += off;
            let b = &self.buf;
            if i + 4 <= len && b[i + 1] == 0x4B {
                if b[i + 2] == 7 && b[i + 3] == 8 {
                    let before = self.emitted + (i - self.start) as u64;
                    if i + 16 <= len {
                        let crc = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]);
                        let cs = u32::from_le_bytes([b[i + 8], b[i + 9], b[i + 10], b[i + 11]]) as u64;
                        let us = u32::from_le_bytes([b[i + 12], b[i + 13], b[i + 14], b[i + 15]]) as u64;
                        if cs == before && us == before {
                            self.desc = Some((crc, 16));
                            self.done = true;
                            self.avail_end = i;
                            return Ok(());
                        }
                    }
                    if i + 24 <= len {
                        let crc = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]);
                        let cs = u64::from_le_bytes(b[i + 8..i + 16].try_into().unwrap());
                        let us = u64::from_le_bytes(b[i + 16..i + 24].try_into().unwrap());
                        if cs == before && us == before {
                            self.desc = Some((crc, 24));
                            self.done = true;
                            self.avail_end = i;
                            return Ok(());
                        }
                    }
                } else if Self::plausible_header(b, i) {
                    self.ended_by_header = true;
                    self.done = true;
                    // Drop a damaged descriptor sitting right before the header.
                    let mut end = i;
                    for dl in [24usize, 16] {
                        if i >= self.start + dl && b[i - dl..i - dl + 4] == [0x50, 0x4B, 0x07, 0x08] {
                            end = i - dl;
                            break;
                        }
                    }
                    self.avail_end = end;
                    return Ok(());
                }
            }
            i += 1;
        }
        self.avail_end = limit;
        self.scan_pos = limit;
        if self.eof {
            self.done = true;
        }
        Ok(())
    }
}

impl<R: Read> Read for DescReader<'_, R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.start < self.avail_end {
                let n = out.len().min(self.avail_end - self.start);
                out[..n].copy_from_slice(&self.buf[self.start..self.start + n]);
                self.start += n;
                self.emitted += n as u64;
                return Ok(n);
            }
            if self.done {
                return Ok(0);
            }
            self.fill_and_scan()?;
        }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

/// Scan forward from `pos` for a ZIP local file header (`PK\x03\x04`).
fn find_local_header(file: &mut File, mut pos: u64, size: u64) -> io::Result<Option<u64>> {
    const SIG: [u8; 4] = [0x50, 0x4B, 0x03, 0x04];
    let mut chunk = vec![0u8; MIB as usize];
    while pos + 4 <= size {
        file.seek(SeekFrom::Start(pos))?;
        let mut n = 0;
        while n < chunk.len() {
            match file.read(&mut chunk[n..]) {
                Ok(0) => break,
                Ok(k) => n += k,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        if n < 4 {
            return Ok(None);
        }
        if let Some(i) = chunk[..n].windows(4).position(|w| w == SIG) {
            return Ok(Some(pos + i as u64));
        }
        pos += (n - 3) as u64;
    }
    Ok(None)
}

fn sanitize_ext(ext: &str) -> String {
    let e: String = ext.chars().filter(|c| c.is_ascii_alphanumeric()).take(8).collect();
    if e.is_empty() {
        "bin".into()
    } else {
        e
    }
}

/// Turn an archive entry name into a safe relative path (no traversal, no
/// absolute paths, no characters Windows rejects, no reserved device names).
pub fn sanitize_path(name: &str, fallback_idx: u64) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in name.split(['/', '\\']) {
        let comp = comp.trim();
        if comp.is_empty() || comp == "." || comp == ".." {
            continue;
        }
        // Drive letters like "C:" would otherwise become absolute on Windows.
        let mut c: String = comp
            .chars()
            .map(|ch| match ch {
                '<' | '>' | ':' | '"' | '|' | '?' | '*' => '_',
                ch if (ch as u32) < 32 => '_',
                ch => ch,
            })
            .collect();
        c = c.trim_end_matches(['.', ' ']).to_string();
        if c.is_empty() {
            continue;
        }
        if c.len() > 200 {
            let mut cut = 200;
            while !c.is_char_boundary(cut) {
                cut -= 1;
            }
            c.truncate(cut);
        }
        let stem = c.split('.').next().unwrap_or("").to_ascii_uppercase();
        let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
            || (stem.len() == 4
                && (stem.starts_with("COM") || stem.starts_with("LPT"))
                && stem.as_bytes()[3].is_ascii_digit());
        if reserved {
            c.insert(0, '_');
        }
        out.push(c);
    }
    if out.as_os_str().is_empty() {
        out.push(format!("unnamed_{}", fallback_idx));
    }
    out
}

// ──────────────────────────────────────────────────────────
//  Entry point
// ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    ZipCd,
    ZipScan,
    Tar,
    Gzip,
    Bzip2,
    Xz,
    Carve,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::ZipCd => "zip-central-directory",
            Kind::ZipScan => "zip-local-scan",
            Kind::Tar => "tar",
            Kind::Gzip => "gzip-stream",
            Kind::Bzip2 => "bzip2-stream",
            Kind::Xz => "xz-stream",
            Kind::Carve => "carve",
        }
    }
    fn parse(s: &str) -> Option<Self> {
        [Kind::ZipCd, Kind::ZipScan, Kind::Tar, Kind::Gzip, Kind::Bzip2, Kind::Xz, Kind::Carve]
            .into_iter()
            .find(|k| k.name() == s)
    }
}

fn sniff(file: &mut File, cfg: &HugeConfig) -> io::Result<Kind> {
    let mut head = [0u8; 512];
    file.seek(SeekFrom::Start(0))?;
    let mut n = 0;
    while n < head.len() {
        match file.read(&mut head[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    let h = &head[..n];
    if cfg.force_carve {
        return Ok(Kind::Carve);
    }
    if h.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
        // Central directory readable?
        let ok = {
            let f = file.try_clone()?;
            zip::ZipArchive::new(BufReader::new(f)).is_ok()
        };
        return Ok(if ok { Kind::ZipCd } else { Kind::ZipScan });
    }
    if h.starts_with(&[0x1F, 0x8B]) {
        return Ok(Kind::Gzip);
    }
    if h.starts_with(b"BZh") {
        return Ok(Kind::Bzip2);
    }
    if h.starts_with(&[0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00]) {
        return Ok(Kind::Xz);
    }
    if h.len() > 262 && &h[257..262] == b"ustar" {
        return Ok(Kind::Tar);
    }
    Ok(Kind::Carve)
}

/// Recover files from `input` into `out` using bounded memory.
pub fn huge_salvage(
    input: &Path,
    out: &Path,
    engine: &SalvageEngine,
    cfg: &HugeConfig,
    progress: HugeProgressCb<'_>,
) -> HResult<HugeReport> {
    let started = Instant::now();
    let mut file = File::open(input)?;
    let input_size = file.metadata()?.len();

    fs::create_dir_all(out)?;
    let state_dir = out.join(STATE_DIR);
    fs::create_dir_all(&state_dir)?;
    let cp_path = state_dir.join("checkpoint.json");

    // ---- checkpoint handling ----
    let mut resume_cp: Option<Checkpoint> = None;
    if cp_path.exists() {
        let txt = fs::read_to_string(&cp_path)?;
        let cp: Checkpoint = serde_json::from_str(&txt)
            .map_err(|e| HugeError::Checkpoint(format!("unreadable checkpoint: {}", e)))?;
        if !cfg.resume {
            return Err(HugeError::Checkpoint(format!(
                "'{}' already contains a {} run. Use --resume to continue it, or pick an empty output directory.",
                out.display(),
                if cp.completed { "finished" } else { "previous" }
            )));
        }
        if cp.input_size != input_size {
            return Err(HugeError::Checkpoint(format!(
                "checkpoint was made for a {}-byte input, this file is {} bytes",
                cp.input_size, input_size
            )));
        }
        if cp.completed {
            return Err(HugeError::Checkpoint("that run already finished — nothing to resume".into()));
        }
        resume_cp = Some(cp);
    }

    let kind = match &resume_cp {
        Some(cp) => Kind::parse(&cp.mode)
            .ok_or_else(|| HugeError::Checkpoint(format!("unknown mode '{}' in checkpoint", cp.mode)))?,
        None => sniff(&mut file, cfg)?,
    };

    // ---- manifest + dedup files (truncate to checkpoint on resume) ----
    let manifest_path = out.join("manifest.jsonl");
    let dedup_path = state_dir.join("dedup.bin");
    let mut dedup_set = HashSet::new();

    let mem_snapshot = {
        let mut m = MemMonitor::new();
        m.sample()
    };
    let gov = Governor::new(&mem_snapshot, cfg.mem_limit);
    let dedup_cap = ((gov.budget / 8) / 24).max(100_000) as usize;

    if let Some(cp) = &resume_cp {
        // Delete files recorded after the checkpoint, then cut the manifest.
        if manifest_path.exists() {
            let mut f = File::open(&manifest_path)?;
            f.seek(SeekFrom::Start(cp.manifest_len))?;
            let mut tail = String::new();
            f.read_to_string(&mut tail).ok();
            for line in tail.lines() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(p) = v.get("path").and_then(|p| p.as_str()) {
                        let _ = fs::remove_file(out.join(p));
                    }
                }
            }
            OpenOptions::new().write(true).open(&manifest_path)?.set_len(cp.manifest_len)?;
        }
        if let Ok(s) = fs::read_to_string(state_dir.join("current_entry")) {
            let _ = fs::remove_file(s.trim());
            let _ = fs::remove_file(state_dir.join("current_entry"));
        }
        let _ = fs::remove_file(state_dir.join("spill.tmp"));
        if dedup_path.exists() {
            OpenOptions::new().write(true).open(&dedup_path)?.set_len(cp.dedup_len)?;
            let raw = fs::read(&dedup_path)?;
            for c in raw.as_chunks::<16>().0.iter().take(dedup_cap) {
                dedup_set.insert(u128::from_be_bytes(*c));
            }
        }
    }

    let manifest = BufWriter::new(OpenOptions::new().create(true).append(true).open(&manifest_path)?);
    let dedup_file = BufWriter::new(OpenOptions::new().create(true).append(true).open(&dedup_path)?);

    let cp0 = resume_cp.clone().unwrap_or_default();
    let mut run = Run {
        engine,
        cfg,
        out: out.to_path_buf(),
        state_dir,
        input_size,
        mode: kind.name().to_string(),
        mon: MemMonitor::new(),
        buf_min_used: gov.cur_buf,
        gov,
        progress,
        last_progress: Instant::now() - Duration::from_secs(10),
        last_disk_check: Instant::now() - Duration::from_secs(10),
        last_save: Instant::now(),
        manifest,
        dedup_file,
        dedup: dedup_set,
        dedup_cap,
        dedup_overflow: false,
        cp: cp0.clone(),
        cand: cp0.clone(),
        pressure: Pressure::Ok,
        unreadable_regions: 0,
        unreadable_bytes: 0,
        notes: Vec::new(),
        position: cp0.resume_offset,
    };
    run.cp.version = 1;
    run.capture(cp0.resume_offset, cp0.entries_done);
    let resumed_from = resume_cp.as_ref().map(|c| if c.entries_done > 0 { c.entries_done } else { c.resume_offset });
    run.check_disk(true)?;
    run.save(true, false)?; // checkpoint exists from the very start

    let mut mode_name = kind.name().to_string();
    let skip_off = cp0.resume_offset;
    let skip_entries = cp0.entries_done;

    match kind {
        Kind::ZipCd => {
            let f = file.try_clone()?;
            if !run.run_zip_cd(f, skip_entries)? {
                // Directory vanished between sniff and open — fall back.
                mode_name = "zip-local-scan".into();
                run.mode = mode_name.clone();
                run.run_zip_scan(&mut file, skip_off)?;
            }
        }
        Kind::ZipScan => {
            run.note("ZIP central directory is damaged or missing — scanning local file headers (partial / truncated entries are kept)");
            run.run_zip_scan(&mut file, skip_off)?;
        }
        Kind::Tar => {
            file.seek(SeekFrom::Start(0))?;
            run.run_tar(BufReader::with_capacity(MIB as usize, &mut file), skip_entries)?;
        }
        Kind::Carve => {
            run.run_carve_file(&mut file, skip_off)?;
        }
        Kind::Gzip | Kind::Bzip2 | Kind::Xz => {
            file.seek(SeekFrom::Start(0))?;
            let br = BufReader::with_capacity(MIB as usize, &mut file);
            let mut dec: Box<dyn Read> = match kind {
                Kind::Gzip => Box::new(flate2::read::MultiGzDecoder::new(br)),
                Kind::Bzip2 => Box::new(bzip2::read::MultiBzDecoder::new(br)),
                _ => Box::new(xz2::read::XzDecoder::new_multi_decoder(br)),
            };
            // Peek the first decompressed block to see whether it is a tar.
            let mut head = Vec::with_capacity(512);
            let _ = (&mut dec).take(512).read_to_end(&mut head);
            let is_tar = head.len() > 262 && &head[257..262] == b"ustar";
            let mut stream = io::Cursor::new(head).chain(dec);
            if is_tar {
                run.mode = format!("{}+tar", kind.name());
                mode_name = run.mode.clone();
                run.run_tar(stream, skip_entries)?;
            } else {
                run.mode = format!("{}+carve", kind.name());
                mode_name = run.mode.clone();
                // Resume: re-decompress and discard what was already carved.
                if skip_off > 0 {
                    io::copy(&mut (&mut stream).take(skip_off), &mut io::sink())?;
                }
                run.run_compressed_carve(&mut stream, skip_off)?;
            }
        }
    }

    run.position = input_size;
    run.save(true, true)?;
    run.emit_progress(true, Some("done".into()));

    let notes = std::mem::take(&mut run.notes);
    Ok(HugeReport {
        input_size,
        mode: mode_name,
        completed: true,
        bytes_scanned: input_size,
        files_recovered: run.cp.files,
        bytes_recovered: run.cp.bytes_out,
        duplicates_skipped: run.cp.dups,
        partial_files: run.cp.partial,
        unsupported_entries: run.cp.unsupported,
        unreadable_regions: run.unreadable_regions,
        unreadable_bytes: run.unreadable_bytes,
        resumed_from,
        memory_budget: run.gov.budget,
        buffer_max: run.gov.max_buf,
        buffer_min_used: run.buf_min_used,
        buffer_shrinks: run.gov.shrink_events,
        dedup_table_full: run.dedup_overflow,
        notes,
        scan_time_secs: (started.elapsed().as_secs_f64() * 1000.0).round() / 1000.0,
        output_dir: out.display().to_string(),
        manifest: manifest_path.display().to_string(),
    })
}

// ══════════════════════════════════════════════════════════════
//  TESTS
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_blocks_traversal() {
        assert_eq!(sanitize_path("../../etc/passwd", 0), PathBuf::from("etc/passwd"));
        assert_eq!(sanitize_path("/abs/path.txt", 0), PathBuf::from("abs/path.txt"));
        assert_eq!(sanitize_path("C:\\Windows\\x.dll", 0), PathBuf::from("C_/Windows/x.dll"));
        assert_eq!(sanitize_path("a/./b/../c", 0), PathBuf::from("a/b/c"));
    }

    #[test]
    fn sanitize_windows_rules() {
        assert_eq!(sanitize_path("dir/con.txt", 0), PathBuf::from("dir/_con.txt"));
        assert_eq!(sanitize_path("a?b*c.txt", 0), PathBuf::from("a_b_c.txt"));
        assert_eq!(sanitize_path("trail. ", 0), PathBuf::from("trail"));
        assert_eq!(sanitize_path("", 7), PathBuf::from("unnamed_7"));
        assert_eq!(sanitize_path("///", 3), PathBuf::from("unnamed_3"));
    }

    #[test]
    fn local_header_finder_crosses_chunk_boundary() {
        let dir = std::env::temp_dir().join(format!("helix_lh_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.bin");
        let mut data = vec![0u8; MIB as usize * 2 + 100];
        let at = MIB as usize - 2; // straddles the 1 MiB read boundary
        data[at..at + 4].copy_from_slice(&[0x50, 0x4B, 0x03, 0x04]);
        fs::write(&p, &data).unwrap();
        let mut f = File::open(&p).unwrap();
        let r = find_local_header(&mut f, 0, data.len() as u64).unwrap();
        assert_eq!(r, Some(at as u64));
        fs::remove_dir_all(&dir).ok();
    }
}
