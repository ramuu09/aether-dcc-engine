use std::sync::{Mutex, MutexGuard};

pub fn cpu_lock() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Kept so callers compile. The process is left at `THREAD_PRIORITY_NORMAL`.
/// Raising `HIGH_PRIORITY_CLASS` starved the desktop window manager.
pub fn elevate() {}
