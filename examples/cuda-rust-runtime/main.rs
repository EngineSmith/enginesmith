use std::ffi::{c_char, c_void, CStr};
use std::ptr::NonNull;
use std::time::Instant;

unsafe extern "C" {
    fn runtime_create(count: usize, output: *mut *mut c_void) -> i32;
    fn runtime_destroy(state: *mut c_void);
    fn runtime_run(state: *mut c_void, repetitions: u32, graph: i32, elapsed_ms: *mut f32) -> i32;
    fn runtime_read(state: *mut c_void, output: *mut f32, count: usize) -> i32;
    fn runtime_error(status: i32) -> *const c_char;
}

fn check(status: i32) -> Result<(), String> {
    if status == 0 { return Ok(()); }
    let message = unsafe { CStr::from_ptr(runtime_error(status)) }.to_string_lossy();
    Err(format!("CUDA error {status}: {message}"))
}

struct Runtime { state: NonNull<c_void>, count: usize }
impl Runtime {
    fn new(count: usize) -> Result<Self, String> {
        let mut raw = std::ptr::null_mut();
        check(unsafe { runtime_create(count, &mut raw) })?;
        let state = NonNull::new(raw).ok_or("runtime returned a null state")?;
        Ok(Self { state, count })
    }
    fn run(&mut self, repetitions: u32, graph: bool) -> Result<(f32, f64), String> {
        let mut elapsed_ms = 0.0;
        let wall = Instant::now();
        check(unsafe { runtime_run(self.state.as_ptr(), repetitions, i32::from(graph), &mut elapsed_ms) })?;
        let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
        let mut output = vec![0.0; self.count];
        check(unsafe { runtime_read(self.state.as_ptr(), output.as_mut_ptr(), output.len()) })?;
        if output.iter().any(|&v| v != repetitions as f32) {
            return Err("output did not equal the exact expected increment count".into());
        }
        Ok((elapsed_ms, wall_ms))
    }
}
impl Drop for Runtime {
    fn drop(&mut self) { unsafe { runtime_destroy(self.state.as_ptr()) }; }
}

fn main() -> Result<(), String> {
    let repetitions: u32 = std::env::args().nth(1).unwrap_or_else(|| "1000".into())
        .parse().map_err(|_| "expected integer repetitions")?;
    if repetitions == 0 || repetitions > (1 << 24) {
        return Err("repetitions must be in 1..=16777216 for exact float increments".into());
    }
    let mut runtime = Runtime::new(4096)?;
    runtime.run(10, false)?;
    runtime.run(10, true)?;
    let direct = runtime.run(repetitions, false)?;
    let graph = runtime.run(repetitions, true)?;
    println!("{{\"correct\":true,\"elements\":4096,\"repetitions\":{repetitions},\"direct_event_ms\":{},\"direct_wall_ms\":{},\"graph_event_ms\":{},\"graph_wall_ms\":{}}}",
             direct.0, direct.1, graph.0, graph.1);
    Ok(())
}
