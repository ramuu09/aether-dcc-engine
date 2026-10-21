//! Win32 dashboard for AetherDCC Studio v1.0.

use std::ptr::{null, null_mut};

use aether_dcc::{argmax, corrupt, glyph_f32, ratio, weight_census, Alu, BitnetSample, GpuEngine, GPU_ADD, GPU_XOR};

const TITLE: &str = "AetherDCC Studio v1.0 â€” Non-Binary Silicon Compute & 1.58-Bit AI Product";
const CLASS_NAME: &str = "AetherDccStudioWindow";
const KEY: [u8; 16] = *b"K3y-UHD630-DCC!!";
const SALT: [u8; 16] = *b"Salt-Pedestal-4!";

struct Host {
    gpu: GpuEngine,
    pixels: [f32; 64],
    last: BitnetSample,
    ghost: String,
    ghost_us: f64,
    paradox: String,
    sparsity: String,
    noise_state: u64,
    stroke: Option<f32>,
    title_font: *mut core::ffi::c_void,
    body_font: *mut core::ffi::c_void,
    small_font: *mut core::ffi::c_void,
}

pub fn launch() {
    unsafe { SetProcessDPIAware() };
    if unsafe { GetSystemMetrics(80) } <= 0 {
        eprintln!("No display. Run: cargo run --release -- --product-benchmark");
        std::process::exit(2);
    }
    let paradox = paradox_line();
    let mut gpu = GpuEngine::new();
    let pixels = [-1.0f32; 64];
    let last = gpu.bitnet(&pixels);
    let (ghost, ghost_us) = ghost_text(&mut gpu, &pixels);
    let (zeros, total) = weight_census();
    let host = Box::new(Host {
        gpu,
        pixels,
        last,
        ghost,
        ghost_us,
        paradox,
        sparsity: format!("{zeros}/{total} weights are 0 ({:.1}%)", zeros as f64 / total as f64 * 100.0),
        noise_state: 0xC0FFEE,
        stroke: None,
        title_font: make_font(-22, 600),
        body_font: make_font(-16, 400),
        small_font: make_font(-14, 400),
    });
    let class_name = leak_wide(CLASS_NAME);
    let klass = WndClassExW {
        cb_size: std::mem::size_of::<WndClassExW>() as u32,
        style: 0x0001 | 0x0002,
        wnd_proc,
        cb_cls_extra: 0,
        cb_wnd_extra: 0,
        instance: unsafe { GetModuleHandleW(null()) },
        icon: null_mut(),
        cursor: unsafe { LoadCursorW(null_mut(), 32512usize as *const u16) },
        background: null_mut(),
        menu_name: null(),
        class_name: class_name.as_ptr(),
        icon_sm: null_mut(),
    };
    if unsafe { RegisterClassExW(&klass) } == 0 {
        panic!("RegisterClassExW failed ({})", unsafe { GetLastError() });
    }
    let (width, height) = window_size(1240, 780);
    let title = wide(TITLE);
    let hwnd = unsafe {
        CreateWindowExW(
            0,
            class_name.as_ptr(),
            title.as_ptr(),
            0x00CF_0000,
            0x8000_0000u32 as i32,
            0x8000_0000u32 as i32,
            width,
            height,
            null_mut(),
            null_mut(),
            klass.instance,
            null_mut(),
        )
    };
    if hwnd.is_null() {
        panic!("CreateWindowExW failed ({})", unsafe { GetLastError() });
    }
    let ptr = Box::into_raw(host);
    unsafe { SetWindowLongPtrW(hwnd, -21, ptr as isize) };
    unsafe {
        ShowWindow(hwnd, 1);
        UpdateWindow(hwnd);
    }
    println!("{TITLE}");
    println!("Hotkeys: 0-9 load a digit, N noise, G GhostVault, C clear. Left paint +1, right paint -1.");
    let mut msg: Msg = unsafe { std::mem::zeroed() };
    // GetMessageW blocks. There is no WM_TIMER and no idle contention loop.
    loop {
        let code = unsafe { GetMessageW(&mut msg, null_mut(), 0, 0) };
        if code == 0 {
            break;
        }
        if code < 0 {
            panic!("GetMessageW failed ({})", unsafe { GetLastError() });
        }
        unsafe {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

fn paradox_line() -> String {
    let mut alu = Alu::new();
    let line = match alu.probe([1, -1, 0], 0) {
        Ok(wave) => format!(
            "3-node ring  counts {}/{}  Y {:+.3}  {}",
            wave.t_pos,
            wave.t_neg,
            wave.y,
            if wave.y == 0.0 && wave.y.is_finite() { "BOTH / PARADOX" } else { "UNRESOLVED" }
        ),
        Err(error) => format!("paradox probe failed: {error}"),
    };
    aether_dcc::ThermalGovernor::after_contention(u32::try_from(alu.jobs()).unwrap_or(u32::MAX));
    line
}

fn ghost_text(gpu: &mut GpuEngine, pixels: &[f32; 64]) -> (String, f64) {
    let mut block = [0u8; 16];
    for index in 0..16 {
        let ink = u8::from(pixels[index] > 0.0);
        block[index] = (ink << 7) ^ 0x40 ^ index as u8;
    }
    let (xored, first) = gpu.simd_timed(GPU_XOR, &block, &KEY);
    let (cipher, second) = gpu.simd_timed(GPU_ADD, &xored, &SALT);
    let hex = cipher.iter().map(|byte| format!("{byte:02X}")).collect::<Vec<_>>().join(" ");
    (hex, first + second)
}

fn refresh(host: &mut Host) {
    host.last = host.gpu.bitnet(&host.pixels);
    let (ghost, ghost_us) = ghost_text(&mut host.gpu, &host.pixels);
    host.ghost = ghost;
    host.ghost_us = ghost_us;
}

fn paint_at(host: &mut Host, x: i32, y: i32, value: f32) -> bool {
    let Some(index) = cell_at(x, y) else {
        return false;
    };
    if host.pixels[index] == value {
        return false;
    }
    host.pixels[index] = value;
    refresh(host);
    true
}

fn cell_at(x: i32, y: i32) -> Option<usize> {
    let origin_x = 36;
    let origin_y = 96;
    let cell = 48;
    let gap = 6;
    let stride = cell + gap;
    if x < origin_x || y < origin_y {
        return None;
    }
    let col = (x - origin_x) / stride;
    let row = (y - origin_y) / stride;
    if !(0..8).contains(&col) || !(0..8).contains(&row) {
        return None;
    }
    let local_x = (x - origin_x) % stride;
    let local_y = (y - origin_y) % stride;
    if local_x >= cell || local_y >= cell {
        return None;
    }
    Some((row * 8 + col) as usize)
}

unsafe extern "system" fn wnd_proc(hwnd: *mut core::ffi::c_void, msg: u32, wparam: usize, lparam: isize) -> isize {
    if msg == 0x0024 {
        let info = lparam as *mut MinMaxInfo;
        if !info.is_null() {
            (*info).min_track_x = 1100;
            (*info).min_track_y = 760;
        }
        return 0;
    }
    if msg == 0x0014 {
        return 1;
    }
    let host_ptr = GetWindowLongPtrW(hwnd, -21) as *mut Host;
    if host_ptr.is_null() {
        if msg == 0x000F {
            let mut paint: Paint = std::mem::zeroed();
            BeginPaint(hwnd, &mut paint);
            EndPaint(hwnd, &paint);
        }
        return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
    match msg {
        0x000F => {
            paint_window(hwnd, &*host_ptr);
            0
        }
        0x0201 | 0x0204 => {
            SetCapture(hwnd);
            let value = if msg == 0x0201 { 1.0 } else { -1.0 };
            let (x, y) = coords(lparam);
            let redraw = {
                let host = &mut *host_ptr;
                host.stroke = Some(value);
                paint_at(host, x, y, value)
            };
            if redraw {
                InvalidateRect(hwnd, null(), 0);
            }
            0
        }
        0x0202 | 0x0205 => {
            (*host_ptr).stroke = None;
            ReleaseCapture();
            0
        }
        0x0200 => {
            let buttons = wparam & 0x0003;
            if buttons == 0 {
                (*host_ptr).stroke = None;
                return 0;
            }
            let value = if buttons & 0x0001 != 0 { 1.0 } else { -1.0 };
            let (x, y) = coords(lparam);
            if paint_at(&mut *host_ptr, x, y, value) {
                InvalidateRect(hwnd, null(), 0);
            }
            0
        }
        0x0100 => {
            on_key(&mut *host_ptr, wparam);
            InvalidateRect(hwnd, null(), 0);
            0
        }
        0x0002 => {
            let host = Box::from_raw(host_ptr);
            DeleteObject(host.title_font);
            DeleteObject(host.body_font);
            DeleteObject(host.small_font);
            drop(host);
            PostQuitMessage(0);
            0
        }
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

fn on_key(host: &mut Host, key: usize) {
    match key {
        0x30..=0x39 => {
            let digit = key - 0x30;
            host.pixels = glyph_f32(digit);
            refresh(host);
        }
        0x43 => {
            host.pixels = [-1.0; 64];
            refresh(host);
        }
        0x4E => {
            host.pixels = corrupt(&host.pixels, 8, &mut host.noise_state);
            refresh(host);
        }
        0x47 => refresh(host),
        _ => {}
    }
}

fn paint_window(hwnd: *mut core::ffi::c_void, host: &Host) {
    let mut paint: Paint = unsafe { std::mem::zeroed() };
    let hdc = unsafe { BeginPaint(hwnd, &mut paint) };
    let mut client = Rect { left: 0, top: 0, right: 0, bottom: 0 };
    unsafe { GetClientRect(hwnd, &mut client) };
    let width = client.right;
    let height = client.bottom;
    let buffer = unsafe { CreateCompatibleDC(hdc) };
    let bitmap = unsafe { CreateCompatibleBitmap(hdc, width, height) };
    let old = unsafe { SelectObject(buffer, bitmap) };
    draw(buffer, host, width, height);
    unsafe {
        BitBlt(hdc, 0, 0, width, height, buffer, 0, 0, 0x00CC_0020);
        SelectObject(buffer, old);
        DeleteObject(bitmap);
        DeleteDC(buffer);
        EndPaint(hwnd, &paint);
    }
}

fn draw(hdc: *mut core::ffi::c_void, host: &Host, width: i32, height: i32) {
    fill(hdc, rect(0, 0, width, height), rgb(14, 16, 22));
    text(hdc, TITLE, rect(28, 18, width - 28, 52), host.title_font, rgb(226, 230, 238), 0);
    text(
        hdc,
        "8Ã—8 canvas    left +1    right âˆ’1    0â€“9 code glyph    N noise    G GhostVault    C clear",
        rect(28, 54, width - 28, 82),
        host.small_font,
        rgb(148, 156, 174),
        0,
    );
    let predicted = argmax(host.last.scores);
    let label = if predicted < 0 { "BLANK / NO_SIGNAL".to_string() } else { format!("DIGIT {predicted}") };
    text(hdc, &label, rect(36, 560, 500, 592), host.body_font, rgb(240, 176, 64), 0);
    for row in 0..8 {
        for col in 0..8 {
            let index = row * 8 + col;
            let left = 36 + (col as i32) * 54;
            let top = 96 + (row as i32) * 54;
            let ink = host.pixels[index] > 0.0;
            fill(
                hdc,
                rect(left, top, left + 48, top + 48),
                if ink { rgb(64, 206, 204) } else { rgb(32, 38, 54) },
            );
        }
    }
    let class_left = 520;
    text(hdc, "10-class confidence", rect(class_left, 96, class_left + 280, 122), host.body_font, rgb(226, 230, 238), 0);
    for class in 0..10 {
        let top = 132 + (class as i32) * 42;
        let score = host.last.scores[class];
        let y = ratio(host.last.class_pos[class], host.last.class_neg[class]);
        text(
            hdc,
            &format!("{class}"),
            rect(class_left, top, class_left + 24, top + 28),
            host.body_font,
            rgb(226, 230, 238),
            0,
        );
        fill(hdc, rect(class_left + 32, top + 6, class_left + 240, top + 26), rgb(10, 12, 18));
        let span = ((score.max(0) as i32).min(64) * 208) / 64;
        if span > 0 {
            fill(
                hdc,
                rect(class_left + 32, top + 6, class_left + 32 + span, top + 26),
                if predicted == class as i32 { rgb(64, 196, 120) } else { rgb(88, 98, 124) },
            );
        }
        text(
            hdc,
            &format!("{score:+4}  Y {y:+.3}"),
            rect(class_left + 250, top, class_left + 420, top + 28),
            host.small_font,
            rgb(226, 230, 238),
            0,
        );
    }
    text(
        hdc,
        &format!("GPU dispatch  {:.1} Âµs", host.last.dispatch_us),
        rect(class_left, 560, class_left + 360, 588),
        host.body_font,
        rgb(240, 176, 64),
        0,
    );
    text(
        hdc,
        &format!("Zero-weight sparsity  {}", host.sparsity),
        rect(class_left, 592, width - 28, 620),
        host.small_font,
        rgb(148, 156, 174),
        0,
    );
    let side = 520;
    let _ = height;
    text(
        hdc,
        &format!(
            "{}   vendor 0x{:04X}   device 0x{:04X}   feature {}",
            host.gpu.info.name,
            host.gpu.info.vendor_id,
            host.gpu.info.device_id,
            host.gpu.feature_name()
        ),
        rect(side, 640, width - 24, 666),
        host.small_font,
        rgb(226, 230, 238),
        0,
    );
    text(
        hdc,
        "CPU i5-10210U    logical cores 0 / 2 / 4 / 6    lock xadd quaternary ALU",
        rect(side, 666, width - 24, 690),
        host.small_font,
        rgb(148, 156, 174),
        0,
    );
    text(hdc, &format!("GhostVault  {}", host.ghost), rect(28, 640, width - 24, 666), host.small_font, rgb(64, 206, 204), 0);
    text(
        hdc,
        &format!("GhostVault {:.1} Âµs     {}", host.ghost_us, host.paradox),
        rect(28, 666, side - 12, 700),
        host.small_font,
        rgb(148, 156, 174),
        0,
    );
}

fn coords(lparam: isize) -> (i32, i32) {
    ((lparam as i16) as i32, ((lparam >> 16) as i16) as i32)
}

fn rgb(r: u32, g: u32, b: u32) -> u32 {
    r | (g << 8) | (b << 16)
}

fn rect(left: i32, top: i32, right: i32, bottom: i32) -> Rect {
    Rect { left, top, right, bottom }
}

fn fill(hdc: *mut core::ffi::c_void, area: Rect, color: u32) {
    if area.right <= area.left || area.bottom <= area.top {
        return;
    }
    unsafe {
        SetDCBrushColor(hdc, color);
        FillRect(hdc, &area, GetStockObject(18));
    }
}

fn text(hdc: *mut core::ffi::c_void, value: &str, mut area: Rect, font: *mut core::ffi::c_void, color: u32, flags: u32) {
    let wide_text = wide(value);
    unsafe {
        let old = SelectObject(hdc, font);
        SetTextColor(hdc, color);
        SetBkMode(hdc, 1);
        DrawTextW(hdc, wide_text.as_ptr(), -1, &mut area, flags | 0x0800);
        SelectObject(hdc, old);
    }
}

fn make_font(height: i32, weight: i32) -> *mut core::ffi::c_void {
    let face = wide("Segoe UI");
    let font = unsafe { CreateFontW(height, 0, 0, 0, weight, 0, 0, 0, 1, 0, 0, 5, 0, face.as_ptr()) };
    if font.is_null() {
        panic!("CreateFontW failed");
    }
    font
}

fn window_size(client_w: i32, client_h: i32) -> (i32, i32) {
    let mut area = rect(0, 0, client_w, client_h);
    unsafe { AdjustWindowRect(&mut area, 0x00CF_0000, 0) };
    (area.right - area.left, area.bottom - area.top)
}

fn leak_wide(text: &str) -> &'static [u16] {
    Box::leak(wide(text).into_boxed_slice())
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

#[repr(C)]
struct WndClassExW {
    cb_size: u32,
    style: u32,
    wnd_proc: unsafe extern "system" fn(*mut core::ffi::c_void, u32, usize, isize) -> isize,
    cb_cls_extra: i32,
    cb_wnd_extra: i32,
    instance: *mut core::ffi::c_void,
    icon: *mut core::ffi::c_void,
    cursor: *mut core::ffi::c_void,
    background: *mut core::ffi::c_void,
    menu_name: *const u16,
    class_name: *const u16,
    icon_sm: *mut core::ffi::c_void,
}

#[repr(C)]
struct Msg {
    hwnd: *mut core::ffi::c_void,
    message: u32,
    _pad_msg: u32,
    wparam: usize,
    lparam: isize,
    time: u32,
    pt_x: i32,
    pt_y: i32,
    _pad_tail: u32,
}

#[repr(C)]
struct Paint {
    hdc: *mut core::ffi::c_void,
    erase: i32,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
    restore: i32,
    inc_update: i32,
    reserved: [u8; 32],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

#[repr(C)]
struct MinMaxInfo {
    reserved_x: i32,
    reserved_y: i32,
    max_size_x: i32,
    max_size_y: i32,
    max_pos_x: i32,
    max_pos_y: i32,
    min_track_x: i32,
    min_track_y: i32,
    max_track_x: i32,
    max_track_y: i32,
}

const _: () = assert!(std::mem::size_of::<WndClassExW>() == 80);
const _: () = assert!(std::mem::size_of::<Msg>() == 48);
const _: () = assert!(std::mem::size_of::<Paint>() == 72);
const _: () = assert!(std::mem::size_of::<Rect>() == 16);
const _: () = assert!(std::mem::size_of::<MinMaxInfo>() == 40);

#[link(name = "user32", kind = "raw-dylib")]
extern "system" {
    fn SetProcessDPIAware() -> i32;
    fn GetSystemMetrics(index: i32) -> i32;
    fn RegisterClassExW(class: *const WndClassExW) -> u16;
    fn CreateWindowExW(
        ex_style: u32,
        class_name: *const u16,
        window_name: *const u16,
        style: u32,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        parent: *mut core::ffi::c_void,
        menu: *mut core::ffi::c_void,
        instance: *mut core::ffi::c_void,
        param: *mut core::ffi::c_void,
    ) -> *mut core::ffi::c_void;
    fn ShowWindow(hwnd: *mut core::ffi::c_void, cmd: i32) -> i32;
    fn UpdateWindow(hwnd: *mut core::ffi::c_void) -> i32;
    fn GetMessageW(msg: *mut Msg, hwnd: *mut core::ffi::c_void, min: u32, max: u32) -> i32;
    fn TranslateMessage(msg: *const Msg) -> i32;
    fn DispatchMessageW(msg: *const Msg) -> isize;
    fn DefWindowProcW(hwnd: *mut core::ffi::c_void, msg: u32, wparam: usize, lparam: isize) -> isize;
    fn PostQuitMessage(code: i32);
    fn BeginPaint(hwnd: *mut core::ffi::c_void, paint: *mut Paint) -> *mut core::ffi::c_void;
    fn EndPaint(hwnd: *mut core::ffi::c_void, paint: *const Paint) -> i32;
    fn GetClientRect(hwnd: *mut core::ffi::c_void, rect: *mut Rect) -> i32;
    fn InvalidateRect(hwnd: *mut core::ffi::c_void, rect: *const Rect, erase: i32) -> i32;
    fn LoadCursorW(instance: *mut core::ffi::c_void, name: *const u16) -> *mut core::ffi::c_void;
    fn SetWindowLongPtrW(hwnd: *mut core::ffi::c_void, index: i32, value: isize) -> isize;
    fn GetWindowLongPtrW(hwnd: *mut core::ffi::c_void, index: i32) -> isize;
    fn SetCapture(hwnd: *mut core::ffi::c_void) -> *mut core::ffi::c_void;
    fn ReleaseCapture() -> i32;
    fn FillRect(hdc: *mut core::ffi::c_void, rect: *const Rect, brush: *mut core::ffi::c_void) -> i32;
    fn DrawTextW(hdc: *mut core::ffi::c_void, text: *const u16, count: i32, rect: *mut Rect, format: u32) -> i32;
    fn AdjustWindowRect(rect: *mut Rect, style: u32, menu: i32) -> i32;
}

#[link(name = "gdi32", kind = "raw-dylib")]
extern "system" {
    fn CreateFontW(
        height: i32,
        width: i32,
        escapement: i32,
        orientation: i32,
        weight: i32,
        italic: u32,
        underline: u32,
        strike: u32,
        charset: u32,
        out_precision: u32,
        clip_precision: u32,
        quality: u32,
        pitch: u32,
        face: *const u16,
    ) -> *mut core::ffi::c_void;
    fn CreateCompatibleDC(hdc: *mut core::ffi::c_void) -> *mut core::ffi::c_void;
    fn CreateCompatibleBitmap(hdc: *mut core::ffi::c_void, w: i32, h: i32) -> *mut core::ffi::c_void;
    fn SelectObject(hdc: *mut core::ffi::c_void, object: *mut core::ffi::c_void) -> *mut core::ffi::c_void;
    fn DeleteObject(object: *mut core::ffi::c_void) -> i32;
    fn DeleteDC(hdc: *mut core::ffi::c_void) -> i32;
    fn BitBlt(
        dst: *mut core::ffi::c_void,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        src: *mut core::ffi::c_void,
        sx: i32,
        sy: i32,
        rop: u32,
    ) -> i32;
    fn SetBkMode(hdc: *mut core::ffi::c_void, mode: i32) -> i32;
    fn SetTextColor(hdc: *mut core::ffi::c_void, color: u32) -> u32;
    fn GetStockObject(index: i32) -> *mut core::ffi::c_void;
    fn SetDCBrushColor(hdc: *mut core::ffi::c_void, color: u32) -> u32;
}

#[link(name = "kernel32", kind = "raw-dylib")]
extern "system" {
    fn GetModuleHandleW(name: *const u16) -> *mut core::ffi::c_void;
    fn GetLastError() -> u32;
}
