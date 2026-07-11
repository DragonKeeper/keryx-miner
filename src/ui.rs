use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode};
use crossterm::{execute, queue};
use log::Level;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::io::{stdout, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;
use time::{macros::format_description, OffsetDateTime};

use crate::stats::MinerStats;

const MAX_LOG_LINES: usize = 2000;
const REDRAW_RATE: Duration = Duration::from_millis(300);
const MIN_LOG_ROWS: u16 = 5;

#[derive(Copy, Clone)]
struct Palette {
    bg: Color,
    panel: Color,
    accent: Color,
    text: Color,
    muted: Color,
    ok: Color,
    dim: Color,
    bright: Color,
    warn: Color,
    mid: Color,
    err: Color,
}

const PALETTE_TRUECOLOR: Palette = Palette {
    bg: Color::Rgb {
        r: 0x07,
        g: 0x0a,
        b: 0x08,
    },
    panel: Color::Rgb {
        r: 0x0d,
        g: 0x11,
        b: 0x0d,
    },
    accent: Color::Rgb {
        r: 0x23,
        g: 0xd6,
        b: 0x58,
    },
    text: Color::Rgb {
        r: 0x6f,
        g: 0xd5,
        b: 0x83,
    },
    muted: Color::Rgb {
        r: 0x35,
        g: 0x44,
        b: 0x3a,
    },
    ok: Color::Rgb {
        r: 0x2e,
        g: 0xe3,
        b: 0x58,
    },
    dim: Color::Rgb {
        r: 0x4f,
        g: 0x9a,
        b: 0x60,
    },
    bright: Color::Rgb {
        r: 0x30,
        g: 0xff,
        b: 0x67,
    },
    warn: Color::Rgb {
        r: 0xff,
        g: 0xbf,
        b: 0x3f,
    },
    mid: Color::Rgb {
        r: 0x4e,
        g: 0xcb,
        b: 0x6a,
    },
    err: Color::Rgb {
        r: 0xe0,
        g: 0x52,
        b: 0x52,
    },
};

const PALETTE_ANSI: Palette = Palette {
    bg: Color::AnsiValue(232),
    panel: Color::AnsiValue(233),
    accent: Color::AnsiValue(41),
    text: Color::AnsiValue(114),
    muted: Color::AnsiValue(240),
    ok: Color::AnsiValue(46),
    dim: Color::AnsiValue(71),
    bright: Color::AnsiValue(82),
    warn: Color::AnsiValue(220),
    mid: Color::AnsiValue(78),
    err: Color::AnsiValue(203),
};

fn palette() -> Palette {
    static CHOSEN: OnceLock<Palette> = OnceLock::new();
    *CHOSEN.get_or_init(|| {
        if supports_truecolor() {
            PALETTE_TRUECOLOR
        } else {
            PALETTE_ANSI
        }
    })
}

fn supports_truecolor() -> bool {
    if let Ok(force) = env::var("KERYX_TRUECOLOR") {
        let force = force.trim().to_ascii_lowercase();
        if matches!(force.as_str(), "1" | "true" | "yes" | "on") {
            return true;
        }
        if matches!(force.as_str(), "0" | "false" | "no" | "off") {
            return false;
        }
    }

    let colorterm = env::var("COLORTERM")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if colorterm.contains("truecolor") || colorterm.contains("24bit") {
        return true;
    }

    let term = env::var("TERM").unwrap_or_default().to_ascii_lowercase();
    if term.starts_with("screen") || term == "linux" {
        return false;
    }

    term.contains("direct")
        || term.contains("kitty")
        || term.contains("wezterm")
        || term.contains("alacritty")
        || term.contains("foot")
}

pub struct UiState {
    lines: Mutex<VecDeque<String>>,
    scrollback_lines: AtomicU64,
    blocks_found: AtomicU64,
    rejected: AtomicU64,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            lines: Mutex::new(VecDeque::with_capacity(MAX_LOG_LINES)),
            scrollback_lines: AtomicU64::new(0),
            blocks_found: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
        }
    }

    pub fn push_log(&self, level: Level, message: &str) {
        if matches!(level, Level::Info | Level::Warn | Level::Error) {
            let mut lines = self.lines.lock().expect("ui log mutex poisoned");
            for entry in sanitize_log_message(message) {
                let line = format!("{} [{}] {}", tty_timestamp(), level, entry);
                lines.push_back(line);
            }
            while lines.len() > MAX_LOG_LINES {
                lines.pop_front();
            }
            if self.scrollback_lines.load(Ordering::Acquire) > 0 {
                let max_offset = lines.len().saturating_sub(1) as u64;
                let current = self.scrollback_lines.load(Ordering::Acquire);
                self.scrollback_lines
                    .store(current.saturating_add(1).min(max_offset), Ordering::Release);
            }
        }

        if message.contains("Found a block") {
            self.blocks_found.fetch_add(1, Ordering::AcqRel);
        }

        if message.contains("Failed submitting block")
            || message.contains("Failed submitting PoM block")
            || message.to_ascii_lowercase().contains("rejected")
        {
            self.rejected.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn blocks_found(&self) -> u64 {
        self.blocks_found.load(Ordering::Acquire)
    }

    fn rejected(&self) -> u64 {
        self.rejected.load(Ordering::Acquire)
    }

    fn visible_lines(&self, n: usize) -> Vec<String> {
        let lines = self.lines.lock().expect("ui log mutex poisoned");
        let len = lines.len();
        let offset = self.scrollback_lines.load(Ordering::Acquire) as usize;
        let end = len.saturating_sub(offset);
        let start = end.saturating_sub(n);
        lines.iter().skip(start).cloned().collect()
    }

    fn scroll_up(&self, amount: usize) {
        let len = self.lines.lock().expect("ui log mutex poisoned").len();
        let max_offset = len.saturating_sub(1) as u64;
        let current = self.scrollback_lines.load(Ordering::Acquire);
        self.scrollback_lines
            .store(current.saturating_add(amount as u64).min(max_offset), Ordering::Release);
    }

    fn scroll_down(&self, amount: usize) {
        let current = self.scrollback_lines.load(Ordering::Acquire);
        self.scrollback_lines
            .store(current.saturating_sub(amount as u64), Ordering::Release);
    }

    fn scroll_to_top(&self) {
        let len = self.lines.lock().expect("ui log mutex poisoned").len();
        self.scrollback_lines
            .store(len.saturating_sub(1) as u64, Ordering::Release);
    }

    fn scroll_to_live(&self) {
        self.scrollback_lines.store(0, Ordering::Release);
    }

    fn is_scrolled(&self) -> bool {
        self.scrollback_lines.load(Ordering::Acquire) > 0
    }

    fn scrollback_lines(&self) -> u64 {
        self.scrollback_lines.load(Ordering::Acquire)
    }
}

fn sanitize_log_message(message: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in message.lines() {
        // Drop escape/control characters so dependency output cannot execute
        // terminal control sequences inside the alternate-screen UI.
        let mut cleaned = String::new();
        let mut chars = raw.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\u{1b}' {
                // Skip basic ANSI CSI/OSC/SS3 escape payloads.
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if ('@'..='~').contains(&next) || next == '\u{7}' {
                        break;
                    }
                }
                continue;
            }
            if c.is_control() {
                continue;
            }
            cleaned.push(c);
        }

        if cleaned.trim().is_empty() {
            continue;
        }
        out.push(cleaned);
    }

    if out.is_empty() {
        vec!["(log)".to_string()]
    } else {
        out
    }
}

fn tty_timestamp() -> String {
    let format = format_description!("[year]-[month]-[day] [hour]:[minute]:[second]");
    OffsetDateTime::now_utc()
        .format(&format)
    .map(|stamp| format!("{} UTC", stamp))
    .unwrap_or_else(|_| "0000-00-00 00:00:00 UTC".to_string())
}

pub struct UiGuard {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Drop for UiGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub fn spawn_ui(stats: Arc<MinerStats>, ui_state: Arc<UiState>) -> UiGuard {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop);

    let handle = thread::spawn(move || {
        let mut out = stdout();
        let _ = enable_raw_mode();
        let _ = wait_for_stable_terminal_size();
        let _ = execute!(out, EnterAlternateScreen, Hide);
        let mut last_size: Option<(u16, u16)> = None;
        let mut pending_resize: Option<(u16, u16)> = None;
        let mut first_frame = true;
        let mut last_draw_key: Option<(u16, u16, bool, bool, u64, usize, u64, u64, u64)> = None;
        let mut last_drawn_at = std::time::Instant::now();

        while !stop_clone.load(Ordering::Acquire) {
            handle_input(&ui_state);
            let current_size = terminal::size()
                .ok()
                .filter(|(w, h)| *w > 0 && *h > 0);

            let Some(current_size) = current_size else {
                thread::sleep(REDRAW_RATE);
                continue;
            };

            let should_clear = if first_frame {
                first_frame = false;
                last_size = Some(current_size);
                pending_resize = None;
                true
            } else if last_size == Some(current_size) {
                pending_resize = None;
                false
            } else if pending_resize == Some(current_size) {
                last_size = Some(current_size);
                pending_resize = None;
                true
            } else {
                // Ignore a single-frame size wobble; clear only after stable resize confirmation.
                pending_resize = Some(current_size);
                thread::sleep(REDRAW_RATE);
                continue;
            };

            let snapshot = stats.snapshot();
            let draw_key = (
                current_size.0,
                current_size.1,
                snapshot.synced,
                snapshot.opoi_challenge_active,
                snapshot.total_hashrate_hs,
                snapshot.devices.len(),
                snapshot.last_update_epoch_s,
                ui_state.blocks_found(),
                ui_state.rejected(),
            );
            let periodic_refresh_due = last_drawn_at.elapsed() >= Duration::from_secs(1);
            if should_clear || periodic_refresh_due || last_draw_key != Some(draw_key) {
                draw_frame(&mut out, current_size, &snapshot, &ui_state, should_clear);
                last_draw_key = Some(draw_key);
                last_drawn_at = std::time::Instant::now();
            }
            thread::sleep(REDRAW_RATE);
        }

        let _ = execute!(out, Show, LeaveAlternateScreen);
        let _ = disable_raw_mode();
    });

    UiGuard {
        stop,
        handle: Some(handle),
    }
}

fn wait_for_stable_terminal_size() -> Option<(u16, u16)> {
    let mut last = None;
    for _ in 0..10 {
        let current = terminal::size().ok();
        if let Some((w, h)) = current {
            if w > 0 && h > 0 && last == current {
                return current;
            }
            last = current;
        }
        thread::sleep(Duration::from_millis(50));
    }
    last
}

enum PanelRow {
    Plain { text: String, fg: Color, bold: bool },
    Segments(Vec<(String, Color)>),
}

fn metric_row(label: &str, value: String, value_color: Color) -> PanelRow {
    PanelRow::Segments(vec![
        (format!(" {:<16}", label), palette().dim),
        (value, value_color),
    ])
}

fn draw_frame(
    out: &mut std::io::Stdout,
    size: (u16, u16),
    snapshot: &crate::stats::MinerStatsSnapshot,
    ui_state: &UiState,
    _clear_screen: bool,
) {
    let (w, h) = size;
    let total_width = w as usize;

    // Avoid terminal clear escapes (ED/EL): on some terminals they erase using
    // the shell default background, causing brief color flashes and smear.
    // We repaint the full visible frame explicitly instead.
    let _ = queue!(out, MoveTo(0, 0), SetBackgroundColor(palette().bg));

    let kernel_by_device: HashMap<u32, keryx_miner::pom_gpu::GpuKernelInfo> =
        keryx_miner::pom_gpu::list_gpu_kernel_info()
            .into_iter()
            .map(|k| (k.device_id, k))
            .collect();

    let left_w = total_width.saturating_sub(1) / 2;
    let right_w = total_width.saturating_sub(left_w + 1);
    let divider_x = left_w as u16;
    let right_x = left_w as u16 + 1;
    let compact = total_width < 132;

    let hashrate_value = format_hashrate(snapshot.total_hashrate_hs);
    let opoi_pause_value = if snapshot.opoi_challenge_active { "Active" } else { "Idle" };
    let blocks_found_value = ui_state.blocks_found();
    let rejected_value = ui_state.rejected();
    let uptime_value = format_duration(snapshot.uptime_s);
    let last_update_age = seconds_since(snapshot.last_update_epoch_s);
    let last_update_value = format!("{}s ago", last_update_age);
    let (memory_value, memory_color) = match system_memory_summary() {
        Some((used_mb, total_mb)) if total_mb > 0 => {
            let used_pct = ((used_mb as f64 / total_mb as f64) * 100.0).round() as u32;
            let color = if used_pct >= 97 {
                palette().err
            } else if used_pct >= 90 {
                palette().warn
            } else if used_pct >= 80 {
                palette().mid
            } else {
                palette().ok
            };
            (
                format!(
                    "{:.1}/{:.1} GB ({}%)",
                    used_mb as f64 / 1024.0,
                    total_mb as f64 / 1024.0,
                    used_pct
                ),
                color,
            )
        }
        _ => ("--".to_string(), palette().muted),
    };
    let (load_value, load_color) = match load_average_summary() {
        Some((load_1m, load_5m, load_15m)) => {
            let cores = std::thread::available_parallelism().ok().map(|n| n.get()).unwrap_or(1) as f64;
            let pct = if cores > 0.0 {
                ((load_1m / cores) * 100.0).round().clamp(0.0, 999.0) as u32
            } else {
                0
            };
            let color = if pct >= 95 {
                palette().err
            } else if pct >= 80 {
                palette().warn
            } else if pct >= 60 {
                palette().mid
            } else {
                palette().ok
            };
            (
                format!("{:.2}/{:.2}/{:.2}", load_1m, load_5m, load_15m),
                color,
            )
        }
        None => ("--".to_string(), palette().muted),
    };
    let address_value = snapshot
        .mining_address
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("(recovery mode)");
    let api_port_value = snapshot
        .api_port
        .map(|p| p.to_string())
        .unwrap_or_else(|| "--".to_string());
    let address_metric = truncate_middle(address_value, left_w.saturating_sub(12));
    let log_mode_value = if ui_state.is_scrolled() {
        format!("{} +{}", uppercase_first_char("scrolled"), ui_state.scrollback_lines())
    } else {
        uppercase_first_char("live")
    };

    let top_left = vec![
        (" Keryx Miner ".to_string(), palette().accent),
        ("| ".to_string(), palette().muted),
        (format!("{}  ", hashrate_value), palette().ok),
        ("| Node ".to_string(), palette().muted),
        (
            format!("{}  ", if snapshot.synced { "Synced" } else { "Not Synced" }),
            if snapshot.synced { palette().ok } else { palette().warn },
        ),
        ("| OPoI ".to_string(), palette().muted),
        (
            format!("{}", opoi_pause_value),
            if snapshot.opoi_challenge_active {
                palette().bright
            } else {
                palette().dim
            },
        ),
    ];

    let top_right = vec![
        (" API ".to_string(), palette().muted),
        (":".to_string(), palette().ok),
        (api_port_value, palette().ok),
        ("/v1/miner/stats".to_string(), palette().ok),
        (" | RAM ".to_string(), palette().muted),
        (memory_value, memory_color),
        (" | Load Avg [1m/5m/15m] ".to_string(), palette().muted),
        (load_value, load_color),
    ];

    draw_colored_segments_cell(
        out,
        0,
        0,
        left_w,
        &top_left,
        palette().panel,
    );
    draw_colored_cell(out, divider_x, 0, 1, "|", palette().muted, palette().panel, false);
    draw_colored_segments_cell(out, right_x, 0, right_w, &top_right, palette().panel);

    let available_content_rows = h.saturating_sub(MIN_LOG_ROWS + 3) as usize;
    let content_budget = available_content_rows.max(4);

    let mut left_rows: Vec<PanelRow> = vec![
        PanelRow::Plain {
            text: " Metrics".to_string(),
            fg: palette().accent,
            bold: true,
        },
        metric_row("Hashrate", hashrate_value.clone(), palette().ok),
        metric_row("Uptime", uptime_value.clone(), palette().text),
        PanelRow::Segments(vec![
            (format!(" {:<16}", "Address"), palette().dim),
            (address_metric, palette().ok),
        ]),
        PanelRow::Segments(vec![
            (format!(" {:<16}", "OPoI"), palette().dim),
            (
                opoi_pause_value.to_string(),
                if snapshot.opoi_challenge_active {
                    palette().bright
                } else {
                    palette().dim
                },
            ),
        ]),
        metric_row("Blocks Accepted", blocks_found_value.to_string(), palette().bright),
        metric_row(
            "Blocks Rejected",
            rejected_value.to_string(),
            if rejected_value > 0 { palette().err } else { palette().ok },
        ),
        metric_row(
            "Stats Updated",
            last_update_value,
            if last_update_age <= 3 { palette().ok } else { palette().warn },
        ),
        metric_row(
            "Log Mode",
            log_mode_value.clone(),
            if ui_state.is_scrolled() { palette().bright } else { palette().dim },
        ),
        PanelRow::Plain {
            text: "".to_string(),
            fg: palette().text,
            bold: false,
        },
        PanelRow::Plain {
            text: " Controls".to_string(),
            fg: palette().accent,
            bold: true,
        },
        PanelRow::Plain {
            text: if compact { " U/D Scroll" } else { " Up/Down      Scroll" }.to_string(),
            fg: palette().text,
            bold: false,
        },
        PanelRow::Plain {
            text: if compact {
                " Pg Fast"
            } else {
                " PgUp/PgDn    Scroll faster"
            }
            .to_string(),
            fg: palette().text,
            bold: false,
        },
        PanelRow::Plain {
            text: if compact {
                " Home/End L/O"
            } else {
                " Home/End     Oldest/Live"
            }
            .to_string(),
            fg: palette().text,
            bold: false,
        },
    ];

    if left_rows.len() > content_budget {
        left_rows.truncate(content_budget);
    }

    let mut right_rows: Vec<(String, Color, bool)> = Vec::new();
    right_rows.push((" Devices".to_string(), palette().accent, true));

    let id_w = if compact { 3usize } else { 4usize };
    let rate_w = if compact { 9usize } else { 11usize };
    let cc_w = if compact { 3usize } else { 5usize };
    let cm_w = if compact { 7usize } else { 9usize };
    let fan_w = 4usize;
    let bar_w = if compact {
        (right_w / 10).clamp(3, 7)
    } else {
        (right_w / 8).clamp(5, 11)
    };
    let kernel_w = right_w.saturating_sub(id_w + rate_w + cc_w + cm_w + fan_w + bar_w + 10);
    right_rows.push((
        format!(
            " {:<id_w$} {:<rate_w$} {:<cc_w$} {:<cm_w$} {:<fan_w$} {:<bar_w$} {:<kernel_w$}",
            "ID",
            "Hashrate",
            "CC",
            "Core/Mem",
            "Fan",
            "Load",
            "Kernel",
            id_w = id_w,
            rate_w = rate_w,
            cc_w = cc_w,
            cm_w = cm_w,
            fan_w = fan_w,
            bar_w = bar_w,
            kernel_w = kernel_w,
        ),
        palette().muted,
        true,
    ));

    let max_rate = snapshot
        .devices
        .iter()
        .map(|d| d.hashrate_hs)
        .max()
        .unwrap_or(1)
        .max(1);

    let max_device_rows = content_budget.saturating_sub(3).max(1);
    let shown = snapshot.devices.len().min(max_device_rows);
    for d in snapshot.devices.iter().take(shown) {
        let dev_id = parse_device_id(&d.id);
        let (compute, kernel) = dev_id
            .and_then(|id| kernel_by_device.get(&id))
            .map(|k| {
                let compute = match (k.cc_major, k.cc_minor) {
                    (Some(maj), Some(min)) => format!("{}.{}", maj, min),
                    _ => "n/a".to_string(),
                };
                (compute, k.image.clone())
            })
            .unwrap_or_else(|| ("n/a".to_string(), "n/a".to_string()));

        let load_blocks = (((d.hashrate_hs as f64 / max_rate as f64) * bar_w as f64).round() as usize)
            .clamp(0, bar_w);
        let load_bar = format!("{}{}", "=".repeat(load_blocks), ".".repeat(bar_w.saturating_sub(load_blocks)));
        let id_short = parse_device_id(&d.id)
            .map(|id| format!("#{}", id))
            .unwrap_or_else(|| trim_to_width(&d.id, id_w));
        let rate_short = trim_to_width(&format_hashrate(d.hashrate_hs), rate_w);
        let temp = d.temp_c.map(|v| format!("{}C", v)).unwrap_or_else(|| "--".to_string());
        let mem = d
            .memory_temp_c
            .map(|v| format!("{}C", v))
            .unwrap_or_else(|| "--".to_string());
        let core_mem = trim_to_width(&format!("{}/{}", temp, mem), cm_w);
        let fan = d
            .fan_percent
            .map(|v| format!("{}%", v))
            .unwrap_or_else(|| "--".to_string());
        let kernel_short = trim_to_width(&kernel, kernel_w);
        right_rows.push((
            format!(
                " {:<id_w$} {:<rate_w$} {:<cc_w$} {:<cm_w$} {:<fan_w$} {:<bar_w$} {:<kernel_w$}",
                id_short,
                rate_short,
                compute,
                core_mem,
                fan,
                load_bar,
                kernel_short,
                id_w = id_w,
                rate_w = rate_w,
                cc_w = cc_w,
                cm_w = cm_w,
                fan_w = fan_w,
                bar_w = bar_w,
                kernel_w = kernel_w,
            ),
            thermal_level_color(d.temp_c, d.memory_temp_c, d.fan_percent),
            false,
        ));
    }

    if snapshot.devices.len() > shown {
        right_rows.push((
            format!(" +{} more devices", snapshot.devices.len().saturating_sub(shown)),
            palette().muted,
            false,
        ));
    }

    let content_rows = left_rows.len().max(right_rows.len());

    for row in 0..content_rows {
        let y = row as u16 + 1;
        let left_bg = if row == 0 {
            palette().panel
        } else if row % 2 == 0 {
            palette().bg
        } else {
            palette().panel
        };
        let right_bg = if row == 0 {
            palette().panel
        } else if row % 2 == 0 {
            palette().bg
        } else {
            palette().panel
        };

        match left_rows.get(row) {
            Some(PanelRow::Plain { text, fg, bold }) => {
                draw_colored_cell(out, 0, y, left_w, text, *fg, left_bg, *bold);
            }
            Some(PanelRow::Segments(segments)) => {
                draw_colored_segments_cell(out, 0, y, left_w, segments, left_bg);
            }
            None => draw_colored_cell(out, 0, y, left_w, "", palette().text, left_bg, false),
        }

        draw_colored_cell(out, divider_x, y, 1, "|", palette().muted, palette().panel, false);

        let (right_text, right_fg, right_bold) = right_rows
            .get(row)
            .cloned()
            .unwrap_or_else(|| ("".to_string(), palette().text, false));
        draw_colored_cell(out, right_x, y, right_w, &right_text, right_fg, right_bg, right_bold);
    }

    let separator_y = content_rows as u16 + 1;
    draw_colored_line(
        out,
        separator_y,
        &"-".repeat(total_width),
        palette().muted,
        Some(palette().panel),
        false,
    );

    let log_header_y = separator_y + 1;
    draw_colored_segments_cell(
        out,
        0,
        log_header_y,
        total_width,
        &[
            (" Logs ".to_string(), palette().accent),
            ("| ".to_string(), palette().muted),
            (
                format!("{}  ", log_mode_value),
                if ui_state.is_scrolled() { palette().bright } else { palette().dim },
            ),
            (
                if compact {
                    "| U/D Scroll  Pg Fast  Home/End"
                } else {
                    "| Up/Down Scroll  PgUp/PgDn Fast  Home Oldest  End Live"
                }
                .to_string(),
                palette().text,
            ),
        ],
        palette().panel,
    );

    let log_start_y = log_header_y + 1;
    let log_rows = h.saturating_sub(log_start_y);
    let log_lines = ui_state.visible_lines(log_rows as usize);

    for (i, line) in log_lines.iter().enumerate() {
        let y = log_start_y + i as u16;
        if y >= h {
            break;
        }
        let clipped = trim_to_width(line, total_width);
        draw_colored_segments_cell(
            out,
            0,
            y,
            total_width,
            &[(clipped, color_for_log_line(line))],
            palette().bg,
        );
    }

    // Clear any remaining rows in the log panel (mostly relevant during startup / resize).
    for y in (log_start_y + log_lines.len() as u16)..h {
        draw_colored_cell(out, 0, y, total_width, "", palette().text, palette().bg, false);
    }

    let _ = queue!(out, ResetColor, SetAttribute(Attribute::Reset));
    let _ = out.flush();
}

fn draw_colored_line(
    out: &mut std::io::Stdout,
    y: u16,
    text: &str,
    fg: Color,
    bg: Option<Color>,
    bold: bool,
) {
    let _ = queue!(out, MoveTo(0, y));
    if let Some(bg) = bg {
        let _ = queue!(out, SetBackgroundColor(bg));
    }
    let _ = queue!(out, SetForegroundColor(fg));
    if bold {
        let _ = queue!(out, SetAttribute(Attribute::Bold));
    }
    let _ = queue!(out, Print(text));
    let _ = queue!(out, ResetColor, SetAttribute(Attribute::Reset));
}

fn draw_colored_cell(
    out: &mut std::io::Stdout,
    x: u16,
    y: u16,
    width: usize,
    text: &str,
    fg: Color,
    bg: Color,
    bold: bool,
) {
    let clipped = trim_to_width(text, width);
    let pad = width.saturating_sub(clipped.len());
    let _ = queue!(out, MoveTo(x, y), SetBackgroundColor(bg), SetForegroundColor(fg));
    if bold {
        let _ = queue!(out, SetAttribute(Attribute::Bold));
    }
    let _ = queue!(out, Print(clipped));
    if pad > 0 {
        let _ = queue!(out, Print(" ".repeat(pad)));
    }
    let _ = queue!(out, ResetColor, SetAttribute(Attribute::Reset));
}

fn draw_colored_segments_cell(
    out: &mut std::io::Stdout,
    x: u16,
    y: u16,
    width: usize,
    segments: &[(String, Color)],
    bg: Color,
) {
    let _ = queue!(out, MoveTo(x, y), SetBackgroundColor(bg));
    let mut remaining = width;

    for (text, fg) in segments {
        if remaining == 0 {
            break;
        }
        let clipped = trim_to_width(text, remaining);
        let taken = clipped.len();
        if taken == 0 {
            continue;
        }
        let _ = queue!(out, SetForegroundColor(*fg), Print(clipped));
        remaining = remaining.saturating_sub(taken);
    }

    if remaining > 0 {
        let _ = queue!(out, SetForegroundColor(palette().text), Print(" ".repeat(remaining)));
    }

    let _ = queue!(out, ResetColor, SetAttribute(Attribute::Reset));
}

fn color_for_log_line(line: &str) -> Color {
    let lower = line.to_ascii_lowercase();
    if lower.contains("[error") || lower.contains(" error ") {
        palette().err
    } else if lower.contains("[warn") || lower.contains(" warning ") {
        palette().warn
    } else if lower.contains("opoi inference in progress") {
        palette().bright
    } else if lower.contains("inference") {
        palette().mid
    } else if lower.contains("escrowwatcher") && lower.contains("claim accepted") {
        palette().bright
    } else if line.contains("Found a block") || line.contains("Block submitted successfully") {
        palette().bright
    } else {
        palette().text
    }
}

fn seconds_since(epoch_s: u64) -> u64 {
    let now = OffsetDateTime::now_utc().unix_timestamp();
    if now <= 0 {
        0
    } else {
        (now as u64).saturating_sub(epoch_s)
    }
}

fn format_duration(seconds: u64) -> String {
    let h = seconds / 3600;
    let m = (seconds % 3600) / 60;
    let s = seconds % 60;

    if h > 0 {
        format!("{}h {}m", h, m)
    } else if m > 0 {
        format!("{}m {}s", m, s)
    } else {
        format!("{}s", s)
    }
}

fn truncate_middle(value: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return value.to_string();
    }
    if max_chars <= 3 {
        return "...".chars().take(max_chars).collect();
    }

    let keep = max_chars - 3;
    let left = keep / 2;
    let right = keep - left;
    let mut out = String::new();
    out.extend(chars.iter().take(left));
    out.push_str("...");
    out.extend(chars.iter().skip(chars.len().saturating_sub(right)));
    out
}

fn system_memory_summary() -> Option<(u64, u64)> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = None;
    let mut avail_kb = None;

    for line in meminfo.lines() {
        if let Some(v) = line.strip_prefix("MemTotal:") {
            total_kb = parse_meminfo_kb(v);
        } else if let Some(v) = line.strip_prefix("MemAvailable:") {
            avail_kb = parse_meminfo_kb(v);
        }
        if total_kb.is_some() && avail_kb.is_some() {
            break;
        }
    }

    let total_kb = total_kb?;
    let avail_kb = avail_kb?;
    let used_kb = total_kb.saturating_sub(avail_kb);
    Some((used_kb / 1024, total_kb / 1024))
}

fn parse_meminfo_kb(value: &str) -> Option<u64> {
    value
        .split_whitespace()
        .next()
        .and_then(|x| x.parse::<u64>().ok())
}

fn load_average_summary() -> Option<(f64, f64, f64)> {
    let loadavg = std::fs::read_to_string("/proc/loadavg").ok()?;
    let mut values = loadavg
        .split_whitespace()
        .take(3)
        .map(|v| v.parse::<f64>().ok());
    let load_1m = values.next().flatten()?;
    let load_5m = values.next().flatten()?;
    let load_15m = values.next().flatten()?;
    Some((load_1m, load_5m, load_15m))
}

fn thermal_level_color(core_temp_c: Option<u32>, memory_temp_c: Option<u32>, fan_percent: Option<u32>) -> Color {
    // NVIDIA-oriented thermal bands:
    // Core: warm 76-85, hot 86-92, dangerous 93+
    // Mem : warm 86-95, hot 96-104, dangerous 105+
    let core_crit = core_temp_c.is_some_and(|t| t >= 93);
    let mem_crit = memory_temp_c.is_some_and(|t| t >= 105);
    if core_crit || mem_crit {
        return palette().err;
    }

    let core_hot = core_temp_c.is_some_and(|t| t >= 86);
    let mem_hot = memory_temp_c.is_some_and(|t| t >= 96);
    let fan_warn = fan_percent.is_some_and(|f| f >= 90);
    if core_hot || mem_hot || fan_warn {
        return palette().warn;
    }

    let core_warm = core_temp_c.is_some_and(|t| t >= 76);
    let mem_warm = memory_temp_c.is_some_and(|t| t >= 86);
    if core_warm || mem_warm {
        return palette().mid;
    }

    if core_temp_c.is_some() || memory_temp_c.is_some() {
        palette().ok
    } else if fan_percent.is_some() {
        palette().dim
    } else {
        palette().muted
    }
}

fn handle_input(ui_state: &UiState) {
    while event::poll(Duration::from_millis(0)).unwrap_or(false) {
        let Ok(Event::Key(key)) = event::read() else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        if matches!(key.code, KeyCode::Char('c') | KeyCode::Char('C'))
            && key.modifiers.contains(KeyModifiers::CONTROL)
        {
            let mut out = stdout();
            let _ = execute!(out, Show, LeaveAlternateScreen);
            let _ = disable_raw_mode();
            std::process::exit(130);
        }
        match key.code {
            KeyCode::Up => ui_state.scroll_up(1),
            KeyCode::Down => ui_state.scroll_down(1),
            KeyCode::PageUp => ui_state.scroll_up(10),
            KeyCode::PageDown => ui_state.scroll_down(10),
            KeyCode::Home => ui_state.scroll_to_top(),
            KeyCode::End => ui_state.scroll_to_live(),
            _ => {}
        }
    }
}

fn trim_to_width(s: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut out = String::new();
    for c in s.chars() {
        if out.len() + c.len_utf8() > width {
            break;
        }
        out.push(c);
    }
    out
}

fn format_hashrate(hs: u64) -> String {
    let v = hs as f64;
    if v < 1_000.0 {
        format!("{} hs", hs)
    } else if v < 1_000_000.0 {
        format!("{:.2} Khs", v / 1_000.0)
    } else if v < 1_000_000_000.0 {
        format!("{:.2} Mhs", v / 1_000_000.0)
    } else if v < 1_000_000_000_000.0 {
        format!("{:.2} Ghs", v / 1_000_000_000.0)
    } else {
        format!("{:.2} Ths", v / 1_000_000_000_000.0)
    }
}

fn parse_device_id(worker_id: &str) -> Option<u32> {
    worker_id
        .strip_prefix('#')
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<u32>().ok())
}

fn uppercase_first_char(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
        None => String::new(),
    }
}

