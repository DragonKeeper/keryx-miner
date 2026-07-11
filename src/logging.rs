use env_logger::Logger;
use log::{LevelFilter, Metadata, Record, SetLoggerError};
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;
use std::sync::Arc;

use crate::ui::UiState;

struct MinerLogger {
    inner: Logger,
    emit_inner: bool,
    ui: Option<Arc<UiState>>,
    plain_log_file: Option<Mutex<File>>,
}

impl log::Log for MinerLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        self.inner.enabled(metadata)
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        if self.emit_inner {
            self.inner.log(record);
        }

        if let Some(ui) = &self.ui {
            ui.push_log(record.level(), &format!("{}", record.args()));
        }

        if let Some(file) = &self.plain_log_file {
            if let Ok(mut f) = file.lock() {
                let _ = writeln!(f, "[{}] {}", record.level(), record.args());
            }
        }
    }

    fn flush(&self) {
        self.inner.flush();
    }
}

pub fn init_logging(
    level: LevelFilter,
    ui: Option<Arc<UiState>>,
    emit_inner: bool,
    plain_log_file: Option<File>,
) -> Result<(), SetLoggerError> {
    let mut builder = env_logger::Builder::new();
    builder.filter_level(level).parse_default_env();
    let inner = builder.build();
    let max_level = inner.filter();

    let logger = MinerLogger {
        inner,
        emit_inner,
        ui,
        plain_log_file: plain_log_file.map(Mutex::new),
    };

    log::set_boxed_logger(Box::new(logger))?;
    log::set_max_level(max_level);
    Ok(())
}
