#![forbid(unsafe_code)]

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table, Tabs};
use ratatui::Terminal;
use ucf_client::Endpoint;
use ucf_console::{export_view, load_snapshot, ConsoleConfig, ViewTab};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (cfg, once, once_out) = parse_args()?;
    if once {
        let snapshot = load_snapshot(&cfg)?;
        let out = once_out.unwrap_or_else(|| PathBuf::from("./out/console_once.json"));
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(out, serde_json::to_string_pretty(&snapshot.overview)?)?;
        return Ok(());
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut tab = ViewTab::Overview;
    let mut snapshot = load_snapshot(&cfg)?;

    loop {
        terminal.draw(|f| draw_ui(f, tab, &snapshot))?;
        if event::poll(Duration::from_millis(250))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('1') => tab = ViewTab::Overview,
                    KeyCode::Char('2') => tab = ViewTab::Alerts,
                    KeyCode::Char('3') => tab = ViewTab::Drift,
                    KeyCode::Char('4') => tab = ViewTab::Runs,
                    KeyCode::Char('r') => snapshot = load_snapshot(&cfg)?,
                    KeyCode::Char('e') => export_view(&snapshot, tab, &cfg.export_path)?,
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn parse_args() -> Result<(ConsoleConfig, bool, Option<PathBuf>), Box<dyn std::error::Error>> {
    let mut cfg = ConsoleConfig::default();
    let mut once = false;
    let mut once_out = None;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--once" => once = true,
            "--workdir" => {
                i += 1;
                cfg.workdir = PathBuf::from(args.get(i).ok_or("missing value for --workdir")?);
            }
            "--endpoint" => {
                i += 1;
                cfg.endpoint = Endpoint::parse(args.get(i).ok_or("missing value for --endpoint")?)?;
            }
            "--token" => {
                i += 1;
                cfg.token = args.get(i).ok_or("missing value for --token")?.clone();
            }
            "--alerts" => {
                i += 1;
                cfg.alerts_path = PathBuf::from(args.get(i).ok_or("missing value for --alerts")?);
            }
            "--drift" => {
                i += 1;
                cfg.drift_path = PathBuf::from(args.get(i).ok_or("missing value for --drift")?);
            }
            "--out" => {
                i += 1;
                once_out = Some(PathBuf::from(args.get(i).ok_or("missing value for --out")?));
            }
            _ => {}
        }
        i += 1;
    }

    if cfg.token.is_empty() {
        cfg.token = std::env::var("UCF_GATEWAY_TOKEN").unwrap_or_default();
    }

    Ok((cfg, once, once_out))
}

fn draw_ui(frame: &mut ratatui::Frame, tab: ViewTab, snapshot: &ucf_console::ConsoleSnapshot) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(2),
        ])
        .split(frame.size());

    let tab_titles = ["Overview", "Alerts", "Drift", "Runs"];
    let selected = match tab {
        ViewTab::Overview => 0,
        ViewTab::Alerts => 1,
        ViewTab::Drift => 2,
        ViewTab::Runs => 3,
    };
    let tabs = Tabs::new(
        tab_titles
            .iter()
            .copied()
            .map(Line::from)
            .collect::<Vec<_>>(),
    )
    .select(selected)
    .block(Block::default().borders(Borders::ALL).title("ucf-console"))
    .highlight_style(Style::default().fg(Color::Cyan));
    frame.render_widget(tabs, chunks[0]);

    match tab {
        ViewTab::Overview => {
            let o = &snapshot.overview;
            let text = vec![
                Line::from(format!(
                    "status={} drift_status={} strict_mode={}",
                    o.status, o.drift_status, o.strict_mode
                )),
                Line::from(format!("run_id={}", o.run_id)),
                Line::from(format!(
                    "policy_digest={} manifest_digest={}",
                    o.policy_graph_digest_prefix, o.manifest_digest_prefix
                )),
                Line::from(format!(
                    "last_tick_age_ms={} emergency_active={}",
                    o.last_tick_age_ms, o.emergency_active
                )),
                Line::from(format!(
                    "active_slots={} alarms={} violations={}",
                    o.active_slots_summary, o.drift_alarms, o.violations
                )),
            ];
            frame.render_widget(
                Paragraph::new(text).block(Block::default().borders(Borders::ALL)),
                chunks[1],
            );
        }
        ViewTab::Alerts => {
            let rows = snapshot.alerts_active.iter().map(|a| {
                Row::new(vec![
                    a.alert_id.clone(),
                    a.severity.clone(),
                    a.triggered_at_t.to_string(),
                    a.rule_id.clone(),
                ])
            });
            let table = Table::new(
                rows,
                [
                    Constraint::Length(24),
                    Constraint::Length(10),
                    Constraint::Length(14),
                    Constraint::Min(10),
                ],
            )
            .header(
                Row::new(vec!["alert_id", "severity", "since_t", "rule_id"])
                    .style(Style::default().fg(Color::Yellow)),
            )
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("active alerts (bounded)"),
            );
            frame.render_widget(table, chunks[1]);
        }
        ViewTab::Drift => {
            let rows = snapshot.drift.iter().map(|d| {
                Row::new(vec![
                    d.stage_id.clone(),
                    d.status.clone(),
                    d.active_alarms.join(","),
                    d.windows_count.to_string(),
                ])
            });
            let table = Table::new(
                rows,
                [
                    Constraint::Length(20),
                    Constraint::Length(12),
                    Constraint::Min(20),
                    Constraint::Length(10),
                ],
            )
            .header(
                Row::new(vec!["stage", "status", "active_alarms", "windows"])
                    .style(Style::default().fg(Color::Yellow)),
            )
            .block(Block::default().borders(Borders::ALL).title("drift"));
            frame.render_widget(table, chunks[1]);
        }
        ViewTab::Runs => {
            let rows = snapshot.runs.iter().map(|r| {
                Row::new(vec![
                    r.run_id.clone(),
                    r.started_at_tick.to_string(),
                    r.status.clone(),
                    r.profile.clone(),
                ])
            });
            let table = Table::new(
                rows,
                [
                    Constraint::Length(24),
                    Constraint::Length(14),
                    Constraint::Length(10),
                    Constraint::Length(12),
                ],
            )
            .header(
                Row::new(vec!["run_id", "started_tick", "status", "profile"])
                    .style(Style::default().fg(Color::Yellow)),
            )
            .block(Block::default().borders(Borders::ALL).title("last 20 runs"));
            frame.render_widget(table, chunks[1]);
        }
    }

    frame.render_widget(
        Paragraph::new(Line::from(vec![Span::raw(
            "keys: 1-4 tabs  r refresh  e export  q quit  | read-only",
        )]))
        .block(Block::default().borders(Borders::ALL)),
        chunks[2],
    );
}
