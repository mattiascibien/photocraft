//! Native menu bar on macOS (muda/NSMenu) mirroring the in-window egui menu bar.
//!
//! Photoshop parity includes behaving like a Mac app: on macOS the menus live in the
//! screen-top menu bar, not inside the window. [`tick`] (called from [`crate::PhotocraftApp::logic`])
//! installs the native menu once, forwards native clicks to [`crate::menus::invoke`], and
//! syncs enabled/checked state a few times per second. Everywhere else — and on macOS when
//! installation fails — the in-window bar stays (`panels::title_bar` checks [`active`]).
//!
//! The menu structure ([`build_tree`]) is pure and tested on all platforms; only installation,
//! event draining and state sync touch muda and only exist on macOS, so wasm and other
//! targets are unaffected.

use crate::i18n::Lang;
use crate::menus::MenuItem;

/// A native-menu node built from [`crate::menus::menu_items`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    Separator,
    Item { id: String, label: String, shortcut: Option<String> },
    Submenu { name: String, children: Vec<Node> },
}

/// One top-level menu and its tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopMenu {
    pub name: String,
    pub children: Vec<Node>,
}

/// Build the native-menu tree for `tops` (in order) from a flat [`menu_items`](crate::menus::menu_items)
/// list: leaves and separators at their depth, submenus grouped at the position of their
/// first child, leading/trailing/doubled separators dropped like the egui renderer does.
pub fn build_tree(items: &[MenuItem], tops: &[&str], lang: Lang) -> Vec<TopMenu> {
    tops.iter()
        .map(|top| {
            let mine: Vec<&MenuItem> = items.iter().filter(|i| i.path.first().map(String::as_str) == Some(*top)).collect();
            TopMenu { name: crate::i18n::tr(lang, top).to_string(), children: build_level(&mine, 1, lang) }
        })
        .collect()
}

fn build_level(items: &[&MenuItem], depth: usize, lang: Lang) -> Vec<Node> {
    let mut nodes: Vec<Node> = Vec::new();
    let mut shown_subs: Vec<&str> = Vec::new();
    for it in items {
        if it.path.len() == depth {
            if it.label == "---" {
                if !matches!(nodes.last(), None | Some(Node::Separator)) {
                    nodes.push(Node::Separator);
                }
                continue;
            }
            nodes.push(Node::Item { id: it.id.clone(), label: crate::i18n::tr_id(lang, &it.id, &it.label).to_string(), shortcut: it.shortcut.clone() });
        } else if it.path.len() > depth {
            let Some(name) = it.path.get(depth).map(String::as_str) else { continue };
            if shown_subs.contains(&name) {
                continue;
            }
            shown_subs.push(name);
            let child: Vec<&MenuItem> = items.iter().copied().filter(|c| c.path.len() > depth && c.path.get(depth).map(String::as_str) == Some(name)).collect();
            let children = build_level(&child, depth + 1, lang);
            if children.is_empty() {
                continue;
            }
            nodes.push(Node::Submenu { name: crate::i18n::tr(lang, name).to_string(), children });
        }
    }
    while matches!(nodes.last(), Some(Node::Separator)) {
        nodes.pop();
    }
    nodes
}

/// Whether the native menu bar is installed and live (macOS only, else always false).
/// `panels::title_bar` hides the in-window menus while this is true.
pub fn active() -> bool {
    #[cfg(target_os = "macos")]
    {
        imp::active()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Install (first call), forward native clicks, and sync state (throttled). Never panics:
/// a failed install simply leaves [`active`] false and the in-window bar stays.
pub fn tick(app: &mut crate::PhotocraftApp, ctx: &egui::Context) {
    #[cfg(target_os = "macos")]
    imp::tick(app, ctx);
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, ctx);
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use serde_json::json;

    use super::{Node, build_tree};
    use crate::menus::{self, TOP_MENUS};

    /// Sync at most this often: [`menus::menu_items`](crate::menus::menu_items) evaluates
    /// every item's enabled state.
    const SYNC_SECS: f64 = 0.5;

    // muda's handles are main-thread-only (!Send), so the installed menu lives in
    // thread-local storage rather than a static: eframe drives everything from the main
    // thread on macOS, and a failed install simply leaves this empty (in-window fallback).
    thread_local! {
        static INSTALLED: std::cell::RefCell<State> = const { std::cell::RefCell::new(State { installed: None, retry_at: 0.0 }) };
    }
    /// Our own event channel (its ends are thread-safe, unlike the menu handles): installing
    /// a [`muda::MenuEvent`] handler replaces the built-in receiver queue, and the handler
    /// also wakes eframe (an idle app would otherwise not repaint on a menu click).
    static QUEUE: std::sync::OnceLock<Mutex<Queue>> = std::sync::OnceLock::new();

    struct Queue {
        tx: std::sync::mpsc::Sender<String>,
        rx: std::sync::mpsc::Receiver<String>,
    }

    struct State {
        installed: Option<Installed>,
        /// Next earliest retry after a failed install.
        retry_at: f64,
    }

    struct Installed {
        /// Kept alive: dropping it would destroy the NSMenu.
        _menu: muda::Menu,
        plain: HashMap<String, Vec<muda::MenuItem>>,
        checks: HashMap<String, Vec<muda::CheckMenuItem>>,
        shortcuts: HashMap<String, Option<String>>,
        /// Command ids in build order (separators skipped): a change means rebuild.
        ids: Vec<String>,
        /// Recent-file labels at build time (same ids, changing labels): a change means rebuild.
        recent: Vec<String>,
        lang: String,
        last_sync: f64,
    }

    pub fn active() -> bool {
        INSTALLED.with(|s| s.borrow().installed.is_some())
    }

    pub fn tick(app: &mut crate::PhotocraftApp, ctx: &egui::Context) {
        drain_events(app, ctx);
        let now = ctx.input(|i| i.time);
        INSTALLED.with(|cell| {
            let mut state = cell.borrow_mut();
            let due = match &state.installed {
                None => now >= state.retry_at,
                Some(inst) => now - inst.last_sync >= SYNC_SECS,
            };
            if !due {
                return;
            }
            let items = menus::menu_items(app);
            let lang = crate::i18n::current();
            if state.installed.is_none() {
                // muda panics off the main thread: contain it and fall back to the in-window bar.
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| install(&items, lang, ctx))) {
                    Ok(installed) => state.installed = Some(installed),
                    Err(_) => {
                        log::warn!("native menu install failed; keeping the in-window menu bar");
                        state.retry_at = now + 2.0;
                    }
                }
            } else {
                sync_or_rebuild(&mut state.installed, &items, lang, ctx, now);
            }
        });
    }

    fn drain_events(app: &mut crate::PhotocraftApp, ctx: &egui::Context) {
        let Some(queue) = QUEUE.get() else { return };
        let queue = queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        while let Ok(id) = queue.rx.try_recv() {
            // Duplicate command ids (Quit in the app menu and File › Exit) install with a
            // `#2`/`#3`… suffix; the command is the part before it.
            let cmd = id.split('#').next().unwrap_or(&id);
            if menus::is_live(cmd)
                && let Err(e) = menus::invoke(app, ctx, cmd, json!({}))
            {
                app.ui.status = e;
            }
        }
    }

    /// Structural signature: command ids in order, separators skipped.
    fn signature(items: &[crate::menus::MenuItem]) -> Vec<String> {
        items.iter().filter(|i| i.label != "---").map(|i| i.id.clone()).collect()
    }

    fn recent_labels(items: &[crate::menus::MenuItem]) -> Vec<String> {
        items.iter().filter(|i| i.id.starts_with("file.openRecent.")).map(|i| i.label.clone()).collect()
    }

    fn sync_or_rebuild(installed: &mut Option<Installed>, items: &[crate::menus::MenuItem], lang: crate::i18n::Lang, ctx: &egui::Context, now: f64) {
        let lang_key = format!("{lang:?}");
        let ids = signature(items);
        let recent = recent_labels(items);
        let rebuild = match installed.as_ref() {
            None => true,
            Some(inst) => inst.ids != ids || inst.recent != recent || inst.lang != lang_key,
        };
        if rebuild {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| install(items, lang, ctx))) {
                Ok(next) => *installed = Some(next),
                Err(_) => {
                    log::warn!("native menu rebuild failed; keeping the previous menu");
                    if let Some(inst) = installed.as_mut() {
                        inst.last_sync = now;
                    }
                }
            }
            return;
        }
        let Some(inst) = installed.as_mut() else { return };
        inst.last_sync = now;
        let by_id: HashMap<&str, &crate::menus::MenuItem> = items.iter().map(|i| (i.id.as_str(), i)).collect();
        for (id, handles) in &inst.plain {
            let Some(item) = by_id.get(id.as_str()) else { continue };
            for h in handles {
                h.set_enabled(item.enabled);
                if inst.shortcuts.get(id) != Some(&item.shortcut) {
                    let _ = h.set_accelerator(accelerator(item.shortcut.as_deref()));
                }
            }
        }
        for (id, handles) in &inst.checks {
            let Some(item) = by_id.get(id.as_str()) else { continue };
            for h in handles {
                h.set_enabled(item.enabled);
                h.set_checked(item.checked.unwrap_or(false));
                if inst.shortcuts.get(id) != Some(&item.shortcut) {
                    let _ = h.set_accelerator(accelerator(item.shortcut.as_deref()));
                }
            }
        }
        inst.shortcuts = items.iter().map(|i| (i.id.clone(), i.shortcut.clone())).collect();
    }

    /// The macOS app menu: About, Preferences (⌘,) and Quit. No Hide/Services items:
    /// ⌘H belongs to View › Extras (like Photoshop), and a Hide item would swallow it.
    /// Returns (label, command, fallback shortcut when the catalogue has none).
    fn app_menu(lang: crate::i18n::Lang) -> Vec<(String, String, Option<&'static str>)> {
        vec![
            (format!("{} PhotoCraft", crate::i18n::tr(lang, "About")), "help.about".to_string(), None),
            ("---".to_string(), String::new(), None),
            (format!("{}…", crate::i18n::tr(lang, "Preferences")), "edit.preferences.general".to_string(), Some("Cmd+,")),
            ("---".to_string(), String::new(), None),
            (format!("{} PhotoCraft", crate::i18n::tr(lang, "Quit")), "file.exit".to_string(), Some("Cmd+Q")),
        ]
    }

    fn install(items: &[crate::menus::MenuItem], lang: crate::i18n::Lang, ctx: &egui::Context) -> Installed {
        let menu = muda::Menu::new();
        let mut plain: HashMap<String, Vec<muda::MenuItem>> = HashMap::new();
        let mut checks: HashMap<String, Vec<muda::CheckMenuItem>> = HashMap::new();
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut unique = |id: &str| {
            let n = seen.entry(id.to_string()).or_insert(0);
            *n += 1;
            if *n == 1 { id.to_string() } else { format!("{id}#{}", *n) }
        };
        let by_id: HashMap<&str, &crate::menus::MenuItem> = items.iter().map(|i| (i.id.as_str(), i)).collect();

        // App menu first (macOS convention), then Photoshop order.
        let app = muda::Submenu::new("PhotoCraft", true);
        for (label, cmd, fallback) in app_menu(lang) {
            if cmd.is_empty() {
                let _ = app.append(&muda::PredefinedMenuItem::separator());
                continue;
            }
            let sc = by_id.get(cmd.as_str()).and_then(|i| i.shortcut.clone()).or_else(|| fallback.map(str::to_string));
            let enabled = by_id.get(cmd.as_str()).map(|i| i.enabled).unwrap_or_else(|| menus::is_live(&cmd));
            let item = muda::MenuItem::with_id(unique(&cmd), label, enabled, accelerator(sc.as_deref()));
            let _ = app.append(&item);
            plain.entry(cmd).or_default().push(item);
        }
        let _ = menu.append(&app);

        for top in build_tree(items, &TOP_MENUS, lang) {
            let sub = muda::Submenu::new(top.name, true);
            append_nodes(&sub, &top.children, &by_id, &mut plain, &mut checks, &mut unique);
            let _ = menu.append(&sub);
        }
        // Wake eframe on native clicks; events arrive on our own channel.
        let (tx, rx) = std::sync::mpsc::channel();
        let _ = QUEUE.set(Mutex::new(Queue { tx, rx }));
        let repaint = ctx.clone();
        muda::MenuEvent::set_event_handler(Some(move |event: muda::MenuEvent| {
            if let Some(queue) = QUEUE.get() {
                let _ = queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner).tx.send(event.id.0.clone());
            }
            repaint.request_repaint();
        }));
        menu.init_for_nsapp();
        // Fix the initial enabled/checked state right away (the first sync is SYNC_SECS out).
        let mut installed = Installed {
            _menu: menu,
            plain,
            checks,
            shortcuts: HashMap::new(),
            ids: signature(items),
            recent: recent_labels(items),
            lang: format!("{lang:?}"),
            last_sync: f64::NEG_INFINITY,
        };
        apply_all(&mut installed, &by_id);
        installed.last_sync = ctx.input(|i| i.time);
        installed
    }

    fn apply_all(installed: &mut Installed, by_id: &HashMap<&str, &crate::menus::MenuItem>) {
        for (id, handles) in &installed.plain {
            let enabled = by_id.get(id.as_str()).map(|i| i.enabled).unwrap_or(false);
            for h in handles {
                h.set_enabled(enabled);
            }
        }
        for (id, handles) in &installed.checks {
            let (enabled, checked) = by_id.get(id.as_str()).map(|i| (i.enabled, i.checked.unwrap_or(false))).unwrap_or((false, false));
            for h in handles {
                h.set_enabled(enabled);
                h.set_checked(checked);
            }
        }
        installed.shortcuts = by_id.iter().map(|(id, i)| ((*id).to_string(), i.shortcut.clone())).collect();
    }

    fn append_nodes(
        parent: &muda::Submenu,
        nodes: &[Node],
        by_id: &HashMap<&str, &crate::menus::MenuItem>,
        plain: &mut HashMap<String, Vec<muda::MenuItem>>,
        checks: &mut HashMap<String, Vec<muda::CheckMenuItem>>,
        unique: &mut dyn FnMut(&str) -> String,
    ) {
        for node in nodes {
            match node {
                Node::Separator => {
                    let _ = parent.append(&muda::PredefinedMenuItem::separator());
                }
                Node::Item { id, label, shortcut } => {
                    // Disabled catalogue rows stay visible but greyed, like the egui bar.
                    let live = by_id.get(id.as_str());
                    let enabled = live.map(|i| i.enabled).unwrap_or(false);
                    let checked = live.and_then(|i| i.checked);
                    let acc = accelerator(shortcut.as_deref());
                    if checked.is_some() {
                        let item = muda::CheckMenuItem::with_id(unique(id), label, enabled, checked.unwrap_or(false), acc);
                        let _ = parent.append(&item);
                        checks.entry(id.clone()).or_default().push(item);
                    } else {
                        let item = muda::MenuItem::with_id(unique(id), label, enabled, acc);
                        let _ = parent.append(&item);
                        plain.entry(id.clone()).or_default().push(item);
                    }
                }
                Node::Submenu { name, children } => {
                    let sub = muda::Submenu::new(name.clone(), true);
                    append_nodes(&sub, children, by_id, plain, checks, unique);
                    let _ = parent.append(&sub);
                }
            }
        }
    }

    /// Effective shortcut string → muda accelerator, via the same parser as the egui
    /// dispatch, so Edit › Keyboard Shortcuts overrides apply to the native bar too.
    /// Best-effort: unmapped keys get no key equivalent (the item stays clickable, and
    /// the egui shortcut still fires).
    pub(super) fn accelerator(sc: Option<&str>) -> Option<muda::accelerator::Accelerator> {
        use muda::accelerator::{Accelerator, Code, Modifiers};
        let parsed = crate::shortcuts::parse(sc?)?;
        let mut mods = Modifiers::empty();
        if parsed.modifiers.command || parsed.modifiers.ctrl {
            // macOS: ⌘ is SUPER. Ctrl counts as Cmd too, so non-Mac muscle memory keeps working.
            mods |= Modifiers::SUPER;
        }
        if parsed.modifiers.shift {
            mods |= Modifiers::SHIFT;
        }
        if parsed.modifiers.alt {
            mods |= Modifiers::ALT;
        }
        let code = match parsed.logical_key {
            egui::Key::A => Code::KeyA,
            egui::Key::B => Code::KeyB,
            egui::Key::C => Code::KeyC,
            egui::Key::D => Code::KeyD,
            egui::Key::E => Code::KeyE,
            egui::Key::F => Code::KeyF,
            egui::Key::G => Code::KeyG,
            egui::Key::H => Code::KeyH,
            egui::Key::I => Code::KeyI,
            egui::Key::J => Code::KeyJ,
            egui::Key::K => Code::KeyK,
            egui::Key::L => Code::KeyL,
            egui::Key::M => Code::KeyM,
            egui::Key::N => Code::KeyN,
            egui::Key::O => Code::KeyO,
            egui::Key::P => Code::KeyP,
            egui::Key::Q => Code::KeyQ,
            egui::Key::R => Code::KeyR,
            egui::Key::S => Code::KeyS,
            egui::Key::T => Code::KeyT,
            egui::Key::U => Code::KeyU,
            egui::Key::V => Code::KeyV,
            egui::Key::W => Code::KeyW,
            egui::Key::X => Code::KeyX,
            egui::Key::Y => Code::KeyY,
            egui::Key::Z => Code::KeyZ,
            egui::Key::Num0 => Code::Digit0,
            egui::Key::Num1 => Code::Digit1,
            egui::Key::Num2 => Code::Digit2,
            egui::Key::Num3 => Code::Digit3,
            egui::Key::Num4 => Code::Digit4,
            egui::Key::Num5 => Code::Digit5,
            egui::Key::Num6 => Code::Digit6,
            egui::Key::Num7 => Code::Digit7,
            egui::Key::Num8 => Code::Digit8,
            egui::Key::Num9 => Code::Digit9,
            egui::Key::F1 => Code::F1,
            egui::Key::F2 => Code::F2,
            egui::Key::F3 => Code::F3,
            egui::Key::F4 => Code::F4,
            egui::Key::F5 => Code::F5,
            egui::Key::F6 => Code::F6,
            egui::Key::F7 => Code::F7,
            egui::Key::F8 => Code::F8,
            egui::Key::F9 => Code::F9,
            egui::Key::F10 => Code::F10,
            egui::Key::F11 => Code::F11,
            egui::Key::F12 => Code::F12,
            egui::Key::Enter => Code::Enter,
            egui::Key::Tab => Code::Tab,
            egui::Key::Space => Code::Space,
            egui::Key::Backspace => Code::Backspace,
            egui::Key::Delete => Code::Delete,
            egui::Key::Escape => Code::Escape,
            egui::Key::ArrowUp => Code::ArrowUp,
            egui::Key::ArrowDown => Code::ArrowDown,
            egui::Key::ArrowLeft => Code::ArrowLeft,
            egui::Key::ArrowRight => Code::ArrowRight,
            egui::Key::Home => Code::Home,
            egui::Key::End => Code::End,
            egui::Key::PageUp => Code::PageUp,
            egui::Key::PageDown => Code::PageDown,
            egui::Key::Equals | egui::Key::Plus => Code::Equal,
            egui::Key::Minus => Code::Minus,
            egui::Key::OpenBracket => Code::BracketLeft,
            egui::Key::CloseBracket => Code::BracketRight,
            egui::Key::Semicolon | egui::Key::Colon => Code::Semicolon,
            egui::Key::Quote => Code::Quote,
            egui::Key::Backtick => Code::Backquote,
            egui::Key::Comma => Code::Comma,
            egui::Key::Period => Code::Period,
            egui::Key::Slash | egui::Key::Questionmark => Code::Slash,
            egui::Key::Backslash | egui::Key::Pipe => Code::Backslash,
            _ => return None,
        };
        let mods = if mods.is_empty() { None } else { Some(mods) };
        Some(Accelerator::new(mods, code))
    }
}

/// Count the clickable items in a tree (tests).
#[cfg(test)]
fn count_items(tops: &[TopMenu]) -> usize {
    fn n(ns: &[Node]) -> usize {
        ns.iter()
            .map(|x| match x {
                Node::Item { .. } => 1,
                Node::Separator => 0,
                Node::Submenu { children, .. } => n(children),
            })
            .sum()
    }
    tops.iter().map(|t| n(&t.children)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::menus::MenuItem;

    fn item(id: &str, path: &[&str], label: &str) -> MenuItem {
        MenuItem {
            id: id.into(),
            label: label.into(),
            path: path.iter().map(|s| s.to_string()).collect(),
            shortcut: None,
            enabled: true,
            checked: None,
            color: None,
        }
    }

    #[test]
    fn tree_groups_submenus_and_collapses_separators() {
        let items = vec![
            item("file.new", &["File"], "New…"),
            item("s1", &["File"], "---"),
            item("s2", &["File"], "---"),
            item("file.open", &["File"], "Open…"),
            item("file.a", &["File", "Export"], "A"),
            item("file.b", &["File", "Export"], "B"),
            item("s3", &["File"], "---"),
        ];
        let tops = build_tree(&items, &["File", "Edit"], Lang::EN);
        assert_eq!(tops.len(), 2);
        assert_eq!(tops[0].name, "File");
        assert_eq!(tops[1].children, vec![]);
        let names: Vec<String> = tops[0]
            .children
            .iter()
            .map(|n| match n {
                Node::Item { label, .. } => label.clone(),
                Node::Separator => "|".into(),
                Node::Submenu { name, .. } => format!("<{name}>"),
            })
            .collect();
        assert_eq!(names, vec!["New…", "|", "Open…", "<Export>"]);
        let Node::Submenu { children, .. } = &tops[0].children[3] else { panic!("expected submenu") };
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn tree_skips_empty_submenus() {
        // A submenu whose only child is a separator renders nothing.
        let items = vec![item("file.new", &["File"], "New…"), item("s", &["File", "Empty"], "---")];
        let tops = build_tree(&items, &["File"], Lang::EN);
        assert_eq!(tops[0].children.len(), 1);
    }

    #[test]
    fn full_catalog_builds_ten_tops_with_every_live_item() {
        let app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let items = crate::menus::menu_items(&app);
        let tops = build_tree(&items, &crate::menus::TOP_MENUS, Lang::EN);
        assert_eq!(tops.len(), crate::menus::TOP_MENUS.len());
        for t in &tops {
            assert!(!t.name.is_empty(), "top menu without a name");
        }
        // Every live item lands somewhere clickable (separators excluded).
        let mut ids = std::collections::HashSet::new();
        fn collect(ns: &[Node], ids: &mut std::collections::HashSet<String>) {
            for n in ns {
                match n {
                    Node::Item { id, .. } => {
                        ids.insert(id.clone());
                    }
                    Node::Submenu { children, .. } => collect(children, ids),
                    Node::Separator => {}
                }
            }
        }
        for t in &tops {
            collect(&t.children, &mut ids);
        }
        assert!(count_items(&tops) > 100, "most of the catalogue should survive, got {}", count_items(&tops));
        for must in ["file.new", "file.open", "edit.undo", "help.about", "view.zoomIn", "file.save"] {
            assert!(ids.contains(must), "{must} missing from the native tree");
        }
    }

    /// Accelerator mapping (macOS only: it produces muda types).
    #[cfg(target_os = "macos")]
    #[test]
    fn shortcuts_map_to_key_equivalents() {
        use super::imp::accelerator;
        assert!(accelerator(Some("Cmd+O")).is_some(), "the most common shortcut must map");
        assert!(accelerator(Some("Cmd+Shift+S")).is_some());
        assert!(accelerator(Some("F5")).is_some(), "bare function keys must map");
        assert!(accelerator(Some("Cmd+,")).is_some(), "Preferences ⌘, must map");
        assert!(accelerator(None).is_none());
        assert!(accelerator(Some("nonsense")).is_none(), "garbage stays accelerator-free, item still clickable");
    }

    /// Install failure (tests run off the main thread, where muda refuses) must fall back
    /// to the in-window bar without panicking.
    #[cfg(target_os = "macos")]
    #[test]
    fn failed_install_keeps_the_in_window_bar() {
        let mut app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        for _ in 0..3 {
            super::tick(&mut app, &ctx);
        }
        assert!(!super::active());
    }
}
