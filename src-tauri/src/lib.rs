// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
fn set_window_visible(window: tauri::Window, visible: bool) {
    if visible {
        window.show().unwrap();
    } else {
        window.hide().unwrap();
    }
}

#[tauri::command]
fn set_click_through(window: tauri::Window, ignore: bool) {
    window.set_ignore_cursor_events(ignore).unwrap_or_else(|e| eprintln!("Failed to set ignore cursor events: {}", e));
}

#[tauri::command]
fn set_always_on_top(window: tauri::Window, always_on_top: bool) {
    window.set_always_on_top(always_on_top).unwrap_or_else(|e| eprintln!("Failed to set always on top: {}", e));
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, set_window_visible, set_click_through, set_always_on_top])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
