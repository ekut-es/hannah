import os
import time
import glob
import numpy as np
import argparse
import curses

# === CONFIGURATION ===
SCAN_INTERVAL = 5  # How often (in seconds) to rescan directories
DEFAULT_MAX_FILES = 600  # Default max files per experiment
STALE_THRESHOLD = 2 * 3600  # 2 hours in seconds
MEDIAN_WINDOW = 50  # Use last 50 timestamps for median calculation

def find_experiments(base_dir):
    """Finds all experiments with a 'performance_data' folder."""
    experiment_paths = {}
    for root, dirs, _ in os.walk(base_dir):
        if "performance_data" in dirs:
            exp_name = os.path.dirname(root).split("/")[-1]
            exp_path = os.path.join(root, "performance_data")
            experiment_paths[exp_name] = exp_path
    return experiment_paths

def count_json_files(path):
    """Returns a list of JSON file creation times in the given path."""
    json_files = glob.glob(os.path.join(path, "*.json"))
    timestamps = sorted(os.path.getctime(f) for f in json_files)  # Get file creation times
    return len(json_files), timestamps

def calculate_time_diffs(timestamps):
    """Computes different estimates based on past timestamps."""
    if len(timestamps) < 2:
        return np.inf, np.inf, np.inf, np.inf, np.inf  # Not enough data
    diffs = np.diff(timestamps)
    min_diff = np.min(diffs) if diffs.size > 0 else np.inf
    max_diff = np.max(diffs) if diffs.size > 0 else np.inf
    median_diff_total = np.median(diffs) if diffs.size > 0 else np.inf
    recent_diffs = diffs[-MEDIAN_WINDOW:] if len(diffs) >= MEDIAN_WINDOW else diffs
    recent_diffs.sort()
    # if recent_diffs[-1] - recent_diffs[-2] > np.mean(recent_diffs) * 3:
    recent_diffs = recent_diffs[:-1]
    median_diff_window = np.mean(recent_diffs) if recent_diffs.size > 0 else np.inf
    return min_diff, max_diff, median_diff_total, median_diff_window, np.median(diffs)

def format_seconds(seconds):
    """Manually formats seconds as HH:MM:SS."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def create_progress_bar(current, total, length=20):
    """Creates a progress bar using Unicode block characters."""
    if total == 0:
        return "░" * length, "0%"
    filled_length = int(length * current / total)
    empty_length = length - filled_length
    bar = "█" * filled_length + "░" * empty_length
    percentage = f"{(current / total) * 100:.1f}%"
    return bar, percentage

def render_ui(stdscr, base_dir, max_files):
    """Handles the terminal UI with scrolling support."""
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(SCAN_INTERVAL * 1000)
    scroll_y, scroll_x = 0, 0
    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        max_visible_rows = height - 2
        stdscr.addstr(0, 0, "📊 Experiment Progress Monitor (Use ↑/↓ to scroll, 'q' to quit)", curses.A_BOLD)
        try:
            experiments = find_experiments(base_dir)
        except Exception as e:
            stdscr.addstr(2, 0, f"⚠️ Error scanning experiments: {str(e)}")
            experiments = {}
        max_name_length = max((len(name) for name in experiments), default=0)
        padding = max_name_length + 2
        active_experiments, stale_experiments = [], []
        current_time = time.time()
        row_data = []
        for exp_name, path in experiments.items():
            num_files, timestamps = count_json_files(path)
            min_diff, max_diff, median_diff_total, median_diff_window, median_iter = calculate_time_diffs(timestamps)
            last_file_time = max(timestamps) if timestamps else 0
            time_since_last_file = format_seconds(current_time - last_file_time) if last_file_time else "N/A"
            is_stale = (current_time - last_file_time) > STALE_THRESHOLD
            remaining_files = max(0, max_files - num_files)
            remaining_times = {
                "Min": format_seconds(remaining_files * min_diff) if min_diff != np.inf else "N/A",
                "Max": format_seconds(remaining_files * max_diff) if max_diff != np.inf else "N/A",
                "Median (Total)": f"{format_seconds(remaining_files * median_diff_total)} ({format_seconds(median_diff_total)})" if median_diff_total != np.inf else "N/A",
                "Median (Window)": f"{format_seconds(remaining_files * median_diff_window)} ({format_seconds(median_diff_window)})" if median_diff_window != np.inf else "N/A",
            }
            exp_info = {"name": exp_name, "num_files": num_files, "remaining_times": remaining_times, "last_file": time_since_last_file}
            (stale_experiments if is_stale else active_experiments).append(exp_info)

        active_experiments.sort(key=lambda x: -x["num_files"])
        stale_experiments.sort(key=lambda x: -x["num_files"])

        row_data.append("🟢 Active Experiments\n")
        for exp in active_experiments:
            bar, progress_str = create_progress_bar(exp["num_files"], max_files)
            row_data.append(f"{exp['name'].ljust(padding)} [{bar}] {progress_str}\n")
            row_data.append(f"{' ' * padding} 🕒 Last File: {exp['last_file']}\n")
            row_data.append(f"{' ' * padding} ⏳ Min: {exp['remaining_times']['Min']} | Max: {exp['remaining_times']['Max']}\n")
            row_data.append(f"{' ' * padding} ⚖️ Median (Total): {exp['remaining_times']['Median (Total)']} | Mean (Window): {exp['remaining_times']['Median (Window)']}\n\n")
        row_data.append("⚠️ Stale Experiments\n")
        for exp in stale_experiments:
            bar, progress_str = create_progress_bar(exp["num_files"], max_files)
            row_data.append(f"{exp['name'].ljust(padding)} [{bar}] {progress_str}\n")
            row_data.append(f"{' ' * padding} 🕒 Last File: {exp['last_file']}\n")
            row_data.append(f"{' ' * padding} ⏳ Min: {exp['remaining_times']['Min']} | Max: {exp['remaining_times']['Max']}\n")
            row_data.append(f"{' ' * padding} ⚖️ Median (Total): {exp['remaining_times']['Median (Total)']} | Mean (Window): {exp['remaining_times']['Median (Window)']}\n\n")

        max_scroll = max(0, len(row_data) - max_visible_rows)
        scroll_y = min(max(0, scroll_y), max_scroll)
        for i in range(min(len(row_data) - scroll_y, max_visible_rows)):
            try:
                stdscr.addstr(i + 2, max(0, -scroll_x), row_data[scroll_y + i][:width + scroll_x])
            except curses.error:
                pass
        stdscr.refresh()
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key == curses.KEY_DOWN:
            scroll_y += 1
        elif key == curses.KEY_UP:
            scroll_y -= 1


def main():
    parser = argparse.ArgumentParser(description="Monitor experiment progress.")
    parser.add_argument("base_dir", nargs="?", default=os.getcwd(), help="Base directory containing experiments.")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, help="Max expected files.")
    args = parser.parse_args()
    curses.wrapper(lambda stdscr: render_ui(stdscr, args.base_dir, args.max_files))


if __name__ == "__main__":
    main()
