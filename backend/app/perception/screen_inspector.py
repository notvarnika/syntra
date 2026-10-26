import pyautogui
import pygetwindow as gw


class ScreenInspector:
    """
    Maps a 2D screen coordinate (X, Y) purely to the real OS window or active tab
    currently visible at that pixel, with zero hardcoded screen percentages.
    """

    # Hidden Windows OS background processes to ignore
    IGNORED_TITLES = {
        "",
        "program manager",
        "windows input experience",
        "microsoft text input application",
        "settings",
        "syntra",
        "taskbar",
    }

    def __init__(self) -> None:
        self.screen_w, self.screen_h = pyautogui.size()

    def get_active_window_title(self) -> str:
        """Returns the exact application or tab name of the currently focused OS window."""
        try:
            win = gw.getActiveWindow()
            if win and win.title:
                clean = self._normalize_window_title(win.title)
                if clean:
                    return clean
        except Exception:
            pass
        return "Desktop"

    def resolve_target_at(self, x: int, y: int) -> str:
        """
        Finds the actual OS window or open tab sitting at pixel (x, y).
        """
        # 1. Check all visible OS windows stacked at coordinate (x, y) in Z-order (top to bottom)
        try:
            windows = gw.getWindowsAt(x, y)
            for win in windows:
                if not win.visible or win.isMinimized:
                    continue
                raw_title = (win.title or "").strip()
                if raw_title.lower() in self.IGNORED_TITLES:
                    continue

                normalized = self._normalize_window_title(raw_title)
                if normalized and normalized not in ("Desktop", "Windows Input Experience"):
                    return normalized
        except Exception:
            pass

        # 2. Fallback to whichever window is currently active in the foreground
        return self.get_active_window_title()

    @staticmethod
    def _normalize_window_title(raw_title: str) -> str:
        """
        Extracts the active tab or application name from the OS window title.
        Example: 'YouTube - Google Chrome' -> 'YouTube'
        Example: 'main.py - syntra - Visual Studio Code' -> 'VS Code (main.py)'
        """
        low = raw_title.lower().strip()
        if low in ScreenInspector.IGNORED_TITLES:
            return ""

        if any(k in low for k in ("cmd.exe", "command prompt", "powershell", "windows terminal", "python -c")):
            return "Terminal"

        # Keep specific browser tab awareness before generic Chrome
        if "youtube" in low:
            return "YouTube"
        if "spotify" in low:
            return "Spotify"
        if "whatsapp" in low:
            return "WhatsApp"
        if "github" in low:
            return "GitHub"

        # If it's a browser tab, return the actual tab title + Browser
        if any(b in low for b in ("google chrome", "microsoft edge", "firefox", "brave")):
            parts = [p.strip() for p in raw_title.split("-") if p.strip()]
            tab_name = parts[0][:28] if len(parts) > 1 else "Chrome"
            return tab_name if tab_name.lower() != "new tab" else "Chrome"

        if "visual studio code" in low or "vs code" in low or ".py" in low or ".tsx" in low:
            return "VS Code"
        if "explorer" in low:
            return "File Explorer"
        if "notepad" in low:
            return "Notepad"

        parts = [p.strip() for p in raw_title.split("-") if p.strip()]
        return parts[0][:30] if parts else "Desktop"


screen_inspector = ScreenInspector()