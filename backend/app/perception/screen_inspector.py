from typing import Optional
import pyautogui
import pygetwindow as gw


class ScreenInspector:
    """
    Maps a 2D screen coordinate (X, Y) to a named UI target or application.
    1. Checks actual OS window bounding boxes via pygetwindow.
    2. Maps to the SYNTRA Synopsis Virtual Dock (Chrome | YouTube | VS Code | Spotify)
       when looking at the desktop or top dock bar so Demo 3 & Demo 4 work reliably.
    """

    def __init__(self) -> None:
        self.screen_w, self.screen_h = pyautogui.size()

    def get_active_window_title(self) -> str:
        """Returns a clean application name for the currently focused OS window."""
        try:
            win = gw.getActiveWindow()
            if win and win.title:
                return self._normalize_window_title(win.title)
        except Exception:
            pass
        return "Desktop"

    def resolve_target_at(self, x: int, y: int) -> str:
        """
        Determines which application or target sits at screen pixel (x, y).
        Matches the synopsis layout: Chrome | YouTube | VS Code | Spotify.
        """
        if y <= int(self.screen_h * 0.28):
            quarter = self.screen_w / 4.0
            if x < quarter:
                return "Chrome"
            elif x < quarter * 2:
                return "YouTube"
            elif x < quarter * 3:
                return "VS Code"
            else:
                return "Spotify"

        try:
            windows = gw.getWindowsAt(x, y)
            for win in windows:
                title = (win.title or "").strip()
                if not title or title.lower() in ("program manager", "settings", "syntra"):
                    continue
                normalized = self._normalize_window_title(title)
                if normalized not in ("Desktop", "Terminal"):
                    return normalized
        except Exception:
            pass

        quarter = self.screen_w / 4.0
        if x < quarter:
            return "Chrome"
        elif x < quarter * 2:
            return "YouTube"
        elif x < quarter * 3:
            return "VS Code"
        return "Spotify"

    @staticmethod
    def _normalize_window_title(raw_title: str) -> str:
        low = raw_title.lower()
        if any(k in low for k in ("cmd.exe", "command prompt", "powershell", "windows terminal", "python -c")):
            return "Terminal"
        if "youtube" in low:
            return "YouTube"
        if "chrome" in low or "google" in low or "edge" in low or "firefox" in low:
            return "Chrome"
        if "visual studio code" in low or "vs code" in low or ".py" in low or ".tsx" in low:
            return "VS Code"
        if "spotify" in low:
            return "Spotify"
        if "explorer" in low or "file" in low:
            return "File Explorer"
        if "notepad" in low:
            return "Notepad"
        parts = [p.strip() for p in raw_title.split("-") if p.strip()]
        return parts[-1][:30] if parts else "Desktop"


screen_inspector = ScreenInspector()