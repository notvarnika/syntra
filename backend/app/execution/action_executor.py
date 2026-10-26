import os
import platform
import shutil
import subprocess
import time
import urllib.parse
import webbrowser
from datetime import datetime
from typing import Optional, Tuple
import pyautogui
import pygetwindow as gw
from app.config import DATA_DIR
from app.schemas import StructuredIntent

# Configure PyAutoGUI for responsive real-time HCI control
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01


class ActionExecutor:
    """
    Executes validated StructuredIntents on the host Operating System (Windows / Linux)
    using PyAutoGUI, webbrowser, and subprocess hooks.
    """

    def __init__(self) -> None:
        self.os_name = platform.system().lower()
        self.screenshots_dir = DATA_DIR / "screenshots"
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._is_grabbing: bool = False
        self._last_click_time: float = 0.0

    def move_cursor(self, x: int, y: int) -> None:
        """Moves the system mouse pointer smoothly to (x, y) without blocking."""
        try:
            pyautogui.moveTo(x, y, _pause=False)
        except pyautogui.FailSafeException:
            pass

    def scroll_direct(self, delta: float) -> None:
        """Executes immediate continuous scrolling from the Two-Fingers gesture."""
        if abs(delta) >= 1.0:
            pyautogui.scroll(int(delta), _pause=False)

    def execute(
        self,
        intent: StructuredIntent,
        gaze_xy: Optional[Tuple[int, int]] = None,
        cursor_xy: Optional[Tuple[int, int]] = None,
    ) -> Tuple[bool, str]:
        """
        Executes a high-level StructuredIntent and returns (success, details_message).
        """
        action = intent.intent.upper()
        target = (intent.target or "").strip()

        try:
            # 1. OPEN APPLICATION OR URL (Demo 1, Demo 2, Demo 4, Demo 5)
            if action == "OPEN_APPLICATION":
                return self._open_application(target)

            # 2. WEB SEARCH (Demo 5 & LLM Complex Query Example)
            if action == "WEB_SEARCH":
                query = intent.query or target
                return self._perform_web_search(query)

            # 3. CLICK / SELECT TARGET ON CURRENT SCREEN OR TAB (Gaze / Gesture / Context)
            if action in ("CLICK", "SELECT"):
                now = time.time()
                if now - self._last_click_time < 0.45:
                    return True, "Click debounced"
                self._last_click_time = now

                # Special case: Demo 5 ("Open the first result")
                if target.upper() in ("FIRST_RESULT", "FIRST_ONE"):
                    return self._open_first_search_result(gaze_xy)

                # Focus the real window/tab currently under the user's gaze or target name
                if target and target.lower() not in ("desktop", "none", "screen"):
                    try:
                        matching_wins = [
                            w for w in gw.getAllWindows()
                            if w.title and target.lower() in w.title.lower() and not w.isMinimized
                        ]
                        if matching_wins:
                            matching_wins[0].activate()
                    except Exception:
                        pass

                # Move and click directly at the live Gaze or Gesture coordinate on the current screen
                click_pos = cursor_xy or gaze_xy
                if click_pos and click_pos != (0, 0):
                    pyautogui.moveTo(click_pos[0], click_pos[1], duration=0.1)
                    pyautogui.click(click_pos[0], click_pos[1])
                    return True, f"Selected '{target or 'Screen'}' at ({click_pos[0]}, {click_pos[1]})"

                pyautogui.click()
                return True, f"Clicked active element in '{target or 'Screen'}'"

            # 4. SCROLL (Voice "Scroll down a little" or Gesture)
            if action == "SCROLL":
                direction = (intent.direction or "DOWN").upper()
                amount_str = (intent.amount or "MEDIUM").upper()
                clicks = 250 if amount_str == "SMALL" else (700 if amount_str == "LARGE" else 450)
                if direction == "DOWN":
                    clicks = -clicks
                pyautogui.scroll(clicks)
                return True, f"Scrolled {direction} ({amount_str})"

            # 5. SCREENSHOT ("Take a screenshot")
            if action == "SCREENSHOT":
                filename = f"syntra_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = self.screenshots_dir / filename
                pyautogui.screenshot(str(filepath))
                return True, f"Screenshot saved to {filename}"

            # 6. MEDIA & PLAYBACK CONTROL ("Play this", "Pause", "Stop")
            if action in ("PLAY_MEDIA", "PAUSE_MEDIA", "PLAY", "PAUSE"):
                if gaze_xy and action == "PLAY":
                    pyautogui.click(gaze_xy[0], gaze_xy[1])
                else:
                    pyautogui.press("playpause")
                return True, f"Executed {action}"

            if action == "STOP":
                if self._is_grabbing:
                    pyautogui.mouseUp()
                    self._is_grabbing = False
                return True, "Automation / active hold stopped"

            # 7. GRAB / HOLD (Fist Gesture)
            if action == "GRAB":
                if not self._is_grabbing:
                    pyautogui.mouseDown()
                    self._is_grabbing = True
                return True, "Mouse grab held"

            # 8. NAVIGATION & WINDOW MANAGEMENT ("Go back", "Close that")
            if action == "GO_BACK":
                pyautogui.hotkey("alt", "left")
                return True, "Navigated back"

            if action == "CLOSE_APPLICATION":
                pyautogui.hotkey("alt", "f4")
                return True, f"Closed active window ({target or 'current'})"

            # 9. HIGH-RISK FILE DELETION (Demo 6 — Only reached after Security Layer approval)
            if action == "DELETE_FILE":
                pyautogui.press("delete")
                return True, f"Sent delete command for target: {target or 'selected item'}"

            return False, f"Unrecognized action: {action}"

        except Exception as exc:
            return False, f"Execution error: {str(exc)}"

    def _open_application(self, target_name: str) -> Tuple[bool, str]:
        name = target_name.lower().strip()
        if not name:
            return False, "No application target specified"

        # Smart Focus: If a window matching the app name is already open, bring it to foreground
        try:
            open_windows = gw.getWindowsWithTitle(name)
            for win in open_windows:
                if win.title and win.title.lower() not in ("desktop", "program manager") and not win.isMinimized:
                    try:
                        win.activate()
                        return True, f"Brought existing '{win.title}' to foreground"
                    except Exception:
                        pass
        except Exception:
            pass

        if "youtube" in name:
            webbrowser.open("https://www.youtube.com")
            return True, "Launched YouTube"

        if "spotify" in name:
            if shutil.which("spotify"):
                subprocess.Popen(["spotify"], shell=True)
            else:
                webbrowser.open("https://open.spotify.com")
            return True, "Launched Spotify"

        if any(k in name for k in ("chrome", "browser", "google")):
            if "win" in self.os_name:
                try:
                    os.startfile("chrome")
                    return True, "Launched Google Chrome"
                except Exception:
                    webbrowser.open("https://www.google.com")
                    return True, "Launched Default Browser"
            else:
                browser_cmd = shutil.which("google-chrome") or shutil.which("chromium")
                if browser_cmd:
                    subprocess.Popen([browser_cmd])
                else:
                    webbrowser.open("https://www.google.com")
                return True, "Launched Browser"

        if any(k in name for k in ("vs code", "vscode", "code", "visual studio")):
            subprocess.Popen(["code"], shell=True)
            return True, "Launched VS Code"

        if "notepad" in name:
            subprocess.Popen(
                ["notepad.exe"] if "win" in self.os_name else ["gedit"], shell=True
            )
            return True, "Launched Notepad"

        if "calc" in name:
            subprocess.Popen(
                ["calc.exe"] if "win" in self.os_name else ["gnome-calculator"],
                shell=True,
            )
            return True, "Launched Calculator"

        webbrowser.open(
            f"https://www.google.com/search?q={urllib.parse.quote(target_name)}"
        )
        return True, f"Opened {target_name} in browser"

    def _perform_web_search(self, query: str) -> Tuple[bool, str]:
        clean_query = query.strip() or "Python"
        encoded = urllib.parse.quote(clean_query)
        url = f"https://www.google.com/search?q={encoded}"
        webbrowser.open(url)
        return True, f"Searched web for '{clean_query}'"

    def _open_first_search_result(
        self, gaze_xy: Optional[Tuple[int, int]]
    ) -> Tuple[bool, str]:
        screen_w, screen_h = pyautogui.size()
        target_x = int(screen_w * 0.28)
        target_y = int(screen_h * 0.34)
        if (
            gaze_xy
            and (0.15 * screen_w < gaze_xy[0] < 0.75 * screen_w)
            and (0.20 * screen_h < gaze_xy[1] < 0.85 * screen_h)
        ):
            target_x, target_y = gaze_xy

        pyautogui.moveTo(target_x, target_y, duration=0.2)
        pyautogui.click(target_x, target_y)
        return True, f"Opened first search result at ({target_x}, {target_y})"


action_executor = ActionExecutor()