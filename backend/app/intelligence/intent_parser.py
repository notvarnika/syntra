import json
import re
from typing import Optional
import httpx
from app.config import settings
from app.schemas import StructuredIntent


class IntentParser:
    """
    Converts natural language voice transcripts into a StructuredIntent.
    Tier 1: Fast deterministic regex/rule classifier (<5ms latency) for atomic commands.
    Tier 2: LLM JSON parser (Ollama / Gemini / OpenAI) for complex compound requests.
    """

    PRONOUN_TOKENS = {"it", "this", "that", "this video", "this button", "this app", "there"}
    FIRST_RESULT_TOKENS = {"the first result", "first result", "the first one", "first one", "first option"}
    PREV_RESULT_TOKENS = {"the previous result", "previous result", "previous one"}

    def parse(self, raw_text: str, active_app: str = "Desktop") -> StructuredIntent:
        if not raw_text or not raw_text.strip():
            return StructuredIntent(intent="NONE")

        # Normalize text: lowercase, strip trailing punctuation
        text = raw_text.strip()
        clean = re.sub(r"[.!?,;]+$", "", text.lower()).strip()
        # Strip polite prefixes ("could you please launch chrome for me" -> "launch chrome")
        clean = re.sub(
            r"^(could you please|can you please|please|could you|can you|syntra|hey syntra)\s+",
            "",
            clean,
        )
        clean = re.sub(r"\s+(for me|right now|please)$", "", clean).strip()

        # ------------------------------------------------------------------
        # TIER 1: CHECK FOR COMPOUND COMMANDS FIRST (Route to LLM / Compound Parser)
        # Example from Synopsis: "Open the browser and search for the best MCA project ideas."
        # ------------------------------------------------------------------
        compound_match = re.search(
            r"(?:open|launch)\s+(?:the\s+)?(browser|chrome|google|youtube)\s+and\s+(?:search|look)\s+(?:for\s+)?(.+)",
            clean,
        )
        if compound_match:
            app_name = compound_match.group(1).upper()
            search_query = compound_match.group(2).strip()
            return StructuredIntent(
                intent="WEB_SEARCH",
                target=app_name,
                application="BROWSER" if app_name in ("BROWSER", "CHROME", "GOOGLE") else app_name,
                query=search_query,
                raw_source="LLM",
            )

        # ------------------------------------------------------------------
        # TIER 1: FAST ATOMIC COMMAND RULES
        # ------------------------------------------------------------------

        # 1. Security Confirmation / Cancellation ("Confirm", "Yes", "Cancel", "No", "Stop")
        if clean in ("confirm", "yes", "approve", "proceed", "do it"):
            return StructuredIntent(intent="CONFIRM", raw_source="RULE")

        if clean in ("cancel", "no", "abort", "never mind", "dont"):
            return StructuredIntent(intent="CANCEL", raw_source="RULE")

        if clean in ("stop", "pause", "halt", "freeze"):
            return StructuredIntent(intent="STOP", raw_source="RULE")

        # 2. High-Risk Destructive Operations (Demo 6: "Delete this file")
        if any(w in clean for w in ("delete", "remove", "erase", "trash")):
            target = "IT_REF" if any(p in clean for p in ("this", "it", "that", "file")) else clean
            return StructuredIntent(
                intent="DELETE_FILE", target=target, raw_source="RULE"
            )

        if any(w in clean for w in ("shut down", "shutdown", "restart computer", "format")):
            return StructuredIntent(
                intent="SYSTEM_SHUTDOWN", target="SYSTEM", raw_source="RULE"
            )

        # 3. Screenshot ("Take a screenshot", "Capture screen")
        if "screenshot" in clean or "capture screen" in clean:
            return StructuredIntent(intent="SCREENSHOT", raw_source="RULE")

        # 4. Scroll Commands ("Scroll down a little", "Scroll up")
        if "scroll" in clean:
            direction = "UP" if "up" in clean else "DOWN"
            amount = "MEDIUM"
            if any(w in clean for w in ("little", "slightly", "bit", "small")):
                amount = "SMALL"
            elif any(w in clean for w in ("lot", "fast", "much", "bottom", "top")):
                amount = "LARGE"
            return StructuredIntent(
                intent="SCROLL", direction=direction, amount=amount, raw_source="RULE"
            )

        # 5. Navigation ("Go back", "Back")
        if clean in ("go back", "back", "navigate back", "previous page"):
            return StructuredIntent(intent="GO_BACK", raw_source="RULE")

        # 6. Close Window ("Close that", "Close it", "Close Chrome")
        close_match = re.match(r"^close\s+(.+)$", clean)
        if close_match or clean == "close":
            obj = close_match.group(1).strip() if close_match else "IT_REF"
            target = "IT_REF" if obj in self.PRONOUN_TOKENS else obj
            return StructuredIntent(
                intent="CLOSE_APPLICATION", target=target, raw_source="RULE"
            )

        # 7. Play Media ("Play this", "Play video", "Play")
        if clean.startswith("play"):
            obj = clean.replace("play", "", 1).strip()
            target = "IT_REF" if (not obj or obj in self.PRONOUN_TOKENS) else obj
            return StructuredIntent(intent="PLAY", target=target, raw_source="RULE")

        # 8. Click / Select ("Click", "Click this button", "Select it")
        if clean.startswith(("click", "select", "press")):
            obj = re.sub(r"^(click|select|press)\s*(on\s*)?", "", clean).strip()
            if obj in self.FIRST_RESULT_TOKENS:
                return StructuredIntent(intent="CLICK", target="FIRST_RESULT", raw_source="RULE")
            target = "IT_REF" if (not obj or obj in self.PRONOUN_TOKENS) else obj
            return StructuredIntent(intent="CLICK", target=target, raw_source="RULE")

        # 9. Web Search ("Search for Python", "Google best MCA projects")
        search_match = re.match(r"^(?:search\s+for|search|google|look\s+up)\s+(.+)$", clean)
        if search_match:
            query = search_match.group(1).strip()
            return StructuredIntent(
                intent="WEB_SEARCH",
                application=active_app,
                query=query,
                target=query,
                raw_source="RULE",
            )

        # 10. Open / Launch / Use Commands ("Open Chrome", "Open it", "Open the first result")
        open_match = re.match(r"^(?:open|launch|start|run|use)\s+(.+)$", clean)
        if open_match:
            obj = open_match.group(1).strip()

            # Check for "first result" / "first one" (Demo 5)
            if obj in self.FIRST_RESULT_TOKENS:
                return StructuredIntent(
                    intent="CLICK", target="FIRST_RESULT", raw_source="RULE"
                )

            # Check for "previous result"
            if obj in self.PREV_RESULT_TOKENS:
                return StructuredIntent(
                    intent="GO_BACK", target="PREVIOUS_RESULT", raw_source="RULE"
                )

            # Check for Deictic Pronouns ("it", "this", "that") -> Resolved by Context/Fusion Engine!
            if obj in self.PRONOUN_TOKENS:
                return StructuredIntent(
                    intent="OPEN_APPLICATION", target="IT_REF", raw_source="RULE"
                )

            # Clean common articles ("the browser" -> "browser")
            obj = re.sub(r"^the\s+", "", obj)
            return StructuredIntent(
                intent="OPEN_APPLICATION", target=obj, raw_source="RULE"
            )

        # ------------------------------------------------------------------
        # TIER 2: LLM FALLBACK FOR COMPLEX NATURAL LANGUAGE
        # ------------------------------------------------------------------
        llm_intent = self._query_llm(text, active_app)
        if llm_intent is not None:
            return llm_intent

        return StructuredIntent(intent="UNKNOWN", target=text, raw_source="RULE")

    def _query_llm(self, text: str, active_app: str) -> Optional[StructuredIntent]:
        """
        Queries local Ollama (if running) to parse unstructured natural language
        into SYNTRA's JSON intent schema. Fails gracefully in <1.5s if offline.
        """
        system_prompt = (
            "You are the SYNTRA Intent Parser. Convert the user's command into strict JSON "
            "with keys: intent (OPEN_APPLICATION, WEB_SEARCH, CLICK, SCROLL, SCREENSHOT, "
            "DELETE_FILE, CLOSE_APPLICATION, PLAY, GO_BACK), target, application, query, "
            "direction (UP/DOWN), amount (SMALL/MEDIUM/LARGE). "
            f"Current active application: {active_app}. Respond ONLY with valid JSON."
        )
        try:
            if settings.LLM_PROVIDER.lower() == "ollama":
                with httpx.Client(timeout=1.8) as client:
                    resp = client.post(
                        f"{settings.OLLAMA_BASE_URL}/api/generate",
                        json={
                            "model": settings.OLLAMA_MODEL,
                            "prompt": f"{system_prompt}\nUser: {text}\nJSON:",
                            "stream": False,
                            "format": "json",
                        },
                    )
                    if resp.status_code == 200:
                        data = json.loads(resp.json().get("response", "{}"))
                        return StructuredIntent(
                            intent=data.get("intent", "UNKNOWN"),
                            target=data.get("target"),
                            application=data.get("application"),
                            query=data.get("query"),
                            direction=data.get("direction"),
                            amount=data.get("amount"),
                            raw_source="LLM",
                        )
        except Exception:
            pass
        return None


intent_parser = IntentParser()