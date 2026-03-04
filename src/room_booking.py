"""RBS (Room Booking System) client for authenticated scraping of room data."""
import json as _json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup


class RBSClient:
    """Encapsulates all interactions with the CIHE Room Booking System."""

    BASE_URL = "https://rbs.cihe.edu.hk"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self._username: Optional[str] = None
        self._logged_in: bool = False
        self._rooms_cache: Optional[List[Dict]] = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def login(self, username: str, password: str) -> bool:
        """Authenticate with RBS. Returns True on success."""
        resp = self.session.get(self.BASE_URL, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        token_input = soup.find("input", {"name": "__RequestVerificationToken"})
        if not token_input:
            raise RuntimeError("Could not find __RequestVerificationToken on login page")

        token = token_input["value"]

        login_payload = {
            "UserCode": username,
            "Password": password,
            "__RequestVerificationToken": token,
        }
        login_resp = self.session.post(
            self.BASE_URL,
            data=login_payload,
            timeout=15,
            allow_redirects=True,
        )
        login_resp.raise_for_status()

        if "BookingDashboard" in login_resp.url or "Dashboard" in login_resp.url:
            self._username = username
            self._logged_in = True
            self._rooms_cache = None
            return True

        if self._has_auth_cookies():
            self._username = username
            self._logged_in = True
            self._rooms_cache = None
            return True

        return False

    def logout(self):
        """Clear session state."""
        self.session.cookies.clear()
        self._username = None
        self._logged_in = False
        self._rooms_cache = None

    @property
    def is_authenticated(self) -> bool:
        return self._logged_in and self._username is not None

    @property
    def username(self) -> Optional[str]:
        return self._username

    def _has_auth_cookies(self) -> bool:
        cookie_names = [c.name for c in self.session.cookies]
        return any(
            name in cookie_names
            for name in (".AspNetCore.Cookies", ".AspNet.ApplicationCookie", "ASP.NET_SessionId")
        )

    # ------------------------------------------------------------------
    # Room listing
    # ------------------------------------------------------------------

    _ROOM_API_PATHS = [
        "/api/Room/GetAll",
        "/api/rooms",
        "/Scheduler/GetResources",
        "/BookingDashboard/GetRooms",
        "/api/Scheduler/Resources",
        "/api/BookingDashboard/Resources",
        "/Room/GetAll",
    ]

    _FILTER_SELECT_IDS = {"site", "area", "roomtype", "floor", "equipment"}

    def get_rooms(self, force_refresh: bool = False) -> List[Dict]:
        """Discover individual rooms using multiple strategies. Results are cached."""
        if self._rooms_cache and not force_refresh:
            return self._rooms_cache

        resp = self.session.get(f"{self.BASE_URL}/BookingDashboard", timeout=15)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        rooms: List[Dict] = []

        # Strategy 1: Parse room cards from the page text.
        # The BookingDashboard renders cards like "CBCC - Classroom (302)"
        # with Room Type, Capacity, etc. underneath each title.
        if not rooms:
            rooms = self._parse_room_cards(html)

        # Strategy 2: Scheduler grid elements (Kendo, Telerik, FullCalendar)
        if not rooms:
            rooms = self._parse_rooms_from_scheduler_grid(soup)

        # Strategy 3: JSON embedded in <script> tags
        if not rooms:
            rooms = self._parse_rooms_from_scripts(html)

        # Strategy 4: Probe common REST endpoints
        if not rooms:
            rooms = self._probe_room_api_endpoints()

        # Strategy 5: data-* attribute elements
        if not rooms:
            rooms = self._parse_room_data_attrs(soup)

        # Strategy 6: Extract room types from the roomtype <select> only
        if not rooms:
            rooms = self._parse_room_types_select(soup)

        self._rooms_cache = rooms
        return rooms

    def get_room_types(self) -> List[Dict]:
        """Return room-type categories from the roomtype filter dropdown."""
        resp = self.session.get(f"{self.BASE_URL}/BookingDashboard", timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return self._parse_room_types_select(soup)

    def _parse_room_types_select(self, soup) -> List[Dict]:
        """Parse only the 'roomtype' <select> for room type options."""
        rooms = []
        select = soup.find("select", id="roomtype") or soup.find("select", {"name": "roomtype"})
        if not select:
            return rooms
        for opt in select.find_all("option"):
            val = opt.get("value", "").strip()
            label = opt.get_text(strip=True)
            if not val or val in ("all", "", "0", "-1", "null", "-"):
                continue
            rooms.append({
                "id": val,
                "name": label,
                "area": "", "floor": "", "capacity": "", "type": val,
            })
        return rooms

    def _parse_rooms_from_scheduler_grid(self, soup) -> List[Dict]:
        """Extract individual rooms from the scheduler widget's rendered HTML."""
        rooms = []

        scheduler_selectors = [
            ".k-scheduler .k-scheduler-header th",
            ".k-scheduler .k-resource",
            ".scheduler-resource",
            ".rsResource",
            "[class*='resource'] a",
            "[class*='resource'] span",
            "[class*='room-name']",
            "[class*='room-label']",
            ".fc-resource",
            "td.room-name",
            "th.room-name",
            "div.room-header",
        ]
        for selector in scheduler_selectors:
            for el in soup.select(selector):
                text = el.get_text(strip=True)
                if text and len(text) < 100:
                    rid = el.get("data-id") or el.get("data-resource-id") or ""
                    rooms.append({
                        "id": str(rid) if rid else text,
                        "name": text,
                        "area": "", "floor": "", "capacity": "", "type": "",
                    })

        for link in soup.find_all("a", href=True):
            href = link["href"]
            text = link.get_text(strip=True)
            m = re.search(r'/(?:Scheduler|Room|BookingDashboard)/?\??.*?(?:room|id)=(\w+)', href, re.I)
            if m and text and len(text) < 80:
                rooms.append({
                    "id": m.group(1),
                    "name": text,
                    "area": "", "floor": "", "capacity": "", "type": "",
                })

        return self._dedup_rooms(rooms)

    _CARD_TITLE_RE = re.compile(
        r'^(.+?)\s*-\s*(.+)\(([A-Za-z]?\d{3,4}[A-Za-z]?)\)\s*$'
    )

    def _parse_room_cards(self, html: str) -> List[Dict]:
        """Parse room cards rendered on the BookingDashboard page.

        Each card has a title like ``CBCC - Classroom (302)`` followed by
        labelled fields (Room Type, Capacity, Equipment, Desk Type, Remarks).
        The room number is always the last parenthesized digit group in the title.
        """
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["script", "style"]):
            tag.decompose()

        body = soup.find("body")
        if not body:
            return []

        text = body.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        rooms: List[Dict] = []
        seen_ids: set = set()

        for i, line in enumerate(lines):
            m = self._CARD_TITLE_RE.match(line)
            if not m:
                continue

            area = m.group(1).strip()
            room_type = m.group(2).strip().rstrip()
            room_number = m.group(3).strip()

            if room_number in seen_ids:
                continue
            seen_ids.add(room_number)

            if room_type.endswith(")"):
                idx = room_type.rfind("(")
                if idx != -1:
                    room_type = room_type[:idx].strip()

            capacity = ""
            for j in range(i + 1, min(i + 12, len(lines))):
                if lines[j].lower().startswith("capacity") and j + 1 < len(lines):
                    cap_val = lines[j + 1].strip()
                    if cap_val.isdigit():
                        capacity = cap_val
                    break

            rooms.append({
                "id": room_number,
                "name": f"{area} - {room_type} ({room_number})",
                "area": area,
                "floor": "",
                "capacity": capacity,
                "type": room_type,
            })

        if rooms:
            print(f"[RBS] Parsed {len(rooms)} room cards from BookingDashboard")
        return rooms

    def _parse_rooms_from_scripts(self, html: str) -> List[Dict]:
        """Extract room data from JSON objects embedded inside <script> tags."""
        rooms: List[Dict] = []
        soup = BeautifulSoup(html, "html.parser")

        for script in soup.find_all("script"):
            text = script.string
            if not text:
                continue

            json_objects = re.findall(r'\[[\s\S]*?\{[\s\S]*?"(?:[Ii]d|[Nn]ame|[Rr]oom)"[\s\S]*?\}[\s\S]*?\]', text)
            for blob in json_objects:
                try:
                    arr = _json.loads(blob)
                    if not isinstance(arr, list):
                        continue
                    for item in arr:
                        if not isinstance(item, dict):
                            continue
                        room = self._dict_to_room(item)
                        if room:
                            rooms.append(room)
                except (_json.JSONDecodeError, ValueError):
                    continue

            js_assignments = re.findall(
                r'(?:resources|rooms|roomList|roomData|dataSource)\s*[:=]\s*(\[[\s\S]*?\])(?:\s*[;,}])',
                text, re.IGNORECASE,
            )
            for blob in js_assignments:
                try:
                    arr = _json.loads(blob)
                    if isinstance(arr, list):
                        for item in arr:
                            if isinstance(item, dict):
                                room = self._dict_to_room(item)
                                if room:
                                    rooms.append(room)
                except (_json.JSONDecodeError, ValueError):
                    continue

        return self._dedup_rooms(rooms)

    def _probe_room_api_endpoints(self) -> List[Dict]:
        """Try common REST endpoints that might return room JSON."""
        for path in self._ROOM_API_PATHS:
            try:
                resp = self.session.get(f"{self.BASE_URL}{path}", timeout=10)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                items = data if isinstance(data, list) else data.get("data", data.get("rooms", data.get("resources", [])))
                if not isinstance(items, list) or not items:
                    continue
                rooms = []
                for item in items:
                    if isinstance(item, dict):
                        room = self._dict_to_room(item)
                        if room:
                            rooms.append(room)
                if rooms:
                    print(f"[RBS] Found {len(rooms)} rooms via {path}")
                    return rooms
            except Exception:
                continue
        return []

    def _parse_room_data_attrs(self, soup) -> List[Dict]:
        """Look for elements with data-room-id or similar attributes."""
        rooms = []
        for el in soup.select("[data-room-id], [data-resource-id], .room-item, .room-card, tr[data-id]"):
            room_id = el.get("data-room-id") or el.get("data-resource-id") or el.get("data-id", "")
            name = el.get("data-room-name") or el.get("data-name") or el.get_text(strip=True)
            if not room_id:
                continue
            rooms.append({
                "id": str(room_id),
                "name": name,
                "area": el.get("data-area", ""),
                "floor": el.get("data-floor", ""),
                "capacity": el.get("data-capacity", ""),
                "type": el.get("data-type", ""),
            })
        return rooms

    @staticmethod
    def _dedup_rooms(rooms: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for r in rooms:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
        return out

    @staticmethod
    def _dict_to_room(item: dict) -> Optional[Dict]:
        """Normalize a JSON dict into our room format. Returns None if not a valid room."""
        room_id = (
            item.get("Id") or item.get("id") or item.get("ID")
            or item.get("RoomId") or item.get("roomId")
            or item.get("ResourceId") or item.get("resourceId")
            or item.get("Value") or item.get("value")
            or ""
        )
        name = (
            item.get("Name") or item.get("name")
            or item.get("RoomName") or item.get("roomName")
            or item.get("Text") or item.get("text")
            or item.get("Title") or item.get("title")
            or ""
        )
        if not room_id and not name:
            return None
        room_id = str(room_id)
        return {
            "id": room_id,
            "name": name or room_id,
            "area": str(item.get("Area") or item.get("area") or ""),
            "floor": str(item.get("Floor") or item.get("floor") or ""),
            "capacity": str(item.get("Capacity") or item.get("capacity") or ""),
            "type": str(item.get("Type") or item.get("type") or item.get("Category") or item.get("category") or ""),
        }

    def get_dashboard_debug(self) -> Dict:
        """Return diagnostic info about what the BookingDashboard page contains."""
        resp = self.session.get(f"{self.BASE_URL}/BookingDashboard", timeout=15)
        resp.raise_for_status()
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        selects_info = []
        for select in soup.find_all("select"):
            opts = [(o.get("value", ""), o.get_text(strip=True)) for o in select.find_all("option")]
            selects_info.append({
                "id": select.get("id", ""),
                "name": select.get("name", ""),
                "option_count": len(opts),
                "options_preview": opts[:15],
            })

        scripts_with_json = []
        for i, script in enumerate(soup.find_all("script")):
            text = script.string or ""
            if any(kw in text.lower() for kw in ("room", "resource", "scheduler", "datasource")):
                scripts_with_json.append({
                    "index": i,
                    "src": script.get("src", "inline"),
                    "length": len(text),
                    "preview": text[:500],
                })

        external_scripts = [
            script.get("src") for script in soup.find_all("script", src=True)
        ]

        api_results = {}
        for path in self._ROOM_API_PATHS:
            try:
                r = self.session.get(f"{self.BASE_URL}{path}", timeout=8)
                api_results[path] = {
                    "status": r.status_code,
                    "content_type": r.headers.get("content-type", ""),
                    "preview": r.text[:500] if r.status_code == 200 else "",
                }
            except Exception as exc:
                api_results[path] = {"status": "error", "message": str(exc)}

        data_attrs = []
        for el in soup.select("[data-room-id], [data-resource-id], [data-id]"):
            data_attrs.append({k: v for k, v in el.attrs.items() if k.startswith("data-")})

        body_text_soup = BeautifulSoup(html, "html.parser")
        for tag in body_text_soup.find_all(["script", "style", "select", "head"]):
            tag.decompose()
        body = body_text_soup.find("body")
        body_text = ""
        if body:
            raw = body.get_text(separator="\n")
            body_text = "\n".join(line.strip() for line in raw.splitlines() if line.strip())[:3000]

        body_html_soup = BeautifulSoup(html, "html.parser")
        for tag in body_html_soup.find_all(["script", "style"]):
            tag.decompose()
        main = body_html_soup.find("main") or body_html_soup.find("div", class_=re.compile(r"content|scheduler|main", re.I)) or body_html_soup.find("body")
        body_html_sample = str(main)[:5000] if main else ""

        return {
            "html_length": len(html),
            "title": soup.title.string if soup.title else "",
            "selects": selects_info,
            "external_script_urls": external_scripts,
            "scripts_with_room_keywords": scripts_with_json,
            "api_probe_results": api_results,
            "data_attr_elements": data_attrs[:20],
            "body_visible_text_sample": body_text,
            "body_html_sample": body_html_sample,
            "rooms_found_by_get_rooms": [{"id": r["id"], "name": r["name"], "type": r.get("type", "")} for r in self.get_rooms(force_refresh=True)],
        }

    # ------------------------------------------------------------------
    # Schedule / events
    # ------------------------------------------------------------------

    def get_room_schedule(
        self,
        room_id: str,
        date_from: str,
        date_to: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch booked and blocked time slots for a room.

        Args:
            room_id: The room identifier.
            date_from: Start date in YYYY-MM-DD format.
            date_to: End date (defaults to date_from for single-day lookup).
        """
        if date_to is None:
            date_to = date_from

        events = self._fetch_recurring_events(room_id, date_from, date_to)
        blocks = self._fetch_block_periods(room_id, date_from, date_to)
        return events + blocks

    def _fetch_recurring_events(self, room_id: str, date_from: str, date_to: str) -> List[Dict]:
        url = f"{self.BASE_URL}/api/recurringEvents"
        params = {"from": date_from, "to": date_to, "roomId": room_id}
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
        except Exception:
            return []

        events = []
        items = raw if isinstance(raw, list) else raw.get("data", raw.get("events", []))
        for item in items:
            events.append(self._normalize_event(item, source="booking"))
        return events

    def _fetch_block_periods(self, room_id: str, date_from: str, date_to: str) -> List[Dict]:
        url = f"{self.BASE_URL}/Scheduler/GetBlockPeriods"
        params = {"from": date_from, "to": date_to, "roomId": room_id}
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
        except Exception:
            return []

        blocks = []
        items = raw if isinstance(raw, list) else raw.get("data", raw.get("blocks", []))
        for item in items:
            blocks.append(self._normalize_event(item, source="blocked"))
        return blocks

    @staticmethod
    def _normalize_event(item: dict, source: str = "booking") -> Dict:
        """Convert a raw event/block JSON object into a uniform dict."""
        start_raw = item.get("start") or item.get("Start") or item.get("startDate") or ""
        end_raw = item.get("end") or item.get("End") or item.get("endDate") or ""

        def _parse_dt(val):
            if not val:
                return ""
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    return datetime.strptime(val[:19], fmt[:len(val[:19]) + 2] if '.' in fmt else fmt)
                except (ValueError, IndexError):
                    continue
            return val

        start_dt = _parse_dt(start_raw)
        end_dt = _parse_dt(end_raw)

        return {
            "title": item.get("title") or item.get("Title") or item.get("subject") or "Booked",
            "start": str(start_dt) if start_dt else start_raw,
            "end": str(end_dt) if end_dt else end_raw,
            "start_time": start_dt.strftime("%H:%M") if isinstance(start_dt, datetime) else "",
            "end_time": end_dt.strftime("%H:%M") if isinstance(end_dt, datetime) else "",
            "date": start_dt.strftime("%Y-%m-%d") if isinstance(start_dt, datetime) else "",
            "source": source,
            "organizer": item.get("organizer") or item.get("Organizer") or "",
            "description": item.get("description") or item.get("Description") or "",
        }

    # ------------------------------------------------------------------
    # Availability search
    # ------------------------------------------------------------------

    def search_available_rooms(
        self,
        date: str,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
    ) -> List[Dict]:
        """Return rooms that are free during the given window."""
        rooms = self.get_rooms()
        available = []

        for room in rooms:
            schedule = self.get_room_schedule(room["id"], date)
            if self._is_free(schedule, time_start, time_end):
                available.append({**room, "schedule": schedule})

        return available

    @staticmethod
    def _is_free(schedule: List[Dict], time_start: Optional[str], time_end: Optional[str]) -> bool:
        if not time_start or not time_end:
            return len(schedule) == 0

        req_start = datetime.strptime(time_start, "%H:%M")
        req_end = datetime.strptime(time_end, "%H:%M")

        for event in schedule:
            evt_start_str = event.get("start_time", "")
            evt_end_str = event.get("end_time", "")
            if not evt_start_str or not evt_end_str:
                continue
            try:
                evt_start = datetime.strptime(evt_start_str, "%H:%M")
                evt_end = datetime.strptime(evt_end_str, "%H:%M")
            except ValueError:
                continue
            if req_start < evt_end and req_end > evt_start:
                return False
        return True

    # ------------------------------------------------------------------
    # My bookings
    # ------------------------------------------------------------------

    def get_my_bookings(self) -> List[Dict]:
        """Attempt to scrape the current user's own bookings."""
        for path in ("/MyBookings", "/BookingDashboard/MyBookings", "/Scheduler/MyBookings"):
            try:
                resp = self.session.get(f"{self.BASE_URL}{path}", timeout=15)
                if resp.status_code == 200 and "booking" in resp.text.lower():
                    return self._parse_my_bookings_page(resp.text)
            except Exception:
                continue
        return []

    def _parse_my_bookings_page(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        bookings: List[Dict] = []

        for row in soup.select("table tr"):
            cells = row.find_all("td")
            if len(cells) >= 3:
                bookings.append({
                    "room": cells[0].get_text(strip=True),
                    "date": cells[1].get_text(strip=True),
                    "time": cells[2].get_text(strip=True),
                    "status": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                })

        return bookings

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_schedule_as_text(schedule: List[Dict], room_name: str = "", date: str = "") -> str:
        """Convert structured schedule data into clean text for LLM context."""
        if not schedule:
            header = f"Room {room_name}" if room_name else "Room"
            if date:
                header += f" on {date}"
            return f"{header}: No bookings found — the room appears to be free all day."

        lines = []
        header = f"Schedule for {room_name}" if room_name else "Room schedule"
        if date:
            header += f" on {date}"
        lines.append(header)
        lines.append("-" * len(header))

        for event in sorted(schedule, key=lambda e: e.get("start_time", "")):
            time_range = f"{event.get('start_time', '?')} – {event.get('end_time', '?')}"
            title = event.get("title", "Booked")
            source = event.get("source", "")
            organizer = event.get("organizer", "")

            detail = f"  {time_range}  |  {title}"
            if source == "blocked":
                detail += " [BLOCKED]"
            if organizer:
                detail += f"  (by {organizer})"
            lines.append(detail)

        return "\n".join(lines)

    @staticmethod
    def format_rooms_as_text(rooms: List[Dict]) -> str:
        """Format the room list for LLM context."""
        if not rooms:
            return "No rooms found in the system."

        lines = ["Available rooms:", "-" * 40]
        for r in rooms:
            parts = [f"ID: {r['id']}", f"Name: {r['name']}"]
            if r.get("area"):
                parts.append(f"Area: {r['area']}")
            if r.get("floor"):
                parts.append(f"Floor: {r['floor']}")
            if r.get("capacity"):
                parts.append(f"Capacity: {r['capacity']}")
            if r.get("type"):
                parts.append(f"Type: {r['type']}")
            lines.append("  " + " | ".join(parts))
        return "\n".join(lines)

    @staticmethod
    def format_available_rooms_as_text(
        rooms: List[Dict], date: str, time_start: str = "", time_end: str = ""
    ) -> str:
        """Format search-available-rooms results for LLM context."""
        time_desc = f" between {time_start} and {time_end}" if time_start and time_end else ""
        if not rooms:
            return f"No rooms are available on {date}{time_desc}."

        lines = [f"Rooms available on {date}{time_desc}:", "-" * 40]
        for r in rooms:
            name = r.get("name", r.get("id", "Unknown"))
            cap = f" (capacity: {r['capacity']})" if r.get("capacity") else ""
            lines.append(f"  - {name}{cap}")
        return "\n".join(lines)

    @staticmethod
    def format_my_bookings_as_text(bookings: List[Dict]) -> str:
        if not bookings:
            return "You have no upcoming bookings."

        lines = ["Your bookings:", "-" * 30]
        for b in bookings:
            line = f"  {b.get('room', '?')} | {b.get('date', '?')} | {b.get('time', '?')}"
            if b.get("status"):
                line += f" | {b['status']}"
            lines.append(line)
        return "\n".join(lines)
