"""RBS (Room Booking System) client for authenticated access to room data."""
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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

    _CARD_TITLE_RE = re.compile(
        r'^(.+?)\s*-\s*(.+)\(([A-Za-z]{0,3}\d{1,4}(?:-\d{1,4})?[A-Za-z]?)\)\s*$'
    )
    _SCHEDULER_LINK_RE = re.compile(r'/Scheduler/Index/(\d+)')

    def get_rooms(self, force_refresh: bool = False) -> List[Dict]:
        """Discover rooms from the BookingDashboard page. Results are cached.

        Each room dict contains:
            id:           Display room code (e.g. "302")
            scheduler_id: Internal ID used in /Scheduler/Index/{id} URLs
            name, area, capacity, type: descriptive metadata
        """
        if self._rooms_cache and not force_refresh:
            return self._rooms_cache

        resp = self.session.get(f"{self.BASE_URL}/BookingDashboard", timeout=15)
        resp.raise_for_status()

        rooms = self._parse_room_cards(resp.text)
        self._rooms_cache = rooms
        return rooms

    def _parse_room_cards(self, html: str) -> List[Dict]:
        """Parse room cards from the BookingDashboard page.

        Each card has a title like ``CBCC - Classroom (302)`` followed by
        labelled fields (Room Type, Capacity, Equipment, Desk Type, Remarks).
        Links to ``/Scheduler/Index/{id}`` are scanned to discover the internal
        scheduler ID for each room.
        """
        soup = BeautifulSoup(html, "html.parser")

        scheduler_id_map = self._find_scheduler_ids(soup)

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
            equipment: List[str] = []
            remarks = ""

            # Scan a limited window of lines below the card title for labelled fields.
            for j in range(i + 1, min(i + 20, len(lines))):
                lower = lines[j].lower()

                if lower.startswith("capacity") and j + 1 < len(lines):
                    cap_val = lines[j + 1].strip()
                    if cap_val.isdigit():
                        capacity = cap_val

                elif lower.startswith("equipment") and j + 1 < len(lines):
                    eq_line = lines[j + 1].strip()
                    if eq_line:
                        # Equipment items are often comma-separated; keep both raw and split form useful.
                        equipment = [e.strip() for e in eq_line.split(",") if e.strip()]

                elif lower.startswith("remarks") and j + 1 < len(lines):
                    remarks = lines[j + 1].strip()

            rooms.append({
                "id": room_number,
                "scheduler_id": scheduler_id_map.get(room_number, ""),
                "name": f"{area} - {room_type} ({room_number})",
                "area": area,
                "floor": "",
                "capacity": capacity,
                "type": room_type,
                "equipment": equipment,
                "remarks": remarks,
            })

        # Filter out phantom entries: rooms parsed from page text that have
        # no corresponding /Scheduler/Index/ link are unlikely to be real
        # bookable rooms (e.g. old references still on the page).
        valid_rooms = []
        for r in rooms:
            if r["scheduler_id"]:
                valid_rooms.append(r)
            else:
                print(f"[RBS] Dropping room {r['id']} ({r['name']}) — no scheduler ID found")

        if valid_rooms:
            print(f"[RBS] Parsed {len(valid_rooms)} rooms with scheduler IDs "
                  f"(dropped {len(rooms) - len(valid_rooms)} without)")
        return valid_rooms

    _ROOM_CODE_IN_PARENS_RE = re.compile(r'\(([A-Za-z]{0,3}\d{1,4}(?:-\d{1,4})?[A-Za-z]?)\)')

    def _find_scheduler_ids(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Scan the page for /Scheduler/Index/{id} links and map room codes
        to their internal scheduler IDs.

        Walks up the DOM from each matching link. At each level, counts the
        unique room codes visible in the parent text:
        - Exactly 1 code  → safe to map, stop.
        - Multiple codes   → walked into a shared container, stop WITHOUT mapping.
        - Zero codes       → keep walking up.

        Uses ``setdefault`` so the first (shallowest) mapping for a room code
        is never overwritten by a deeper, less-reliable walk from another link.
        """
        mapping: Dict[str, str] = {}

        for link in soup.find_all("a", href=True):
            m = self._SCHEDULER_LINK_RE.search(link["href"])
            if not m:
                continue
            scheduler_id = m.group(1)

            parent = link.parent
            for _ in range(10):
                if parent is None:
                    break
                search_text = parent.get_text(separator=" ", strip=True)
                codes_found = self._ROOM_CODE_IN_PARENS_RE.findall(search_text)
                unique_codes = list(dict.fromkeys(codes_found))

                if len(unique_codes) == 1:
                    mapping.setdefault(unique_codes[0], scheduler_id)
                    break
                elif len(unique_codes) > 1:
                    break
                parent = parent.parent

        if mapping:
            print(f"[RBS] Scheduler ID mapping ({len(mapping)} rooms): {mapping}")
        return mapping

    # ------------------------------------------------------------------
    # Schedule / events
    # ------------------------------------------------------------------

    @staticmethod
    def get_week_range(date_str: str) -> Tuple[str, str]:
        """Return the Monday-to-next-Monday range containing the given date."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        monday = dt - timedelta(days=dt.weekday())
        next_monday = monday + timedelta(days=7)
        return monday.strftime("%Y-%m-%d"), next_monday.strftime("%Y-%m-%d")

    def get_room_schedule(
        self,
        scheduler_id: str,
        date_from: str,
        date_to: Optional[str] = None,
        room_code: str = "",
    ) -> Optional[List[Dict]]:
        """Fetch booked events for a room via the Scheduler page session.

        1. Navigates to /Scheduler/Index/{scheduler_id} to set the session.
        2. Calls /api/recurringEvents with the appropriate week range.

        Returns a list of events on success, or ``None`` if the navigation
        or fetch failed (so callers can distinguish "no bookings" from
        "could not retrieve data").

        Args:
            scheduler_id: Internal room ID (from room["scheduler_id"]).
            date_from: Start date (YYYY-MM-DD).
            date_to:   End date (defaults to end of the week containing date_from).
            room_code: Display room code for logging (e.g. "302").
        """
        if not self._navigate_to_room(scheduler_id, room_code):
            return None

        if date_to is None:
            week_from, week_to = self.get_week_range(date_from)
        else:
            wf, _ = self.get_week_range(date_from)
            _, wt = self.get_week_range(date_to)
            week_from, week_to = wf, wt

        return self._fetch_recurring_events(week_from, week_to)

    def _navigate_to_room(self, scheduler_id: str, room_code: str = "") -> bool:
        """Visit the Scheduler page for a room to set the session context.

        Returns True on success, False on failure.
        """
        label = f"{room_code} (sid={scheduler_id})" if room_code else f"sid={scheduler_id}"
        url = f"{self.BASE_URL}/Scheduler/Index/{scheduler_id}"
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            print(f"[RBS] Navigated to room {label}")
            return True
        except Exception as e:
            print(f"[RBS] Failed to navigate to room {label}: {e}")
            return False

    def _fetch_recurring_events(self, date_from: str, date_to: str) -> Optional[List[Dict]]:
        """Call the recurringEvents API (session determines the room).

        Returns a list of events on success, or ``None`` on failure.
        """
        url = f"{self.BASE_URL}/api/recurringEvents"
        params = {"timeshift": "-480", "from": date_from, "to": date_to}
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            print(f"[RBS] Error fetching recurring events: {e}")
            return None

        items = raw if isinstance(raw, list) else raw.get("data", raw.get("events", []))
        return [self._normalize_event(item) for item in items]

    # ------------------------------------------------------------------
    # Event normalization
    # ------------------------------------------------------------------

    _REASON_RE = re.compile(
        r'^([A-Z]{2,}[\dA-Z]*\d[\dA-Za-z]*)'   # course code
        r'-(.+?)'                                 # course name
        r'-(Lect|Tut|Lab|Sem|Exam|Prac|Wksp)'    # session type
        r'-(\[.*\])$'                             # groups
    )

    @staticmethod
    def _parse_reason(reason: str) -> Dict[str, str]:
        """Parse a reason like 'HDE203-Specialty Nursing-Lect-[A]' into parts."""
        m = RBSClient._REASON_RE.match(reason)
        if m:
            return {
                "course_code": m.group(1),
                "course_name": m.group(2).strip(),
                "session_type": m.group(3),
                "groups": m.group(4),
            }
        return {"course_code": "", "course_name": reason, "session_type": "", "groups": ""}

    @staticmethod
    def _normalize_event(item: dict) -> Dict:
        """Convert a raw recurringEvents JSON object into a structured dict."""
        reason = item.get("reason") or "Booked"
        parsed = RBSClient._parse_reason(reason)

        start_raw = item.get("start_date") or ""
        end_raw = item.get("end_date") or ""

        def _parse_dt(val: str) -> Optional[datetime]:
            if not val:
                return None
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(val.strip(), fmt)
                except ValueError:
                    continue
            return None

        start_dt = _parse_dt(start_raw)
        end_dt = _parse_dt(end_raw)

        teacher_names_raw = item.get("teachernames") or ""
        teachers = "; ".join(t.strip() for t in teacher_names_raw.split(";") if t.strip())

        status_map = {"I": "confirmed", "A": "approved", "P": "pending"}
        status = status_map.get(item.get("status", ""), item.get("status", ""))

        type_map = {"C": "class", "O": "reserved"}
        booking_type = type_map.get(item.get("type", ""), item.get("type", ""))

        return {
            "title": reason,
            "course_code": parsed["course_code"],
            "course_name": parsed["course_name"],
            "session_type": parsed["session_type"],
            "groups": parsed["groups"],
            "start": str(start_dt) if start_dt else start_raw,
            "end": str(end_dt) if end_dt else end_raw,
            "date": start_dt.strftime("%Y-%m-%d") if start_dt else "",
            "start_time": start_dt.strftime("%H:%M") if start_dt else "",
            "end_time": end_dt.strftime("%H:%M") if end_dt else "",
            "teachers": teachers,
            "room_code": item.get("room_code") or "",
            "status": status,
            "booking_type": booking_type,
            "no_of_participant": item.get("no_of_participant"),
            "source": "booking",
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
        """Return rooms that are free during the given window.

        Rooms whose schedule could not be fetched are skipped (not assumed free).
        Uses parallel fetching for speed.
        """
        rooms = self.get_rooms()

        def _check_room(room: Dict) -> Optional[Dict]:
            sid = room.get("scheduler_id")
            if not sid:
                return None
            schedule = self.get_room_schedule(sid, date, room_code=room.get("id", ""))
            if schedule is None:
                return None
            if self._is_free(schedule, date, time_start, time_end):
                return {**room, "schedule": schedule}
            return None

        available = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_check_room, r): r for r in rooms}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    available.append(result)
        return available

    @staticmethod
    def _is_free(
        schedule: List[Dict], date: str,
        time_start: Optional[str], time_end: Optional[str],
    ) -> bool:
        day_events = [e for e in schedule if e.get("date") == date]

        if not time_start or not time_end:
            return len(day_events) == 0

        req_start = datetime.strptime(time_start, "%H:%M")
        req_end = datetime.strptime(time_end, "%H:%M")

        for event in day_events:
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

    @staticmethod
    def _get_free_slots(
        schedule: List[Dict], date: str,
        time_filter_start: Optional[str] = None, time_filter_end: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Return list of (start_time, end_time) free intervals for the given date.
        If time_filter_start/time_filter_end are set, only return free slots that overlap that window.
        Day bounds: Mon-Fri 09:00-22:00, Sat 09:00-18:00."""
        day_events = [e for e in schedule if e.get("date") == date]
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            day_start = "09:00"
            day_end = "18:00" if dt.weekday() == 5 else "22:00"  # Sat=5
        except ValueError:
            day_start, day_end = "09:00", "22:00"

        occupied = []
        for e in day_events:
            s, en = e.get("start_time"), e.get("end_time")
            if s and en:
                try:
                    occupied.append((datetime.strptime(s, "%H:%M"), datetime.strptime(en, "%H:%M")))
                except ValueError:
                    pass
        occupied.sort(key=lambda x: x[0])
        # Merge overlapping
        merged = []
        for s, en in occupied:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], en))
            else:
                merged.append((s, en))

        day_s = datetime.strptime(day_start, "%H:%M")
        day_e = datetime.strptime(day_end, "%H:%M")
        free = []
        cur = day_s
        for s, en in merged:
            if cur < s:
                slot_start = cur.strftime("%H:%M")
                slot_end = min(s, day_e).strftime("%H:%M")
                if slot_start < slot_end:
                    free.append((slot_start, slot_end))
            cur = max(cur, en)
        if cur < day_e:
            free.append((cur.strftime("%H:%M"), day_e.strftime("%H:%M")))

        if time_filter_start and time_filter_end:
            try:
                f_start = datetime.strptime(time_filter_start, "%H:%M")
                f_end = datetime.strptime(time_filter_end, "%H:%M")
                filtered = []
                for slot_s, slot_e in free:
                    slot_start = datetime.strptime(slot_s, "%H:%M")
                    slot_end = datetime.strptime(slot_e, "%H:%M")
                    if slot_start < f_end and slot_end > f_start:
                        overlap_s = max(slot_start, f_start).strftime("%H:%M")
                        overlap_e = min(slot_end, f_end).strftime("%H:%M")
                        filtered.append((overlap_s, overlap_e))
                return filtered
            except ValueError:
                pass
        return free

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
    def format_schedule_as_text(schedule: Optional[List[Dict]], room_name: str = "", date: str = "") -> str:
        """Convert schedule data into clean text for LLM context, grouped by date.

        ``None`` means the fetch failed — availability is unknown.
        An empty list ``[]`` means the fetch succeeded with zero bookings.
        """
        if schedule is None:
            header = f"Room {room_name}" if room_name else "Room"
            return (
                f"{header}: ERROR — Could not retrieve schedule data from the booking system. "
                "Availability CANNOT be confirmed. Do NOT assume the room is free."
            )

        if not schedule:
            header = f"Room {room_name}" if room_name else "Room"
            if date:
                header += f" on {date}"
            return f"{header}: No bookings found — the room appears to be free all day."

        if date:
            day_events = [e for e in schedule if e.get("date") == date]
            if not day_events:
                return f"Room {room_name} on {date}: No bookings — the room is free all day."
        else:
            day_events = schedule

        by_date: Dict[str, List[Dict]] = defaultdict(list)
        for event in day_events:
            by_date[event.get("date", "unknown")].append(event)

        lines = []
        header = f"Schedule for Room {room_name}" if room_name else "Room schedule"
        lines.append(header)
        lines.append("=" * len(header))

        for d in sorted(by_date.keys()):
            try:
                day_label = datetime.strptime(d, "%Y-%m-%d").strftime("%A, %B %d")
            except ValueError:
                day_label = d
            lines.append(f"\n{day_label} ({d})")
            lines.append("-" * 40)

            for event in sorted(by_date[d], key=lambda e: e.get("start_time", "")):
                time_range = f"{event.get('start_time', '?')} - {event.get('end_time', '?')}"
                title = event.get("title", "Booked")
                booking_type = event.get("booking_type", "")
                teachers = event.get("teachers", "")

                line = f"  {time_range}  |  {title}"
                if booking_type == "reserved":
                    line += "  [RESERVED]"
                if teachers:
                    line += f"  (Teacher: {teachers})"
                lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def format_rooms_as_text(rooms: List[Dict]) -> str:
        """Format the room list for LLM context."""
        if not rooms:
            return "No rooms found in the system."

        lines = ["Available rooms:", "-" * 40]
        for r in rooms:
            parts = [f"Room {r['id']}", f"Name: {r['name']}"]
            if r.get("area"):
                parts.append(f"Area: {r['area']}")
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
