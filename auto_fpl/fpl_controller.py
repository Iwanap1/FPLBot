import os
from dotenv import load_dotenv
load_dotenv()
from playwright.sync_api import sync_playwright
import os, time, requests
import json
from typing import List

class FPLController:
    def __init__(
            self,
            email: str = os.getenv("EMAIL", None),
            password: str = os.getenv("PASSWORD", None),
            team_id: int = int(os.getenv("FPL_ID", None)),
    ):
        self.email = email
        self.bootstrap_static = requests.get(
            "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30
        ).json()
        self.password = password
        self.team_id = team_id
        self.fpl_home = "https://fantasy.premierleague.com/"
        if not self.email or not self.password or not self.team_id:
            raise ValueError("Email, password, and team ID must be provided.")
        
    
    def run_login_and_get_token(self, headless=True):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless, slow_mo=100)
            ctx = browser.new_context()
            page = ctx.new_page()

            # 1) Go to homepage and click "Log in"
            page.goto(self.fpl_home, wait_until="domcontentloaded")
            self._accept_cookies_if_present(page)
            try:
                page.locator("button:has-text('Log in')").first.click(timeout=5000)
            except Exception:
                page.goto("https://fantasy.premierleague.com/a/login", wait_until="domcontentloaded")

            # 2) Fill login form
            login_frame = self._find_frame_with_username(page, timeout_ms=20000)
            if not login_frame:
                raise SystemExit("Could not locate username input")

            login_frame.locator("input#username[name='username']").fill(self.email)
            login_frame.locator("input#password[name='password']").fill(self.password)
            login_frame.locator("button[type='submit']").first.click(timeout=4000)

            def is_myteam_req(r):
                u = r.url
                return "/api/my-team/" in u  # do not hardcode TEAM_ID in case the app uses a different id first


            with page.expect_request(is_myteam_req, timeout=20000) as req_info:
                page.locator("a[href='/my-team']").first.click(timeout=10000)

            req = req_info.value
            token_header = req.headers.get("x-api-authorization") or ""
            token = token_header.replace("Bearer ", "") if token_header else None

            browser.close()
            self.access_token = token
            return token
    

    def start_session(self, headless=True):
        if not hasattr(self, 'access_token'):
            print("Obtaining access token, this may take a while")
            self.run_login_and_get_token(headless=headless)
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Authorization": f"Bearer {self.access_token}",
        })
        self.session = s


    def get_current_squad_info(self, retry_once=True, headless=True):
        if not hasattr(self, 'session'):
            self.start_session(headless=headless)
        url = f"https://fantasy.premierleague.com/api/my-team/{self.team_id}/"
        response = self.session.get(url, timeout=30)
        if response.status_code != 200:
            if retry_once:
                print(f"Failed to fetch team data: {response.status_code} {response.reason}")
                print("Obtaining a new access token and retrying once...")
                self.access_token = self.run_login_and_get_token(headless=False)
                self.start_session(headless=False)
                return self.get_current_squad_info(retry_once=False)
            else:
                raise RuntimeError(f"Failed to fetch team data: {response.status_code} {response.reason}")
        
        data = response.json()
        self.data = data
        return data
    
        
    def _find_frame_with_username(self, page, timeout_ms=15000):
        selector = "input#username[name='username']"
        end = time.time() + timeout_ms/1000.0
        while time.time() < end:
            for fr in page.frames:
                try:
                    if fr.locator(selector).count():
                        return fr
                except Exception:
                    continue
            time.sleep(0.25)
        return None


    def upcoming_gw(self) -> int:
        for ev in self.bootstrap_static["events"]:
            if ev.get("is_next"):
                self.gw = int(ev["id"])
                return self.gw
        # fallback: next non-finished event
        for ev in self.bootstrap_static["events"]:
            if not ev.get("finished"):
                self.gw = int(ev["id"])
                return self.gw
        raise RuntimeError("Could not infer upcoming GW")


    def _accept_cookies_if_present(self, page):
        for sel in (
            "#onetrust-accept-btn-handler",
            "button:has-text('Accept All Cookies')",
            "button:has-text('Accept all')",
            "button:has-text('Accept')",
        ):
            try:
                page.locator(sel).click(timeout=1200)
                break
            except Exception:
                pass


    def submit_transfer(self, transfers, wildcard: bool = False, free_hit: bool = False):
        if not hasattr(self, "gw"):
            self.upcoming_gw()
        if not hasattr(self, "session"):
            self.start_session()

        url = "https://fantasy.premierleague.com/api/transfers/"
        payload = {
            "confirmed": False,           # first pass: preview
            "entry": self.team_id,
            "event": self.gw,
            "transfers": transfers,
            "wildcard": wildcard,
            "freehit": free_hit,          # <-- key is 'freehit' (no underscore)
        }

        hdrs = {
            **self.session.headers,
            "Content-Type": "application/json",
            "Origin": "https://fantasy.premierleague.com",
            "Referer": "https://fantasy.premierleague.com/",
        }

        # include the header name FPL actually uses when we sniffed it
        if "x-api-authorization" not in hdrs and "Authorization" in hdrs:
            hdrs["x-api-authorization"] = hdrs["Authorization"]

        r = self.session.post(url, data=json.dumps(payload), headers=hdrs)

        # Accept 200/204, and only parse JSON when present
        if r.status_code not in (200, 204):
            raise RuntimeError(f"Transfer (preview) failed {r.status_code}: {r.text[:300]}")

        if r.headers.get("content-type","").startswith("application/json") and r.text.strip():
            preview = r.json()
            print("Preview:", preview)
        else:
            print(f"Preview: HTTP {r.status_code} with no body")

        # Second pass: confirm the transfers
        payload["confirmed"] = True
        r2 = self.session.post(url, data=json.dumps(payload), headers=hdrs)

        if r2.status_code not in (200, 204):
            raise RuntimeError(f"Transfer (confirm) failed {r2.status_code}: {r2.text[:300]}")

        if r2.headers.get("content-type","").startswith("application/json") and r2.text.strip():
            print("Transfer successful:", r2.json())
            return r2.json()

        print(f"Transfer successful: HTTP {r2.status_code} with no body")
        return {"ok": True, "status_code": r2.status_code}

    
    def organise_squad(self, plan: dict):
        if not hasattr(self, "session"):
            self.start_session()

        starters = plan["StartingXI"]
        captain  = plan["Captain"]
        vice     = plan["ViceCaptain"]
        bench    = plan["BenchOrder"]              # user order

        assert len(starters) == 11, "Need 11 starters"
        assert len(bench) == 4, "BenchOrder must have 4 ids (3 outfield + 1 GK)"
        assert captain in starters and vice in starters and captain != vice

        # map element_id -> element_type (1 GK, 2 DEF, 3 MID, 4 FWD)
        pool = self.session.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()["elements"]
        type_map = {p["id"]: p["element_type"] for p in pool}

        # validate starters: exactly 1 GK
        gk_starters = [e for e in starters if type_map.get(e) == 1]
        assert len(gk_starters) == 1, "StartingXI must contain exactly 1 GK"

        # split bench into GK vs outfield (preserve given order)
        bench_gk = None
        bench_out = []
        for e in bench:
            if type_map.get(e) == 1:
                bench_gk = e
            else:
                bench_out.append(e)
        if bench_gk is None:
            raise ValueError("BenchOrder must include your bench GK")
        if len(bench_out) != 3:
            raise ValueError("BenchOrder must include exactly 3 outfielders")

        # build picks
        picks = []
        # starters positions 1..11
        for i, el in enumerate(starters, start=1):
            picks.append({
                "element": el,
                "position": i,
                "is_captain": (el == captain),
                "is_vice_captain": (el == vice),
                "multiplier": 1
            })

        # bench GK at 12
        picks.append({
            "element": bench_gk,
            "position": 12,
            "is_captain": False,
            "is_vice_captain": False,
            "multiplier": 0
        })

        # bench outfield 13..15 (keep your order)
        for pos, el in zip((13, 14, 15), bench_out):
            picks.append({
                "element": el,
                "position": pos,
                "is_captain": False,
                "is_vice_captain": False,
                "multiplier": 0
            })

        url = f"https://fantasy.premierleague.com/api/my-team/{self.team_id}/"
        headers = {
            **self.session.headers,
            "Content-Type": "application/json",
            "Origin": "https://fantasy.premierleague.com",
            "Referer": "https://fantasy.premierleague.com/",
        }
        payload = {"picks": picks, "chips": []}
        r = self.session.post(url, json=payload, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"Lineup submission failed {r.status_code}: {r.text}")
        return r.json()


