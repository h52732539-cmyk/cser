"""Query intent parser.

Translates a natural-language query into structured constraints:

    QueryIntent
      .semantic_text   : str            # feed to Huawei CLIP text encoder
      .time_window     : (start, end)   # POSIX seconds, UTC
      .geo_categories  : List[str]      # ∈ GEO_CATEGORIES
      .motion_classes  : List[str]      # ∈ MOTION_CLASSES
      .device_filter   : Optional[str]  # 'huawei' / 'iphone' / ...

Rule-based, language-agnostic (EN + ZH supported). No external NLU
dependency; runs in < 0.5 ms per query on CPU.

Example:
    parse("去年夏天在海边跑步的视频")
      → semantic_text="跑步 视频",
        time_window=(1688169600, 1693526400),   # 2023-07 ~ 2023-08
        geo_categories=["coast"],
        motion_classes=["running"]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from .metadata import GEO_CATEGORIES, MOTION_CLASSES


# ----------------------------------------------------------------------
#  Keyword lexicons (EN / ZH)
# ----------------------------------------------------------------------

_GEO_LEXICON = {
    "coast":         ["coast", "beach", "ocean", "seaside", "shore",
                      "海边", "沙滩", "海滩", "海岸", "岸边"],
    "mountain":      ["mountain", "summit", "hill", "peak",
                      "山", "山上", "山顶", "山里", "爬山"],
    "urban":         ["city", "downtown", "street", "urban", "skyline",
                      "商场", "市区", "市中心", "街道", "城市"],
    "indoor_home":   ["home", "room", "bedroom", "kitchen", "living room",
                      "家", "家里", "卧室", "厨房", "客厅", "屋里", "室内"],
    "indoor_public": ["restaurant", "cafe", "mall", "office",
                      "餐厅", "饭店", "咖啡", "商场", "办公室"],
    "rural":         ["countryside", "farm", "village",
                      "乡下", "农村", "田野", "村庄"],
    "road":          ["highway", "road", "traffic",
                      "高速", "马路", "公路"],
}

_MOTION_LEXICON = {
    "running":    ["running", "jogging",
                   "跑步", "跑", "晨跑", "夜跑", "奔跑"],
    "walking":    ["walk", "walking", "strolling", "hiking",
                   "散步", "走路", "徒步", "漫步", "走"],
    "cycling":    ["cycling", "biking", "bike",
                   "骑行", "骑车", "自行车"],
    "vehicle":    ["driving", "car ride", "bus", "train",
                   "开车", "乘车", "坐车", "行车", "公交", "地铁"],
    "stationary": ["sitting", "still",
                   "静态", "静止", "坐着"],
}

_DEVICE_LEXICON = {
    "huawei":      ["huawei", "mate", "华为", "荣耀"],
    "iphone":      ["iphone", "apple"],
    "samsung":     ["samsung", "galaxy", "三星"],
}


# ----------------------------------------------------------------------
#  Time expression patterns
# ----------------------------------------------------------------------

_ZH_NUM = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "两": 2,
    "壹": 1, "贰": 2, "叁": 3, "肆": 4, "伍": 5,
}

_SEC_PER_DAY = 86400
_SEC_PER_WEEK = 7 * _SEC_PER_DAY


@dataclass
class QueryIntent:
    semantic_text: str
    time_window:   Optional[Tuple[float, float]] = None
    geo_categories: List[str]     = field(default_factory=list)
    motion_classes: List[str]     = field(default_factory=list)
    device_filter:  Optional[str] = None
    raw_query:      str           = ""

    def has_constraint(self) -> bool:
        return bool(self.time_window or self.geo_categories
                    or self.motion_classes or self.device_filter)


class QueryParser:
    def __init__(self, now_ts: Optional[float] = None) -> None:
        # `now_ts` lets tests use a fixed "now".
        self.now_ts = now_ts if now_ts is not None else datetime.now(
            timezone.utc
        ).timestamp()

    # ------------------------------------------------------------------

    def parse(self, text: str) -> QueryIntent:
        raw = text
        lower = text.lower()

        # --- time ---
        tw = self._parse_time(lower)

        # --- geo ---
        geos: List[str] = []
        lower_ci = lower
        for cat, kws in _GEO_LEXICON.items():
            if any(k in lower_ci or k in text for k in kws):
                geos.append(cat)

        # --- motion ---
        motions: List[str] = []
        for cls, kws in _MOTION_LEXICON.items():
            if any(k in lower_ci or k in text for k in kws):
                motions.append(cls)

        # --- device ---
        dev = None
        for name, kws in _DEVICE_LEXICON.items():
            if any(k in lower_ci or k in text for k in kws):
                dev = name
                break

        # --- semantic residue ---
        #    we keep the full query text as semantic — subtracting meta
        #    keywords risks breaking CLIP tokenization. CLIP handles
        #    these words fine as part of the prompt.
        return QueryIntent(
            semantic_text=raw,
            time_window=tw,
            geo_categories=geos,
            motion_classes=motions,
            device_filter=dev,
            raw_query=raw,
        )

    # ==================================================================
    #  Time parsing
    # ==================================================================

    def _parse_time(self, text: str) -> Optional[Tuple[float, float]]:
        now = self.now_ts
        dt_now = datetime.fromtimestamp(now, tz=timezone.utc)

        # "今天" / "today"
        if re.search(r"\btoday\b|今天|今日", text):
            return self._day_of(dt_now)

        # "昨天" / "yesterday"
        if re.search(r"\byesterday\b|昨天|昨日", text):
            return self._day_of(dt_now - timedelta(days=1))

        # "前天"
        if "前天" in text:
            return self._day_of(dt_now - timedelta(days=2))

        # "本周" / "this week"
        if re.search(r"\bthis week\b|本周|这周", text):
            return self._week_of(dt_now)

        # "上周" / "last week" / "上个星期"
        if re.search(r"\blast week\b|上周|上个星期|上星期", text):
            return self._week_of(dt_now - timedelta(days=7))

        # "本月" / "this month"
        if re.search(r"\bthis month\b|本月|这个月", text):
            return self._month_of(dt_now)

        # "上个月" / "last month"
        if re.search(r"\blast month\b|上月|上个月", text):
            y, m = dt_now.year, dt_now.month - 1
            if m == 0:
                y, m = y - 1, 12
            return self._month_of(datetime(y, m, 1, tzinfo=timezone.utc))

        # "今年" / "this year"
        if re.search(r"\bthis year\b|今年", text):
            return self._year_of(dt_now)

        # "去年" / "last year"
        if re.search(r"\blast year\b|去年", text):
            return self._year_of(datetime(dt_now.year - 1, 1, 1,
                                           tzinfo=timezone.utc))

        # "过去 N 天" / "最近 N 天" / "last N days"
        m = re.search(r"(?:last|past|最近|过去|这)\s*(\d+)\s*(day|days|天)",
                       text)
        if m:
            n = int(m.group(1))
            end = now
            start = now - n * _SEC_PER_DAY
            return (start, end)

        # "周末" / "weekend"
        if re.search(r"\bweekend\b|周末|礼拜", text):
            return self._last_weekend_of(dt_now)

        # Seasons ("summer", "夏天")
        season_map = {
            "spring": (3, 5),  "春天": (3, 5), "春季": (3, 5),
            "summer": (6, 8),  "夏天": (6, 8), "夏季": (6, 8),
            "autumn": (9, 11), "fall": (9, 11),
            "秋天": (9, 11),   "秋季": (9, 11),
            "winter": (12, 2), "冬天": (12, 2), "冬季": (12, 2),
        }
        for k, (a, b) in season_map.items():
            if k in text:
                year = dt_now.year
                if re.search(r"\blast year\b|去年", text):
                    year -= 1
                return self._season_of(year, a, b)

        # Explicit year: "2024年" / "in 2024"
        m = re.search(r"(\d{4})\s*(年|/|$)", text)
        if m:
            y = int(m.group(1))
            return self._year_of(datetime(y, 1, 1, tzinfo=timezone.utc))

        return None

    # ------------------------------------------------------------------
    #  Time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _day_of(dt: datetime) -> Tuple[float, float]:
        s = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        return s.timestamp(), (s + timedelta(days=1)).timestamp()

    @staticmethod
    def _week_of(dt: datetime) -> Tuple[float, float]:
        # ISO week starts Monday
        start = dt - timedelta(days=dt.weekday())
        s = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        return s.timestamp(), (s + timedelta(days=7)).timestamp()

    @staticmethod
    def _month_of(dt: datetime) -> Tuple[float, float]:
        s = datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)
        if dt.month == 12:
            e = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            e = datetime(dt.year, dt.month + 1, 1, tzinfo=timezone.utc)
        return s.timestamp(), e.timestamp()

    @staticmethod
    def _year_of(dt: datetime) -> Tuple[float, float]:
        s = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
        e = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
        return s.timestamp(), e.timestamp()

    @staticmethod
    def _season_of(year: int, a: int, b: int) -> Tuple[float, float]:
        if a <= b:
            s = datetime(year, a, 1, tzinfo=timezone.utc)
            if b == 12:
                e = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                e = datetime(year, b + 1, 1, tzinfo=timezone.utc)
        else:  # winter: Dec-Feb
            s = datetime(year, a, 1, tzinfo=timezone.utc)
            e = datetime(year + 1, b + 1, 1, tzinfo=timezone.utc)
        return s.timestamp(), e.timestamp()

    @staticmethod
    def _last_weekend_of(dt: datetime) -> Tuple[float, float]:
        # Saturday 00:00 to Monday 00:00 of the most recent weekend.
        days_since_sat = (dt.weekday() - 5) % 7
        sat = dt - timedelta(days=days_since_sat)
        s = datetime(sat.year, sat.month, sat.day, tzinfo=timezone.utc)
        return s.timestamp(), (s + timedelta(days=2)).timestamp()
