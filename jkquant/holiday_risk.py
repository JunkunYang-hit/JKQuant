from __future__ import annotations

from datetime import date, timedelta


HOLIDAY_NAMES = {
    "New Year's Day": "元旦", "Spring Festival": "春节",
    "Tomb-sweeping Day": "清明节", "Labour Day": "劳动节",
    "Dragon Boat Festival": "端午节", "Mid-autumn Festival": "中秋节",
    "National Day": "国庆节",
}


def holiday_risk(day: date) -> dict[str, object] | None:
    """Return a two-day statutory-holiday warning, or a normal-day status.

    None means the installed calendar has no reliable information for the date.
    Ordinary weekends are deliberately not treated as statutory-holiday risk.
    """
    try:
        from chinese_calendar import get_holiday_detail

        today_is_holiday, today_name = get_holiday_detail(day)
        if today_is_holiday and today_name is not None:
            return None
        for days_ahead in (1, 2):
            is_holiday, holiday_name = get_holiday_detail(day + timedelta(days=days_ahead))
            if is_holiday and holiday_name is not None:
                raw_name = getattr(holiday_name, "value", str(holiday_name))
                name = HOLIDAY_NAMES.get(raw_name, raw_name)
                return {
                    "level": "warning", "days_ahead": days_ahead,
                    "holiday_name": name, "holiday_date": day + timedelta(days=days_ahead),
                }
        return {"level": "safe", "days_ahead": None, "holiday_name": None}
    except (ImportError, NotImplementedError, ValueError):
        return None
