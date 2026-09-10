from datetime import date

from jkquant.holiday_risk import holiday_risk


def test_warns_two_days_before_2026_mid_autumn():
    result = holiday_risk(date(2026, 9, 23))
    assert result is not None
    assert result["level"] == "warning"
    assert result["days_ahead"] == 2


def test_ordinary_day_is_safe():
    result = holiday_risk(date(2026, 9, 10))
    assert result is not None
    assert result["level"] == "safe"
