from kalshibot.fair_value import (
    blend,
    extract_team_codes,
    parse_weather_ticker,
)


def test_parse_weather_ticker_above():
    r = parse_weather_ticker("KXHIGHLAX-26APR22-T72", title="Will LA high be above 72?")
    assert r is not None
    assert r["city"]["code"] == "LAX"
    assert r["lo"] == 72
    assert r["direction"] == ">"


def test_parse_weather_ticker_below():
    r = parse_weather_ticker("KXHIGHLAX-26APR22-B64", title="Will LA high be below 64?")
    assert r is not None
    assert r["direction"] == "<"
    assert r["lo"] == 64


def test_parse_weather_ticker_range():
    r = parse_weather_ticker("KXHIGHLAX-26APR21-B65-66",
                              title="Will LA high be between 65 and 66?")
    assert r is not None
    assert r["direction"] in ("between", "<")


def test_extract_team_codes():
    assert "DAL" in extract_team_codes("KXNFLGAME-25-SFO-DAL")
    assert extract_team_codes("KXHIGHNY-26APR22-T64") == []


def test_blend_weights():
    # With no fair value, blend returns ensemble unchanged.
    assert blend(0.7, None) == 0.7
