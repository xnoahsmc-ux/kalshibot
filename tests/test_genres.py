from kalshibot.genres import classify


def test_genre_classifier_known_tickers():
    assert classify("KXNFL-25SEP-WIN").genre == "sports"
    assert classify("KXWEATHERHIGH-NYC-2025").genre == "weather"
    assert classify("KXBTC-EOY-100K").genre == "crypto"
    assert classify("KXPOLPRES-2028-DEM").genre == "politics"
    assert classify("KXOSCAR-BESTPIC-25").genre == "entertainment"
    assert classify("UNKNOWN-TICKER-001").genre == "other"
