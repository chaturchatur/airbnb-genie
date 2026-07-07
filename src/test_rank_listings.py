"""
test_rank_listings.py

tiny synthetic-data smoke test for the flagging + threshold logic in
rank_listings.py. it does NOT need the real data or the trained model -- it
just feeds hand-made numbers through the pure functions so we can prove the
over/underpriced math and the threshold sweep behave exactly as intended.

run: python src/test_rank_listings.py
"""

import numpy as np
import pandas as pd

from rank_listings import flag_listings, run_threshold_sensitivity


def make_fake_data():
    """
    three listings with known, easy-to-check gaps vs a fixed predicted price:
      A: actual 130 vs predicted 100 -> +30% -> overpriced
      B: actual 100 vs predicted 100 ->   0% -> fair
      C: actual  70 vs predicted 100 -> -30% -> underpriced
    """
    meta = pd.DataFrame({
        'id': ['A', 'B', 'C'],
        'neighbourhood': ['x', 'y', 'z'],
        'actual': [130.0, 100.0, 70.0],
    })
    predicted = np.array([100.0, 100.0, 100.0])
    return meta, predicted


def test_flagging():
    meta, predicted = make_fake_data()
    tbl = flag_listings(meta, predicted, threshold=0.15)

    # ranking is sorted by pct_diff descending: A (over), B (fair), C (under).
    flags_by_id = dict(zip(tbl['id'], tbl['flag']))
    assert flags_by_id['A'] == 'overpriced', flags_by_id
    assert flags_by_id['B'] == 'fair', flags_by_id
    assert flags_by_id['C'] == 'underpriced', flags_by_id

    # pct_diff math: (130-100)/100 = 0.30 for A.
    pct_by_id = dict(zip(tbl['id'], tbl['pct_diff']))
    assert abs(pct_by_id['A'] - 0.30) < 1e-9, pct_by_id
    assert abs(pct_by_id['C'] + 0.30) < 1e-9, pct_by_id

    # sorted so most overpriced is first, best deal is last.
    assert tbl.iloc[0]['id'] == 'A'
    assert tbl.iloc[-1]['id'] == 'C'
    print("PASS: flagging + pct_diff + ranking")


def test_threshold_stability():
    meta, predicted = make_fake_data()
    summary = run_threshold_sensitivity(meta, predicted, grid=[0.10, 0.20, 0.40])

    # at cutoff 0.40, the +/-30% listings fall inside the band -> all "fair".
    row_040 = summary[summary['threshold'] == 0.40].iloc[0]
    assert row_040['n_overpriced'] == 0
    assert row_040['n_underpriced'] == 0
    assert row_040['n_fair'] == 3

    # at cutoff 0.10, A is over and C is under.
    row_010 = summary[summary['threshold'] == 0.10].iloc[0]
    assert row_010['n_overpriced'] == 1
    assert row_010['n_underpriced'] == 1
    print("PASS: threshold sensitivity counts")


if __name__ == "__main__":
    test_flagging()
    test_threshold_stability()
    print("all smoke tests passed.")
