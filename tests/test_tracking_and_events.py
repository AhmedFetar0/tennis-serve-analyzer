"""The Kalman tracker and the contact detector, on made-up detections.

These two stages work in pixels and frames, with no camera model, so their
tests are simple sequences of points with known behaviour.
"""

import numpy as np

from detect_events import find_contacts, find_contacts_with_evidence
from track_ball import run_filter

FPS = 240.0


def _projectile(frames, x0=400.0, y0=600.0, vx=3.0, vy=-6.0, ay=0.04):
    t = np.asarray(frames, dtype=float)
    return np.c_[x0 + vx * t, y0 + vy * t + 0.5 * ay * t * t]


def test_filter_fills_short_gaps_and_stays_on_the_ball():
    rng = np.random.default_rng(0)
    truth = _projectile(range(120))
    detections = [[(x + rng.normal(0, 1.5), y + rng.normal(0, 1.5), 0.6)]
                  for x, y in truth]
    for i in range(50, 58):                     # an 8-frame dropout
        detections[i] = []
    rows = run_filter(detections)

    assert all(rows[i]["status"] == "coasted" for i in range(50, 58))
    # Through the gap the filter keeps predicting, and the predictions stay
    # close to where the ball really was.
    for i in range(50, 58):
        error = np.hypot(float(rows[i]["filt_x"]) - truth[i, 0],
                         float(rows[i]["filt_y"]) - truth[i, 1])
        assert error < 25      # a ball-width or so; velocity noise grows with the gap
    assert rows[58]["status"] == "updated"


def test_filter_prefers_the_plausible_detection_over_the_confident_one():
    truth = _projectile(range(60))
    detections = [[(x, y, 0.4)] for x, y in truth]
    # A stationary look-alike with higher confidence, far from the path.
    for i in range(20, 60):
        detections[i].append((1500.0, 200.0, 0.9))
    rows = run_filter(detections)
    for i in range(25, 60):
        assert abs(float(rows[i]["raw_x"]) - truth[i, 0]) < 1e-6


def test_filter_rejects_a_lone_false_detection_and_goes_stale():
    truth = _projectile(range(40))
    detections = [[(x, y, 0.5)] for x, y in truth] + [[] for _ in range(60)]
    detections[45] = [(1700.0, 900.0, 0.8)]     # nothing like the ball's path
    # max_misses=120 is the command-line default (half a second at 240 fps).
    rows = run_filter(detections, max_misses=120)
    assert rows[45]["status"] == "rejected"
    # After max_coast misses no position is reported: a prediction that old
    # is a guess.
    assert rows[75]["status"] == "stale" and rows[75]["filt_x"] == ""


def _serve_track(contact=200, visible=1.0, seed=0):
    """Pixel track of a toss and a serve: up, down, then struck away.

    Returns the (frame, x, y) array that detect_events works on. `visible`
    is the fraction of toss frames that have a detection, except for the
    last 30 before contact, which are always seen. That is the pattern in the
    real far-end clip: the detector loses the small, slow ball near the top
    of the toss and picks it up again as it falls.
    """
    rng = np.random.default_rng(seed)
    rows = []
    apex = contact - 70
    for f in range(apex - 90, contact):          # toss: a parabola in y
        y = 300.0 + 0.02 * (f - apex) ** 2
        if f >= contact - 30 or rng.random() < visible:
            rows.append((f, 900.0 + rng.normal(0, 0.5), y + rng.normal(0, 0.5)))
    y_contact = 300.0 + 0.02 * 70 ** 2
    for f in range(contact, contact + 120):      # flight: fast and straight
        rows.append((f, 900.0 + 4.0 * (f - contact) + rng.normal(0, 0.5),
                     y_contact + 1.5 * (f - contact) + rng.normal(0, 0.5)))
    return np.array(rows)


def test_contact_is_found_at_the_strike():
    contacts = find_contacts(_serve_track(contact=200), FPS)
    assert len(contacts) == 1
    assert abs(contacts[0] - 200) <= 2


def test_a_ball_bounced_on_the_ground_is_not_a_serve():
    # Falls, hits the ground, comes back up: a bigger velocity jump than any
    # serve, but there was no toss before it and it reverses upward.
    rows = [(f, 500.0, 300.0 + 6.0 * f) for f in range(0, 80)]
    rows += [(f, 500.0, 780.0 - 5.0 * (f - 80)) for f in range(80, 160)]
    assert find_contacts(np.array(rows, dtype=float), FPS) == []


def test_partly_seen_toss_is_reported_as_sparse():
    # Most of the toss unseen: below the 60% coverage a clean toss needs.
    found = find_contacts_with_evidence(_serve_track(contact=200, visible=0.15, seed=4),
                                        FPS)
    assert len(found) == 1
    frame, evidence = found[0]
    assert abs(frame - 200) <= 3 and evidence == "sparse toss"
    # The fully seen toss is reported once, as clean.
    assert [e for _, e in find_contacts_with_evidence(_serve_track(), FPS)] == ["clean toss"]
