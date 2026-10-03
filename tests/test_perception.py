from sleep_monitor.perception.target_selector import TargetSelector
from sleep_monitor.schemas.perception import BBox, TrackedPerson


def test_bbox_properties():
    bbox = BBox(x1=10, y1=20, x2=30, y2=40)
    assert bbox.center_x == 20
    assert bbox.center_y == 30
    assert bbox.area == 20 * 20


def test_target_selector_auto_mode():
    selector = TargetSelector(mode="auto")

    # Frame 1: Person 1 appears
    p1_f1 = TrackedPerson(
        track_id=1, bbox=BBox(x1=0, y1=0, x2=10, y2=10), confidence=0.9
    )
    res = selector.select_target(0, [p1_f1])
    assert selector.primary_track_id is None  # Needs 3 frames to stabilize

    # Frame 2: Person 1 still there
    p1_f2 = TrackedPerson(
        track_id=1, bbox=BBox(x1=0, y1=0, x2=10, y2=10), confidence=0.9
    )
    res = selector.select_target(1, [p1_f2])
    assert selector.primary_track_id is None

    # Frame 3: Person 1 becomes stable
    p1_f3 = TrackedPerson(
        track_id=1, bbox=BBox(x1=0, y1=0, x2=10, y2=10), confidence=0.9
    )
    res = selector.select_target(2, [p1_f3])
    assert selector.primary_track_id == 1
    assert res.persons[0].is_primary_target is True

    # Frame 4: Caregiver (Person 2) appears
    p1_f4 = TrackedPerson(
        track_id=1, bbox=BBox(x1=0, y1=0, x2=10, y2=10), confidence=0.9
    )
    p2_f4 = TrackedPerson(
        track_id=2, bbox=BBox(x1=100, y1=100, x2=150, y2=150), confidence=0.8
    )
    res = selector.select_target(3, [p1_f4, p2_f4])
    assert selector.primary_track_id == 1
    assert res.persons[0].is_primary_target is True
    assert res.persons[1].is_primary_target is False
    assert res.other_persons_present is True


def test_target_selector_spatial_disambiguation():
    selector = TargetSelector(mode="auto")

    # Stabilize Person 1
    for i in range(3):
        p1 = TrackedPerson(
            track_id=1, bbox=BBox(x1=10, y1=10, x2=20, y2=20), confidence=0.9
        )
        selector.select_target(i, [p1])

    assert selector.primary_track_id == 1

    # Frame 4: Person 1 track is lost, but a new track (ID 3) appears very close
    p3 = TrackedPerson(
        track_id=3, bbox=BBox(x1=12, y1=12, x2=22, y2=22), confidence=0.9
    )
    res = selector.select_target(3, [p3])

    # Should reassign target to ID 3 due to spatial proximity
    assert selector.primary_track_id == 3
    assert res.persons[0].is_primary_target is True
