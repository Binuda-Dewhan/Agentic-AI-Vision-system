import pytest
from sleep_monitor.perception.target_selector import TargetSelector
from sleep_monitor.schemas.perception import BBox, TrackedPerson

def create_person(track_id, x, y):
    # Create a 10x10 bbox centered at x,y
    return TrackedPerson(
        track_id=track_id,
        bbox=BBox(x1=x-5, y1=y-5, x2=x+5, y2=y+5),
        confidence=0.9
    )

class TestTargetSelector:
    def test_target_remains_tracked_normally(self):
        selector = TargetSelector(mode="auto", reassociation_distance_px=100)
        
        # 1. Initialize target
        for i in range(3):
            selector.select_target(i, [create_person(1, 50, 50)])
        
        assert selector.primary_track_id == 1
        
        # 2. Track normally
        result = selector.select_target(3, [create_person(1, 55, 55)])
        assert result.persons[0].is_primary_target is True
        assert selector.primary_track_id == 1

    def test_reassociation_within_distance(self):
        selector = TargetSelector(mode="auto", reassociation_distance_px=100)
        
        # 1. Initialize target
        for i in range(3):
            selector.select_target(i, [create_person(1, 50, 50)])
            
        assert selector.primary_track_id == 1
        
        # 2. Track ID changes, but is close (dx=80, dy=0 -> dist=80 < 100)
        # We also simulate another person further away just to be sure it picks the right one
        persons = [
            create_person(2, 130, 50), # 80 px away (should reassociate)
            create_person(3, 300, 50)  # 250 px away
        ]
        result = selector.select_target(3, persons)
        
        assert selector.primary_track_id == 2
        assert result.persons[0].is_primary_target is True
        assert result.persons[1].is_primary_target is False

    def test_single_person_fallback_beyond_distance(self):
        selector = TargetSelector(mode="auto", reassociation_distance_px=100)
        
        # 1. Initialize target
        for i in range(3):
            selector.select_target(i, [create_person(1, 50, 50)])
            
        assert selector.primary_track_id == 1
        
        # 2. Track ID changes and is FAR (dx=200, dy=0 -> dist=200 > 100)
        # BUT it is the ONLY person in the frame.
        persons = [create_person(2, 250, 50)]
        result = selector.select_target(3, persons)
        
        assert selector.primary_track_id == 2
        assert result.persons[0].is_primary_target is True

    def test_no_fallback_multiple_people_beyond_distance(self):
        selector = TargetSelector(mode="auto", reassociation_distance_px=100)
        
        # 1. Initialize target
        for i in range(3):
            selector.select_target(i, [create_person(1, 50, 50)])
            
        assert selector.primary_track_id == 1
        
        # 2. Target lost, multiple people appear, ALL are far away (>100px)
        persons = [
            create_person(2, 250, 50),
            create_person(3, 300, 50)
        ]
        result = selector.select_target(3, persons)
        
        # Since there are multiple people and none are within reassociation distance,
        # it should NOT arbitrarily pick one. The target remains lost.
        assert selector.primary_track_id == 1 # Still looks for track 1 next time
        assert result.persons[0].is_primary_target is False
        assert result.persons[1].is_primary_target is False

    def test_genuinely_no_person(self):
        selector = TargetSelector(mode="auto", reassociation_distance_px=100)
        
        # 1. Initialize target
        for i in range(3):
            selector.select_target(i, [create_person(1, 50, 50)])
            
        assert selector.primary_track_id == 1
        
        # 2. No persons detected
        result = selector.select_target(3, [])
        
        assert len(result.persons) == 0
        assert selector.primary_track_id == 1 # State unchanged, waiting for target to return
