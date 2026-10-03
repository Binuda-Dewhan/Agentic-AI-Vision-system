from sleep_monitor.perception.observation_builder import ObservationBuilder
from sleep_monitor.perception.pose import PoseEstimator
from sleep_monitor.perception.spatial import SpatialAnalyzer
from sleep_monitor.schemas.perception import BBox, PoseKeypoints, SpatialFeatures


def test_spatial_analyzer():
    # Bed is from x=100 to 200, y=100 to 200
    bed_poly = [(100, 100), (200, 100), (200, 200), (100, 200)]
    analyzer = SpatialAnalyzer(bed_polygon_points=bed_poly, edge_margin=20)

    assert analyzer.is_configured is True

    # Person fully inside
    bbox_inside = BBox(x1=120, y1=120, x2=180, y2=180)
    feat_inside = analyzer.analyze(bbox_inside)
    assert feat_inside.position_relative_to_bed == "INSIDE"
    assert feat_inside.bed_overlap_ratio == 1.0

    # Person slightly overlapping (on edge)
    bbox_edge = BBox(x1=180, y1=180, x2=240, y2=240)
    feat_edge = analyzer.analyze(bbox_edge)
    assert feat_edge.position_relative_to_bed == "ON_EDGE"

    # Person outside but close
    bbox_close = BBox(x1=210, y1=210, x2=240, y2=240)
    feat_close = analyzer.analyze(bbox_close)
    assert feat_close.position_relative_to_bed == "ON_EDGE"  # within 20px margin

    # Person far outside
    bbox_far = BBox(x1=300, y1=300, x2=350, y2=350)
    feat_far = analyzer.analyze(bbox_far)
    assert feat_far.position_relative_to_bed == "OUTSIDE"
    assert feat_far.bed_overlap_ratio == 0.0


def test_spatial_analyzer_unconfigured():
    analyzer = SpatialAnalyzer(bed_polygon_points=None)
    assert analyzer.is_configured is False

    bbox = BBox(x1=10, y1=10, x2=20, y2=20)
    feat = analyzer.analyze(bbox)
    assert feat.position_relative_to_bed == "UNCONFIGURED"


def test_pose_orientation_classification():
    # Using a dummy instance just to test the classification logic
    estimator = PoseEstimator(model_path="yolov8n-pose.pt")

    # Horizontal: shoulders and hips are aligned horizontally
    # ls(10, 100), rs(10, 120) -> shoulder_y = 110, shoulder_x = 10
    # lh(100, 100), rh(100, 120) -> hip_y = 110, hip_x = 100
    # dx = 90, dy = 0 -> angle = 0
    kpts_horizontal = [(0, 0, 0)] * 17
    kpts_horizontal[5] = (10, 100, 0.9)
    kpts_horizontal[6] = (10, 120, 0.9)
    kpts_horizontal[11] = (100, 100, 0.9)
    kpts_horizontal[12] = (100, 120, 0.9)

    assert estimator._classify_orientation(kpts_horizontal) == "HORIZONTAL"

    # Upright: shoulders above hips
    kpts_upright = [(0, 0, 0)] * 17
    kpts_upright[5] = (100, 10, 0.9)
    kpts_upright[6] = (120, 10, 0.9)
    kpts_upright[11] = (100, 100, 0.9)
    kpts_upright[12] = (120, 100, 0.9)

    assert estimator._classify_orientation(kpts_upright) == "UPRIGHT"


def test_observation_builder_evidence():
    builder = ObservationBuilder()

    # Inside + Horizontal -> High Lying evidence
    pose = PoseKeypoints(keypoints=[], orientation="HORIZONTAL")
    spatial = SpatialFeatures(position_relative_to_bed="INSIDE")
    ev = builder.build_evidence(pose, spatial, movement_mag=0.0)
    assert ev.lying == 0.9
    assert ev.standing == 0.0

    # Outside + Upright + Moving -> High Walking evidence
    pose = PoseKeypoints(keypoints=[], orientation="UPRIGHT")
    spatial = SpatialFeatures(position_relative_to_bed="OUTSIDE")
    ev = builder.build_evidence(pose, spatial, movement_mag=25.0)
    assert ev.walking == 0.9
    assert ev.standing == 0.3

    # Outside + Upright + Not Moving -> High Standing evidence
    ev = builder.build_evidence(pose, spatial, movement_mag=5.0)
    assert ev.standing == 0.8
    assert ev.walking == 0.0
