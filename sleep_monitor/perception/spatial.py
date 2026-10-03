import math

from shapely.geometry import Polygon, box

from sleep_monitor.schemas.perception import BBox, SpatialFeatures


class SpatialAnalyzer:
    """Analyzes the spatial relationship between a person and the bed region."""

    def __init__(
        self,
        bed_polygon_points: list[tuple[int, int]] | None = None,
        edge_margin: int = 20,
    ):
        self.is_configured = (
            bed_polygon_points is not None and len(bed_polygon_points) >= 3
        )
        if self.is_configured:
            self.bed_poly = Polygon(bed_polygon_points)
            self.edge_margin = edge_margin
            self.bed_center = self.bed_poly.centroid
        else:
            self.bed_poly = None
            self.bed_center = None

    def analyze(self, bbox: BBox) -> SpatialFeatures:
        """Calculate overlap, distance, and relative position to bed."""
        if not self.is_configured:
            return SpatialFeatures(
                bed_overlap_ratio=0.0,
                distance_to_bed_center=-1.0,
                position_relative_to_bed="UNCONFIGURED",
            )

        person_box = box(bbox.x1, bbox.y1, bbox.x2, bbox.y2)

        # Calculate overlap
        intersection = self.bed_poly.intersection(person_box)
        overlap_ratio = (
            intersection.area / person_box.area if person_box.area > 0 else 0.0
        )

        # Calculate distance
        dist = math.sqrt(
            (bbox.center_x - self.bed_center.x) ** 2
            + (bbox.center_y - self.bed_center.y) ** 2
        )

        # Determine position
        if overlap_ratio > 0.6:
            position = "INSIDE"
        elif (
            overlap_ratio > 0.1 or self.bed_poly.distance(person_box) < self.edge_margin
        ):
            position = "ON_EDGE"
        else:
            position = "OUTSIDE"

        return SpatialFeatures(
            bed_overlap_ratio=overlap_ratio,
            distance_to_bed_center=dist,
            position_relative_to_bed=position,
        )
