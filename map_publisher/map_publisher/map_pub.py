import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import DurabilityPolicy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import lanelet2
from lanelet2.io import load
from lanelet2.projection import UtmProjector


class LaneletVisualizer(Node):

    def __init__(self):
        super().__init__('lanelet_visualizer')

        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.pub = self.create_publisher(
            MarkerArray,
            '/lanelet_map',
            qos_profile
        )

        origin = lanelet2.io.Origin(
            48.77106330244843,
            11.439444972723058
        )

        projector = UtmProjector(origin)

        self.map = load(
            "/home/adrian/Schreibtisch/ros2_ws/src/map_publisher/map_publisher/crossings_lanelet2map.osm",
            projector
        )

        self.get_logger().info(
            "Loaded map: /home/adrian/Schreibtisch/ros2_ws/src/map_publisher/map_publisher/crossings_lanelet2map.osm"
        )

        lanelet_count = len(self.map.laneletLayer)

        self.get_logger().info(
            f"Number of lanelets: {lanelet_count}"
        )

        if lanelet_count == 0:
            self.get_logger().error(
                "NO LANELETS FOUND → Check your OSM file!"
            )
        else:
            for ll in self.map.laneletLayer:

                subtype = (
                    ll.attributes["subtype"]
                    if "subtype" in ll.attributes
                    else "unknown"
                )

                self.get_logger().info(
                    f"Lanelet ID {ll.id} | subtype: {subtype}"
                )

        self.publish_map()

    def get_lanelet_color(self, subtype):
        """
        Returns RGB color tuple based on lanelet subtype.
        """

        # Vehicle lanes
        if subtype == "road":
            return (1.0, 0.0, 0.0)      # Red

        # Bicycle infrastructure
        elif subtype == "bicycle_lane":
            return (0.0, 0.0, 1.0)      # Blue

        # Pedestrian infrastructure
        elif subtype in [
            "walkway",
            "shared_walkway",
            "crosswalk"
        ]:
            return (0.0, 1.0, 0.0)      # Green

        # Parking
        elif subtype == "parking":
            return (0.5, 0.5, 0.5)      # Grey

        # Unknown
        else:
            return (1.0, 1.0, 1.0)      # White

    def publish_map(self):

        marker_array = MarkerArray()

        first_lanelet = next(iter(self.map.laneletLayer))

        origin_x = first_lanelet.centerline[0].x
        origin_y = first_lanelet.centerline[0].y

        for i, lanelet in enumerate(self.map.laneletLayer):

            subtype = (
                lanelet.attributes["subtype"]
                if "subtype" in lanelet.attributes
                else ""
            )

            r, g, b = self.get_lanelet_color(subtype)

            # ------------------------------
            # CENTERLINE
            # ------------------------------
            center_marker = Marker()

            center_marker.header.frame_id = "map"
            center_marker.ns = "centerlines"

            center_marker.id = i

            center_marker.type = Marker.LINE_STRIP
            center_marker.action = Marker.ADD

            center_marker.scale.x = 0.25

            center_marker.color.r = r
            center_marker.color.g = g
            center_marker.color.b = b
            center_marker.color.a = 1.0

            for pt in lanelet.centerline:

                p = Point()

                p.x = pt.x - origin_x
                p.y = pt.y - origin_y
                p.z = 0.0

                center_marker.points.append(p)

            marker_array.markers.append(center_marker)

            # ------------------------------
            # LEFT BOUNDARY
            # ------------------------------
            left_marker = Marker()

            left_marker.header.frame_id = "map"
            left_marker.ns = "left_bounds"

            left_marker.id = i + 1000

            left_marker.type = Marker.LINE_STRIP
            left_marker.action = Marker.ADD

            left_marker.scale.x = 0.05

            left_marker.color.r = r
            left_marker.color.g = g
            left_marker.color.b = b
            left_marker.color.a = 0.6

            for pt in lanelet.leftBound:

                p = Point()

                p.x = pt.x - origin_x
                p.y = pt.y - origin_y
                p.z = 0.0

                left_marker.points.append(p)

            marker_array.markers.append(left_marker)

            # ------------------------------
            # RIGHT BOUNDARY
            # ------------------------------
            right_marker = Marker()

            right_marker.header.frame_id = "map"
            right_marker.ns = "right_bounds"

            right_marker.id = i + 2000

            right_marker.type = Marker.LINE_STRIP
            right_marker.action = Marker.ADD

            right_marker.scale.x = 0.05

            right_marker.color.r = r
            right_marker.color.g = g
            right_marker.color.b = b
            right_marker.color.a = 0.6

            for pt in lanelet.rightBound:

                p = Point()

                p.x = pt.x - origin_x
                p.y = pt.y - origin_y
                p.z = 0.0

                right_marker.points.append(p)

            marker_array.markers.append(right_marker)

        self.pub.publish(marker_array)


def main():
    rclpy.init()

    node = LaneletVisualizer()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()