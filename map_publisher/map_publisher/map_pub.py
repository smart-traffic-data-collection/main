import rclpy            # Importiert die ROS2 Python Client Library
from rclpy.node import Node         # Importiert die Node-Basisklasse
from visualization_msgs.msg import Marker, MarkerArray
import lanelet2         # Importiert die Lanelet2 Bibliothek
from lanelet2.io import load        # Importiert Funktion zum Laden von Lanelet2 Maps
from lanelet2.projection import UtmProjector        # Importiert UTM-Projektor für Koordinatenumrechnung

class LaneletVisualizer(Node):          # Definiert die Klasse LaneletVisualizer, die von Node erbt
    def __init__(self):         # Konstruktor-Methode der Klasse
        super().__init__('lanelet_visualizer') # Initialisiert die Basisklasse Node mit dem Namen 'lanelet_visualizer'

        self.pub = self.create_publisher(MarkerArray, '/lanelet_map', 10)   # Erstellt einen Publisher für MarkerArray auf dem Topic '/lanelet_map'

        origin = lanelet2.io.Origin(48.77106330244843, 11.439444972723058)  # Definiert den geografischen Ursprung (Breitengrad, Längengrad) für die Projektion
        projector = UtmProjector(origin)        # Erstellt einen UTM-Projektor basierend auf dem definierten Ursprung

        self.map = load("/home/adrian/Schreibtisch/ros2_ws/src/map_publisher/map_publisher/crossings_lanelet2map.osm", projector)         # Lädt die Lanelet2-Karte aus der angegebenen OSM-Datei

        # DEBUG: Welche Datei wird geladen?
        self.get_logger().info("Loaded map: /home/adrian/ros2_ws/src/map_test/crossings_lanelet2map.osm")     # Loggt den Pfad der geladenen Karte
        
        # DEBUG: Anzahl Lanelets
        lanelet_count = len(self.map.laneletLayer)      # Ermittelt die Anzahl der Lanelets in der Karte
        self.get_logger().info(f"Number of lanelets: {lanelet_count}")          # Loggt die Anzahl der gefundenen Lanelets
        
        # Wenn keine Lanelets vorhanden sind -> sofort abbrechen
        if lanelet_count == 0:          # Überprüft, ob die Lanelet-Ebene leer ist
            self.get_logger().error("NO LANELETS FOUND → Check your OSM file!")     # Loggt eine Fehlermeldung bei leeren Karten
        else:
            # 🔍 Details zu jedem Lanelet
            for ll in self.map.laneletLayer:        # Iteriert über jedes einzelne Lanelet in der Karte
                self.get_logger().info(f"Lanelet ID {ll.id} | left pts: {len(ll.leftBound)} | right pts: {len(ll.rightBound)}"
                )

        self.timer = self.create_timer(1.0, self.publish_map)       # Erstellt einen Timer, der jede Sekunde die Funktion publish_map aufruft ANPASSBAR

    def publish_map(self):

        marker_array = MarkerArray()        # Erstellt ein neues MarkerArray

        # Ursprung
        first_lanelet = next(iter(self.map.laneletLayer))
        origin_x = first_lanelet.centerline[0].x
        origin_y = first_lanelet.centerline[0].y

        from geometry_msgs.msg import Point

        for i, lanelet in enumerate(self.map.laneletLayer):

            # -------- CENTERLINE --------
            center_marker = Marker()
            center_marker.header.frame_id = "map"
            center_marker.id = i
            center_marker.type = Marker.LINE_STRIP
            center_marker.scale.x = 0.2
            center_marker.color.g = 1.0
            center_marker.color.a = 1.0

            for pt in lanelet.centerline:
                p = Point()
                p.x = pt.x - origin_x
                p.y = pt.y - origin_y
                p.z = 0.0
                center_marker.points.append(p)

            marker_array.markers.append(center_marker)

            # -------- LEFT BOUND --------
            left_marker = Marker()
            left_marker.header.frame_id = "map"
            left_marker.id = i + 1000
            left_marker.type = Marker.LINE_STRIP
            left_marker.scale.x = 0.1
            left_marker.color.r = 1.0
            left_marker.color.a = 1.0

            for pt in lanelet.leftBound:
                p = Point()
                p.x = pt.x - origin_x
                p.y = pt.y - origin_y
                p.z = 0.0
                left_marker.points.append(p)

            marker_array.markers.append(left_marker)

            # -------- RIGHT BOUND --------
            right_marker = Marker()
            right_marker.header.frame_id = "map"
            right_marker.id = i + 2000
            right_marker.type = Marker.LINE_STRIP
            right_marker.scale.x = 0.1
            right_marker.color.b = 1.0
            right_marker.color.a = 1.0

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

if __name__ == '__main__':
    main()
