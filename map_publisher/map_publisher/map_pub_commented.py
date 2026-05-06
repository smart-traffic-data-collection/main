import rclpy            # Importiert die ROS2 Python Client Library[cite: 1]
from rclpy.node import Node         # Importiert die Node-Basisklasse[cite: 1]
from visualization_msgs.msg import Marker, MarkerArray # Importiert Nachrichtentypen für die Visualisierung in RViz[cite: 1]
import lanelet2         # Importiert die Lanelet2 Bibliothek[cite: 1]
from lanelet2.io import load        # Importiert Funktion zum Laden von Lanelet2 Maps[cite: 1]
from lanelet2.projection import UtmProjector        # Importiert UTM-Projektor für Koordinatenumrechnung[cite: 1]

class LaneletVisualizer(Node): # Definiert die Klasse LaneletVisualizer, die von Node erbt[cite: 1]
    def __init__(self): # Konstruktor-Methode der Klasse[cite: 1]
        super().__init__('lanelet_visualizer') # Initialisiert die Basisklasse Node mit dem Namen 'lanelet_visualizer'[cite: 1]

        self.pub = self.create_publisher(MarkerArray, '/lanelet_map', 10) # Erstellt einen Publisher für MarkerArray auf dem Topic '/lanelet_map'[cite: 1]

        origin = lanelet2.io.Origin(49.0, 11.0) # Definiert den geografischen Ursprung (Breitengrad, Längengrad) für die Projektion[cite: 1]
        projector = UtmProjector(origin) # Erstellt einen UTM-Projektor basierend auf dem definierten Ursprung[cite: 1]

        self.map = load("/home/adrian/ros2_ws/src/map_test/map.osm", projector) # Lädt die Lanelet2-Karte aus der angegebenen OSM-Datei[cite: 1]

        self.map = load("/home/adrian/ros2_ws/src/map_test/map.osm", projector) # Lädt die Karte erneut (redundante Zeile im Originalcode)[cite: 1]

        # DEBUG: Welche Datei wird geladen?
        self.get_logger().info("Loaded map: /home/adrian/ros2_ws/src/map_test/map.osm") # Loggt den Pfad der geladenen Karte[cite: 1]
        
        # DEBUG: Anzahl Lanelets
        lanelet_count = len(self.map.laneletLayer) # Ermittelt die Anzahl der Lanelets in der Karte[cite: 1]
        self.get_logger().info(f"Number of lanelets: {lanelet_count}") # Loggt die Anzahl der gefundenen Lanelets[cite: 1]
        
        # ❌ Wenn keine Lanelets vorhanden sind → sofort abbrechen
        if lanelet_count == 0: # Überprüft, ob die Lanelet-Ebene leer ist[cite: 1]
            self.get_logger().error("❌ NO LANELETS FOUND → Check your OSM file!") # Loggt eine Fehlermeldung bei leeren Karten[cite: 1]
        else: # Falls Lanelets vorhanden sind[cite: 1]
            # 🔍 Details zu jedem Lanelet
            for ll in self.map.laneletLayer: # Iteriert über jedes einzelne Lanelet in der Karte[cite: 1]
                self.get_logger().info(f"Lanelet ID {ll.id} | left pts: {len(ll.leftBound)} | right pts: {len(ll.rightBound)}" # Loggt ID und Punktanzahl der Begrenzungen[cite: 1]
                )

        self.timer = self.create_timer(1.0, self.publish_map) # Erstellt einen Timer, der jede Sekunde die Funktion publish_map aufruft[cite: 1]

    def publish_map(self): # Funktion zum Aufbereiten und Senden der Kartendaten[cite: 1]

        marker_array = MarkerArray() # Erstellt ein neues MarkerArray-Objekt[cite: 1]

        # Ursprung
        first_lanelet = next(iter(self.map.laneletLayer)) # Holt das erste Lanelet aus der Karte für die lokale Referenz[cite: 1]
        origin_x = first_lanelet.centerline[0].x # Setzt die X-Koordinate des ersten Punktes der Mittellinie als Ursprung[cite: 1]
        origin_y = first_lanelet.centerline[0].y # Setzt die Y-Koordinate des ersten Punktes der Mittellinie als Ursprung[cite: 1]

        from geometry_msgs.msg import Point # Importiert lokal den Point-Nachrichtentyp[cite: 1]

        for i, lanelet in enumerate(self.map.laneletLayer): # Iteriert mit Index über alle Lanelets[cite: 1]

            # -------- CENTERLINE --------
            center_marker = Marker() # Erstellt einen neuen Marker für die Mittellinie[cite: 1]
            center_marker.header.frame_id = "map" # Setzt den Koordinatenrahmen auf "map"[cite: 1]
            center_marker.id = i # Weist dem Marker eine eindeutige ID basierend auf dem Index zu[cite: 1]
            center_marker.type = Marker.LINE_STRIP # Definiert den Marker-Typ als Linienzug[cite: 1]
            center_marker.scale.x = 0.2 # Setzt die Breite der Linie auf 0.2 Meter[cite: 1]
            center_marker.color.g = 1.0 # Setzt die Farbe der Mittellinie auf Grün[cite: 1]
            center_marker.color.a = 1.0 # Setzt die Deckkraft (Alpha) auf 100%[cite: 1]

            for pt in lanelet.centerline: # Iteriert über alle Punkte der Mittellinie des aktuellen Lanelets[cite: 1]
                p = Point() # Erstellt ein neues Punkt-Objekt[cite: 1]
                p.x = pt.x - origin_x # Berechnet die relative X-Koordinate zum lokalen Ursprung[cite: 1]
                p.y = pt.y - origin_y # Berechnet die relative Y-Koordinate zum lokalen Ursprung[cite: 1]
                p.z = 0.0 # Setzt die Z-Koordinate auf Null[cite: 1]
                center_marker.points.append(p) # Fügt den Punkt der Punktliste des Markers hinzu[cite: 1]

            marker_array.markers.append(center_marker) # Fügt den fertigen Mittellinien-Marker dem MarkerArray hinzu[cite: 1]

            # -------- LEFT BOUND --------
            left_marker = Marker() # Erstellt einen neuen Marker für die linke Begrenzung[cite: 1]
            left_marker.header.frame_id = "map" # Setzt den Koordinatenrahmen auf "map"[cite: 1]
            left_marker.id = i + 1000 # Weist eine ID im Bereich ab 1000 zu, um Konflikte zu vermeiden[cite: 1]
            left_marker.type = Marker.LINE_STRIP # Definiert den Marker-Typ als Linienzug[cite: 1]
            left_marker.scale.x = 0.1 # Setzt die Linienbreite auf 0.1 Meter[cite: 1]
            left_marker.color.r = 1.0 # Setzt die Farbe der linken Begrenzung auf Rot[cite: 1]
            left_marker.color.a = 1.0 # Setzt die Deckkraft auf 100%[cite: 1]

            for pt in lanelet.leftBound: # Iteriert über alle Punkte der linken Begrenzung[cite: 1]
                p = Point() # Erstellt ein neues Punkt-Objekt[cite: 1]
                p.x = pt.x - origin_x # Berechnet die relative X-Koordinate[cite: 1]
                p.y = pt.y - origin_y # Berechnet die relative Y-Koordinate[cite: 1]
                p.z = 0.0 # Setzt die Z-Koordinate auf Null[cite: 1]
                left_marker.points.append(p) # Fügt den Punkt der Punktliste des Markers hinzu[cite: 1]

            marker_array.markers.append(left_marker) # Fügt den linken Begrenzungs-Marker dem MarkerArray hinzu[cite: 1]

            # -------- RIGHT BOUND --------
            right_marker = Marker() # Erstellt einen neuen Marker für die rechte Begrenzung[cite: 1]
            right_marker.header.frame_id = "map" # Setzt den Koordinatenrahmen auf "map"[cite: 1]
            right_marker.id = i + 2000 # Weist eine ID im Bereich ab 2000 zu[cite: 1]
            right_marker.type = Marker.LINE_STRIP # Definiert den Marker-Typ als Linienzug[cite: 1]
            right_marker.scale.x = 0.1 # Setzt die Linienbreite auf 0.1 Meter[cite: 1]
            right_marker.color.b = 1.0 # Setzt die Farbe der rechten Begrenzung auf Blau[cite: 1]
            right_marker.color.a = 1.0 # Setzt die Deckkraft auf 100%[cite: 1]

            for pt in lanelet.rightBound: # Iteriert über alle Punkte der rechten Begrenzung[cite: 1]
                p = Point() # Erstellt ein neues Punkt-Objekt[cite: 1]
                p.x = pt.x - origin_x # Berechnet die relative X-Koordinate[cite: 1]
                p.y = pt.y - origin_y # Berechnet die relative Y-Koordinate[cite: 1]
                p.z = 0.0 # Setzt die Z-Koordinate auf Null[cite: 1]
                right_marker.points.append(p) # Fügt den Punkt der Punktliste des Markers hinzu[cite: 1]

            marker_array.markers.append(right_marker) # Fügt den rechten Begrenzungs-Marker dem MarkerArray hinzu[cite: 1]

        self.pub.publish(marker_array) # Veröffentlicht das gesamte MarkerArray für RViz[cite: 1]
        
def main(): # Hauptfunktion zum Starten des Nodes[cite: 1]
    rclpy.init() # Initialisiert das ROS2-Kommunikationssystem[cite: 1]
    node = LaneletVisualizer() # Instanziiert die Visualizer-Klasse[cite: 1]
    rclpy.spin(node) # Hält den Node aktiv und verarbeitet Callbacks, bis er gestoppt wird[cite: 1]

if __name__ == '__main__': # Überprüft, ob das Skript direkt ausgeführt wird[cite: 1]
    main() # Ruft die Hauptfunktion auf[cite: 1]