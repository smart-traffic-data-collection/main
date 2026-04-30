import rclpy, requests, json
from rclpy.node import Node
from std_msgs.msg import String

class IngolstadtWeatherNode(Node):
    def __init__(self):
        super().__init__('weather_node_ingolstadt')
        self.pub = self.create_publisher(String, '/verkehr/ingolstadt/wetter', 10)
        self.url = "https://api.open-meteo.com/v1/forecast?latitude=48.7665&longitude=11.4257&current=temperature_2m,precipitation,visibility&timezone=Europe%2FBerlin"
        
        # Speicher für den letzten Stand der Daten
        self.last_weather_data = None
        
        # Timer auf 20 Sekunden gestellt
        self.timer = self.create_timer(20.0, self.fetch_data)
        self.get_logger().info('Wetter-Node aktiv (Delta-Modus, 20s Intervall).')

    def fetch_data(self):
        try:
            res = requests.get(self.url, timeout=10)
            res.raise_for_status()
            cur = res.json().get('current', {})
            
            payload = {
                "temp_c": cur.get("temperature_2m"),
                "regen_mm": cur.get("precipitation"),
                "sicht_m": cur.get("visibility")
            }
            
            # PRÜFUNG: Haben sich die Daten geändert?
            if payload != self.last_weather_data:
                msg = String()
                msg.data = json.dumps(payload)
                self.pub.publish(msg)
                
                self.get_logger().info(f'DATEN GEÄNDERT - Publiziert: {msg.data}')
                
                # Neuen Stand speichern
                self.last_weather_data = payload
            else:
                # Optional: Ein kleiner Log, dass nichts passiert ist
                self.get_logger().info('Keine Änderung der Wetterdaten - Publish übersprungen.')

        except Exception as e:
            self.get_logger().error(f'Fehler: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = IngolstadtWeatherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
