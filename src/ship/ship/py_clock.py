import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
import time

class ClockPublisher(Node):
    def __init__(self):
        super().__init__('clock_publisher')
        self.publisher_ = self.create_publisher(Clock, '/clock', 10)
        self.timer = self.create_timer(0.1, self.publish_clock)  # Publish every 0.1 seconds
        self.start_time = time.time()

    def publish_clock(self):
        msg = Clock()
        current_time = time.time() - self.start_time
        msg.clock.sec = int(current_time)
        msg.clock.nanosec = int((current_time - msg.clock.sec) * 1e9)
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing time: {msg.clock.sec}.{msg.clock.nanosec}')