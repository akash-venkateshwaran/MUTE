import os
import numpy as np
from rclpy.node import Node
from custom_interfaces.msg import State, ShipInfo, SPL
import rclpy

from models.jomopans_echo.model import jomopans_model



class SourceModel(Node):
    def __init__(self, model_type):
        super().__init__('source_model')
        self.setup_logger()
        self.verbose_logging = False
        self.ship_current_state = State()
        self.ship_info = ShipInfo()
        self.model_type = model_type
        # TODO Change freq defintion based on the transmission loss
        self.freq = np.array([12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000])
        
        self.source_noise_level = np.empty(0)
        self.source_noise_freq = np.empty(0)

        self.ship_current_state_subscription = self.create_subscription(
            State,
            'ship_state',
            self.ship_state_callback,
            10
        )

        self.ship_info_subscription = self.create_subscription(
            ShipInfo,
            'ship_info',
            self.ship_info_callback,
            10
        )

        self.spl_publisher = self.create_publisher(SPL, 'source_noise_spl', 10)
        self.publish_timer = self.create_timer(1.0, self.publish_spl_message)  # Publish every second


        

    def setup_logger(self):
        self.get_logger().info("SourceModel Node is started!")

    def ship_state_callback(self, msg):
        self.ship_current_state = msg
        if self.verbose_logging:
            self.get_logger().info(f"Received ship state: {msg}")

        self.compute_source_noise()

    def ship_info_callback(self, msg):
        self.ship_info = msg
        if self.verbose_logging:
            self.get_logger().info(f"Received ship info: {msg}")

    def compute_source_noise(self):
        ship_speed = self.ship_current_state.speed * 1.94384 # m/s to knots
        if ship_speed == 0:
            ship_speed = 0.001 # Avoid dividing by zero
        ship_type = self.ship_info.ship_type
        ship_length = self.ship_info.ship_length

        if ship_type and ship_length > 0:
            try:
                self.source_noise_level, self.source_noise_freq = jomopans_model(self.freq, ship_type, ship_speed, ship_length)

                if self.verbose_logging:
                    self.get_logger().info(f'Computed source noise levels: {self.source_noise_level}')
                    self.get_logger().info(f'Computed source noise frequencies: {self.source_noise_freq}')
            except ValueError as e:
                self.get_logger().error(f'Error computing source noise in py_source_model: {str(e)}')

    def publish_spl_message(self):
            if self.source_noise_level.size == 0 or self.source_noise_freq.size == 0:
                if self.verbose_logging:
                    self.get_logger().warning("SPL or frequency list is empty. Skipping publish.")
                return
            spl_msg = SPL()
            spl_msg.spl = self.source_noise_level.tolist()
            spl_msg.freq = self.source_noise_freq.tolist()
            self.spl_publisher.publish(spl_msg)
            if self.verbose_logging:
                self.get_logger().info(f'Published SPL message: SPL={spl_msg.spl}, Freq={spl_msg.freq}')


def main(args=None):
    rclpy.init(args=args)

    source_model = SourceModel(model_type="example_model_type")

    rclpy.spin(source_model)

    source_model.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
