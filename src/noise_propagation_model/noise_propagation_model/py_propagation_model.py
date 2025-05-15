import rclpy
from rclpy.node import Node


class PropagationModel(Node):

        def __init__(self, model_type, tl_data):
                super().__init__('propagation_model')
                self.setup_logger()
                self.model_type = model_type
                self.tl_data = tl_data
                self.get_logger().info(f"TL Data: {self.tl_data}")

        def setup_logger(self):
                self.get_logger().info("PropagationModel Node is started!")
                self.get_logger().info(f"Model type: {self.model_type}")

def main(args=None):
    rclpy.init(args=args)
    model_type = 'bellhop'  # Default value, can be changed based on user input
    tl_data = 'some_data'   # Default value, can be changed based on processed HDF5 data
    node = PropagationModel(model_type, tl_data)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()