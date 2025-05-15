import rclpy
from rclpy.node import Node
from custom_interfaces.msg import State
from custom_interfaces.srv import BathyService
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import griddata

class Visual(Node):
    def __init__(self):
        super().__init__('visual_node')
        self.setup_logger()

        self.mammal_state = None
        self.ship_state = None
        self.ship_start_state = None
        self.ship_end_state = None
        self.bathy_points = None
        self.bathy_resoln = 200
        self.bathy_surface_computed = False
        self.bathy_max_lat = 49.98125076293945
        self.bathy_min_lat = 46.81041717529297
        self.bathy_max_lon = -121.56041717529297
        self.bathy_min_lon = -125.19791412353516


        self.bathy_client = self.create_client(BathyService, 'bathy_service')
        while not self.bathy_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('BathyService not available, waiting again...')
        self.bathy_points = self.get_bathy_service_response()

        self.mammal_subscription = self.create_subscription(
            State,
            'mammal_state',
            self.mammal_callback,
            10)

        self.ship_subscription = self.create_subscription(
            State,
            'ship_state',
            self.ship_callback,
            10)

        self.ship_start_state_subscription = self.create_subscription(
            State,
            'ship_start_state',
            self.ship_start_state_callback,
            10)

        self.ship_end_state_subscription = self.create_subscription(
            State,
            'ship_end_state',
            self.ship_end_state_callback,
            10)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.plot_surface(
            np.array([[-90, -90], [90, 90]]),
            np.array([[-180, 180], [-180, 180]]),
            np.array([[0, 0], [0, 0]]),
            alpha=0.5,
            color='gray'
        )

        

    def setup_logger(self):
        self.get_logger().info("Visual Node is started!")

    def mammal_callback(self, msg):
        self.mammal_state = (
            msg.position.latitude,
            msg.position.longitude,
            -msg.position.depth,
            msg.angle.heading,
            msg.speed
        )
        self.get_logger().info(f"Getting Mammal Location: {self.mammal_state}")
        self.plot_states()

    def ship_callback(self, msg):
        self.ship_state = (
            msg.position.latitude,
            msg.position.longitude,
            msg.position.depth,
            msg.angle.heading,
            msg.speed
        )
        self.get_logger().info(f"Getting Ship Location: {self.ship_state}")
        self.plot_states()
    
    def ship_start_state_callback(self, msg):
        self.ship_start_state = (
            msg.position.latitude,
            msg.position.longitude,
            msg.position.depth,
            msg.angle.heading,
            msg.speed
        )
        if self.ship_start_state is None:
            self.get_logger().info(f"Getting Ship Start Location: {self.ship_start_state}")
            self.plot_states()

    def ship_end_state_callback(self, msg):
        self.ship_end_state = (
            msg.position.latitude,
            msg.position.longitude,
            msg.position.depth,
            msg.angle.heading,
            msg.speed
        )
        if self.ship_end_state is None:
            self.get_logger().info(f"Getting Ship End Location: {self.ship_end_state}")
            self.plot_states()

    def get_bathy_service_response(self):
        request = BathyService.Request()
        # TODO Set the bathy_min and max in the req

        future = self.bathy_client.call_async(request)
        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                if future.result() is not None:
                    self.get_logger().info(f"Received Bathy Service Response. {len(future.result().bathy_points)} points.")
                    return future.result()
                else:
                    self.get_logger().error('BathyService call failed %r' % (future.exception(),))
                break

    def plot_states(self):
        # Create two subplots
        self.fig.clf()
        ax_surface = self.fig.add_subplot(121, projection='3d')  # 3D surface plot
        ax_contour = self.fig.add_subplot(122)  # Contour plot


        if not self.bathy_surface_computed:
            if self.bathy_points is not None:
                latitudes = [point.latitude for point in self.bathy_points.bathy_points]
                longitudes = [point.longitude for point in self.bathy_points.bathy_points]
                depths = [-point.depth for point in self.bathy_points.bathy_points]

            self.compute_bathy_surface(latitudes, longitudes, depths)
            self.bathy_surface_computed = True
            self.bathy_min_ele = min(depths)
            self.bathy_max_ele = max(depths)

        # Plot 3D surface on the first subplot
        ax_surface.plot_surface(self.bathy_surf_x, self.bathy_surf_y, self.bathy_surf_z, cmap='viridis', alpha=0.7)
        ax_surface.set_xlabel('Latitude')
        ax_surface.set_ylabel('Longitude')
        ax_surface.set_zlabel('Depth')
        ax_surface.set_xlim([self.bathy_min_lat, self.bathy_max_lat])
        ax_surface.set_ylim([self.bathy_min_lon, self.bathy_max_lon])
        ax_surface.set_zlim([self.bathy_min_ele, self.bathy_max_ele])

        # Plot filled contours on the second subplot with 100 levels using seismic colormap
        contour_levels = np.linspace(self.bathy_min_ele, self.bathy_max_ele, 100)
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=self.bathy_min_ele, vmax=self.bathy_max_ele)
        contour_filled = ax_contour.contourf(self.bathy_surf_x, self.bathy_surf_y, self.bathy_surf_z, levels=contour_levels, cmap='seismic', norm=norm)

        ax_contour.set_xlabel('Latitude')
        ax_contour.set_ylabel('Longitude')
        ax_contour.set_xlim([self.bathy_min_lat, self.bathy_max_lat])
        ax_contour.set_ylim([self.bathy_min_lon, self.bathy_max_lon])

        # Add a colorbar to the filled contour plot
        cbar = plt.colorbar(contour_filled, ax=ax_contour, orientation='horizontal', label='Depth')

        if self.mammal_state is not None:
            ax_surface.scatter(*self.mammal_state[:3], c='r', marker='o')
            u, v = self.get_unit_vector(*self.mammal_state[3:5])
            ax_surface.quiver(self.mammal_state[0], self.mammal_state[1], self.mammal_state[2], u, v, 0, color='r', pivot='tail', arrow_length_ratio=0.5, linewidth=2.0)

            ax_contour.quiver(self.mammal_state[0], self.mammal_state[1], u, v, color='k', linewidth=2.0)
            ax_contour.scatter(*self.mammal_state[:2], c='r', marker='o',s=100)

        if self.ship_state is not None:
            ax_surface.scatter(*self.ship_state[:3], c='b', marker='^')
            u, v = self.get_unit_vector(*self.ship_state[3:5])
            ax_surface.quiver(self.ship_state[0], self.ship_state[1], self.ship_state[2], u, v, 0, color='b', pivot='tail', arrow_length_ratio=0.5, linewidth=2.0)

            ax_contour.quiver(self.ship_state[0], self.ship_state[1], u, v, color='k', linewidth=2.0)
            ax_contour.scatter(*self.ship_state[:2], c='b', marker='^',s=100)

        if self.ship_start_state is not None:
            ax_contour.scatter(*self.ship_start_state[:2], c='g', marker='+', label='Port A',s=100)

        if self.ship_end_state is not None:
            ax_contour.scatter(*self.ship_end_state[:2], c='r', marker='x', label='Port B',s=100)

        ax_contour.legend()

        ax_contour.set_aspect('equal')

        self.fig.canvas.draw()
        plt.pause(0.001)

    def compute_bathy_surface(self, latitudes, longitudes, depths):
        self.ax.clear()

        xi = np.linspace(min(latitudes), max(latitudes), self.bathy_resoln)
        yi = np.linspace(min(longitudes), max(longitudes), self.bathy_resoln)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((latitudes, longitudes), depths, (xi, yi), method='linear')
        self.bathy_surf_x = xi
        self.bathy_surf_y = yi
        self.bathy_surf_z = zi



    def get_unit_vector(self, heading, speed):
        vec_length = 0.001 * speed
        angle_rad = np.radians(90 - heading)  # Convert heading to radians (clockwise from true north)

        u = vec_length * np.sin(angle_rad)
        v = vec_length * np.cos(angle_rad)

        return u, v

def main(args=None):
    rclpy.init(args=args)
    visual_obj = Visual()
    rclpy.spin(visual_obj)
    visual_obj.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
