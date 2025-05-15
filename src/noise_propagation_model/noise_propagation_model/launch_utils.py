import os
import h5py
import rclpy
from custom_interfaces.srv import BoundsService

def get_bounds():
    rclpy.init()
    node = rclpy.create_node('bounds_service_client')
    client = node.create_client(BoundsService, 'bounds_service')
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('BoundsService not available, waiting again...')
    request = BoundsService.Request()
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        response = future.result()
        node.get_logger().info(
            f"Received bounds: max_lat={response.max_lat}, min_lat={response.min_lat}, "
            f"max_lon={response.max_lon}, min_lon={response.min_lon}"
        )
        rclpy.shutdown()
        return response
    else:
        node.get_logger().error('BoundsService call failed %r' % (future.exception(),))
        rclpy.shutdown()
        return None

def list_models():
    model_path = "/models"
    if not os.path.exists(model_path):
        print(f"Model directory {model_path} does not exist")
        return []
    
    models = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
    for i, model in enumerate(models):
        print(f"{i+1}: {model}")
    
    return models

def search_hdf5_file(model_type):
    model_path = f"/models/{model_type}"
    hdf5_files = []

    if os.path.exists(model_path):
        for file in os.listdir(model_path):
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(model_path, file))
    
    return hdf5_files

def process_hdf5_file(file_path):
    tl_data = []
    try:
        with h5py.File(file_path, 'r') as hdf5_file:
            for dataset_name in hdf5_file:
                dataset = hdf5_file[dataset_name]
                metadata = dict(dataset.attrs)
                tl_data.append({
                    "dataset_name": dataset_name,
                    "metadata": metadata,
                    "data": dataset[:]
                })
    except Exception as e:
        print(f"Error processing .hdf5 file: {e}")

    return tl_data
