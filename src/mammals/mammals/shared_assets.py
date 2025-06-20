from pathlib import Path
from environment.py_polyshp import Polyshp
from noise_propagation_model.models.RBF.utils_rbf import RBFInterpolatorCustom
TL_MODEL = None
SHORELINE_POLYGONS = None

def init_shared(tl_model_path: str, shoreline_path: str, bounds_dict: dict):
    """
    Load heavyweight resources *once* and cache them.
    """
    global TL_MODEL, SHORELINE_POLYGONS
    
    lat_max = bounds_dict['lat_max']
    lat_min = bounds_dict['lat_min']
    lon_max = bounds_dict['lon_max']
    lon_min = bounds_dict['lon_min']
    
    if TL_MODEL is None:
        TL_MODEL = RBFInterpolatorCustom.import_model(tl_model_path)
    if SHORELINE_POLYGONS is None:
        SHORELINE_POLYGONS = Polyshp(shoreline_path).get_polygons(lat_min, lat_max, lon_min, lon_max)
