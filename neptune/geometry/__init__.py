from .function import Function
from .grid import Grid, DenseGrid, SparseGrid
from .mask import Mask, generate_single_mask, generate_masks, apply_mask
from .mesh_utils import Geometry_T, Dense_T
from .mesh_utils import initial_condition, left_boundary_condition, right_boundary_condition, boundary_condition, interior, to_dense
from .utils import Dimension
