# Placeholder module for editable install support.
# The actual implementation is provided by the compiled C++/CUDA extension
# (pbc_graph_outer_cuda.cpython-*.so) built via setup_outer.py.
#
# This file exists solely so that setuptools editable installs can correctly
# map the module name to the source directory. The real extension module
# will be imported when available.
