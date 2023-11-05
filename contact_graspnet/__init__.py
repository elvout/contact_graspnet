# This package contains a module of the same name ("contact_graspnet").
# Repeatedly adding folders to sys.path and importing that module using a
# relative import (as modules within this package do many times) does not work
# as the authors intended when  __main__ is run from a **different package**,
# since the module name points to the package instead. We can bring functions
# from the module namespace into the package namespace as a workaround.
from contact_graspnet.contact_graspnet import (  # noqa: F401
    placeholder_inputs,
    get_bin_vals,
    get_model,
    build_6d_grasp,
    get_losses,
    multi_bin_labels,
    compute_labels,
)
