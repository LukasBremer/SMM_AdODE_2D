# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "napari[all]",
#     "scikit-image",
#     "numpy",
# ]
# ///

import numpy as np
import napari

# Load the previously saved stack
stack = np.load('datenfuerhendrik.npy')

# Create a napari viewer and add the stack as an image layer
viewer = napari.Viewer()
image_layer = viewer.add_image(stack)

# Print shape of image data
print(f"Loaded stack shape: {stack.shape}")

# Start the event loop and show the viewer
napari.run()