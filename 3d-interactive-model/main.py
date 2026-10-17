import pyvista as pv

# Create the viewer.
plotter = pv.Plotter(window_size=[1024, 768], title="3D Interactive Viewer")
plotter.set_background("#0f172a")

# Build the 3D object.
mesh = pv.Icosahedron(radius=1.5)

# Add the mesh with metallic shading.
plotter.add_mesh(
    mesh,
    color="#3b82f6",
    metallic=0.85,
    roughness=0.15,
    pbr=True,
    smooth_shading=True,
)

# Add the key light.
plotter.add_light(pv.Light(position=(5, 5, 5), intensity=1.5, color="white"))

# Open the viewer.
plotter.show()
