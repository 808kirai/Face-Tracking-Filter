import pyvista as pv

# viewer window with a dark navy background
plotter = pv.Plotter(window_size=[1024, 768], title="3D Interactive Viewer")
plotter.set_background("#0f172a")

# the 3d object
mesh = pv.Icosahedron(radius=1.5)

# pbr shading gives the metal look, low roughness keeps it shiny
plotter.add_mesh(
    mesh,
    color="#3b82f6",
    metallic=0.85,
    roughness=0.15,
    pbr=True,
    smooth_shading=True,
)

# one bright key light from the upper front corner
plotter.add_light(pv.Light(position=(5, 5, 5), intensity=1.5, color="white"))

# opens the window and blocks until it is closed
plotter.show()
