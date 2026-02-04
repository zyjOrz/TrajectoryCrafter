import open3d as o3d

pcd = o3d.io.read_point_cloud("/home/msc-auto/pcs/GenRen4D/TrajectoryCrafter/experiments/20251231_0955_test2/ply_cam2/frame_020.ply")
o3d.visualization.draw_geometries([pcd])

import open3d as o3d
import numpy as np

ply_path = "/home/msc-auto/pcs/GenRen4D/TrajectoryCrafter/experiments/20251231_0955_test2/ply_cam2/frame_020.ply"
pcd = o3d.io.read_point_cloud(ply_path)

if not pcd.has_colors():
    pcd.paint_uniform_color([0.8, 0.8, 0.8])

w, h = 1280, 720
render = o3d.visualization.rendering.OffscreenRenderer(w, h)
mat = o3d.visualization.rendering.MaterialRecord()
mat.point_size = 2.0
render.scene.add_geometry("pcd", pcd, mat)

render.scene.set_background([1, 1, 1, 1])

img = render.render_to_image()
o3d.io.write_image("pcd_view.png", img)
print("Saved to pcd_view.png")
