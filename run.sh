# gradual mode
python inference.py \
    --video_path './test/videos/p7.mp4' \
    --stride 2 \
    --out_dir experiments \
    --radius_scale 1 \
    --camera 'target' \
    --mode 'gradual' \
    --mask \
    --target_pose 0 -30 0.3 0 0 \
    --traj_txt 'test/trajs/loop2.txt' \

# # direct mode
# python inference.py \
#     --video_path './test/videos/p7.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'direct' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \

# # bullet time
# python inference.py \
#     --video_path 'KK.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'bullet' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \

# # dolly-zoom mode
# python inference.py \
#     --video_path './test/videos/p7.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'zoom' \
#     --mask \
#     --target_pose 0 0 0.5 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' 