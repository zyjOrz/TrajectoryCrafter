from huggingface_hub import snapshot_download
import os


def download_model():
    snapshot_download(
        repo_id="TrajectoryCrafter/TrajectoryCrafter",
        local_dir="/media/msc-auto/HDD/chensheng/GenRec4D/TrajectoryCrafter/checkpoints/TrajectoryCrafter",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="tencent/DepthCrafter",
        local_dir="/media/msc-auto/HDD/chensheng/GenRec4D/TrajectoryCrafter/checkpoints/DepthCrafter",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="stabilityai/stable-video-diffusion-img2vid",
        local_dir="/media/msc-auto/HDD/chensheng/GenRec4D/TrajectoryCrafter/checkpoints/stable-video-diffusion-img2vid",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="alibaba-pai/CogVideoX-Fun-V1.1-5b-InP",
        local_dir="/media/msc-auto/HDD/chensheng/GenRec4D/TrajectoryCrafter/checkpoints/CogVideoX-Fun-V1.1-5b-InP",
        local_dir_use_symlinks=False,
    )
    snapshot_download(
        repo_id="Salesforce/blip2-opt-2.7b",
        local_dir="/media/msc-auto/HDD/chensheng/GenRec4D/TrajectoryCrafter/checkpoints/blip2-opt-2.7b",
        local_dir_use_symlinks=False,
    )


download_model()
