#pip3 install torch torchvision

#import torch
#if torch.backends.mps.is_available():
#    mps_device = torch.device("mps")
#    x = torch.ones(1, device=mps_device)
#    print (x)
#    # output expected:
#    # tensor([1.], device='mps:0')
#
#else:
#    print ("MPS device not found.")

import torch
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video

pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.enable_model_cpu_offload()

prompt = "Spiderman is surfing"
video_frames = pipe(prompt).frames
video_path = export_to_video(video_frames)
video_path

