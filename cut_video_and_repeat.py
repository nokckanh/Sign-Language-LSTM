
from ctypes import resize
from time import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("D:\Documents\CodePython\Graduation\videoHuyen.mp4", 20, 30, targetname="a.mp4")


