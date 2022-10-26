


# Import everything needed to edit video clips
from moviepy.editor import *


# loading video dsa gfg intro video
clip = VideoFileClip("test.mp4")


# getting only first 5 seconds
clip = clip.subclip(0, 5)

# looping video 3 times
loopedClip = clip.loop(3)


