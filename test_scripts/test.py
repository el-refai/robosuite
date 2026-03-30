import numpy as np
import robosuite as suite
import imageio
import robosuite.macros as macros

# Set the image convention to opencv so that the images are automatically rendered "right side up"
macros.IMAGE_CONVENTION = "opencv"

# create environment instance
env = suite.make(
    env_name="TwoArmTapeHandover",
    robots=["Panda", "Panda"],
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=512,
    camera_widths=512,
    control_freq=20,
)

# reset the environment
env.reset()

# create a video writer
video_path = "video.mp4"
video_writer = imageio.get_writer(video_path, fps=20)

print(f"Generating video for 100 steps...")

for i in range(100):
    # For two robots, the action space is the concatenation of both robots' action spaces
    action = np.random.uniform(env.action_spec[0], env.action_spec[1])
    
    obs, reward, done, info = env.step(action)  # take action in the environment
    
    # get the frame from the observation
    # Since we set IMAGE_CONVENTION = "opencv", the image is already right-side up
    frame = obs["agentview_image"]
    video_writer.append_data(frame)

video_writer.close()
print(f"Video saved to {video_path}")
