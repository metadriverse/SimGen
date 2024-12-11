import time

import cv2
import gymnasium as gym
import mediapy
import numpy as np
import tqdm
from PIL import Image
from PIL import ImageDraw, ImageFont
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

from simgen import SimGenPipeline


def postprocess_semantic_image(image):
    """
    In order to align with the Segformer's output, we modify the output color of the semantic image from MetaDrive.
    """
    # TODO(pzh): This function can't properly remap the color of the lane line and crosswalk because
    #  there are some color drift (caused by resized?) in the original image. We need to fix this.

    # customized
    old_LANE_LINE = (255, 255, 255)
    old_CROSSWALK = (55, 176, 189)

    # These color might be prettier?
    new_LANE_LINE = (128, 64, 128)
    new_CROSSWALK = (128, 64, 128)

    # Change the color of the lane line and crosswalk
    assert image.dtype == np.uint8

    is_lane_line = (
            (image[..., 0] == old_LANE_LINE[0]) &
            (image[..., 1] == old_LANE_LINE[1]) &
            (image[..., 2] == old_LANE_LINE[2])
    )
    image[is_lane_line] = new_LANE_LINE

    is_crosswalk = (
            (image[..., 0] == old_CROSSWALK[0]) &
            (image[..., 1] == old_CROSSWALK[1]) &
            (image[..., 2] == old_CROSSWALK[2])
    )
    image[is_crosswalk] = new_CROSSWALK

    return image


def add_text(image, text_prompt):
    # Convert the image to RGBA mode
    image = Image.fromarray(image, mode="RGB").convert("RGBA")

    # Create a transparent overlay
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Define the text and font
    font_path = "Arial.ttf"  # Replace with your font path
    font_size = 70
    font = ImageFont.truetype(font_path, font_size)

    # Get text size
    text_width = draw.textlength(text_prompt, font=font)

    # Make the text align in left bottom corner
    position = (
        50,
        image.height - font_size - 50
    )

    # Draw the semi-transparent rectangle
    off = 15
    rect_start = (position[0] - off, position[1] - off)
    rect_end = (position[0] + text_width + off, position[1] + font_size + off)
    draw.rectangle([rect_start, rect_end], fill=(255, 255, 255, int(255 * 0.5)))  # Alpha = 0.5

    # Draw the text
    draw.text(position, text_prompt, font=font, fill="black")

    # Merge the overlay with the original image
    image = Image.alpha_composite(image, overlay)

    # Convert back to RGB mode if needed
    image = image.convert("RGB")

    # Get back the image
    image = np.array(image)
    return image


class SimGenObservation(BaseObservation):
    def __init__(self, config):
        super(SimGenObservation, self).__init__(config)
        assert config["norm_pixel"] is False
        assert config["stack_size"] == 1
        self.seg_obs = ImageObservation(config, "seg_camera", config["norm_pixel"])
        self.rgb_obs = ImageObservation(config, "rgb_camera", config["norm_pixel"])
        self.depth_obs = ImageObservation(config, "depth_camera", config["norm_pixel"])

    @property
    def observation_space(self):
        os = dict(
            rgb=self.rgb_obs.observation_space,
            seg=self.seg_obs.observation_space,
            depth=self.depth_obs.observation_space,
        )
        return gym.spaces.Dict(os)

    def observe(self, vehicle):
        ret = {}

        seg_cam = self.engine.get_sensor("seg_camera").cam
        agent = seg_cam.getParent()
        original_position = seg_cam.getPos()
        heading, pitch, roll = seg_cam.getHpr()
        seg_img = self.seg_obs.observe(agent, position=original_position, hpr=[heading, pitch, roll])
        assert seg_img.ndim == 4
        assert seg_img.shape[-1] == 1
        assert seg_img.dtype == np.uint8
        # Do some postprocessing here
        seg_img = seg_img[..., 0]
        before = seg_img.copy()
        seg_img = postprocess_semantic_image(seg_img)
        seg_img = seg_img[..., ::-1]  # BGR -> RGB
        ret["seg"] = seg_img

        depth_cam = self.engine.get_sensor("depth_camera").cam
        agent = depth_cam.getParent()
        original_position = depth_cam.getPos()
        heading, pitch, roll = depth_cam.getHpr()
        depth_img = self.depth_obs.observe(agent, position=original_position, hpr=[heading, pitch, roll])
        assert depth_img.ndim == 4
        assert depth_img.shape[-1] == 1
        assert depth_img.dtype == np.uint8
        depth_img = depth_img[..., 0]
        # before = depth_img.copy()
        depth_img = cv2.bitwise_not(depth_img)
        depth_img = depth_img[..., None]
        ret["depth"] = depth_img

        rgb_cam = self.engine.get_sensor("rgb_camera").cam
        agent = rgb_cam.getParent()
        original_position = rgb_cam.getPos()
        heading, pitch, roll = rgb_cam.getHpr()
        rgb_img = self.rgb_obs.observe(agent, position=original_position, hpr=[heading, pitch, roll])
        assert rgb_img.ndim == 4
        assert rgb_img.shape[-1] == 1
        assert rgb_img.dtype == np.uint8
        rgb_img = rgb_img[..., 0]
        # Change the color from BGR to RGB
        rgb_img = rgb_img[..., ::-1]
        ret["rgb"] = rgb_img

        return ret


if __name__ == "__main__":

    # ===== SimGen Setup =====
    pipeline = SimGenPipeline()
    ddim_steps = 100

    # ===== MetaDrive Setup =====
    skip_steps = 7
    # We want each frame to stay about 0.7s. To do so, we can repeat every frame for 7 times and set FPS to 10.
    fps = 10
    env = ScenarioEnv(
        {
            'agent_observation': SimGenObservation,

            # To enable onscreen rendering, set this config to True.
            "use_render": False,

            # !!!!! To enable offscreen rendering, set this config to True !!!!!
            "image_observation": True,

            "norm_pixel": False,
            "stack_size": 1,

            # ===== The scenario and MetaDrive config =====
            "agent_policy": ReplayEgoCarPolicy,
            "no_traffic": False,
            "sequential_seed": True,
            "reactive_traffic": False,
            "num_scenarios": 9,
            "horizon": 1000,
            "no_static_vehicles": False,
            "agent_configs": {
                "default_agent": dict(use_special_color=True, vehicle_model="varying_dynamics_bounding_box")
            },
            "vehicle_config": dict(
                show_navi_mark=False,
                show_line_to_dest=False,
                lidar=dict(num_lasers=120, distance=50),
                lane_line_detector=dict(num_lasers=0, distance=50),
                side_detector=dict(num_lasers=12, distance=50),
            ),
            # "use_bounding_box": True,
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
            "height_scale": 1,

            "set_static": True,

            # ===== Set some sensor and visualization configs =====
            "daytime": "08:10",
            "window_size": (800, 450),
            "camera_dist": 0.8,  # 0.8, 1.71
            "camera_height": 1.5,  # 1.5
            "camera_pitch": None,
            "camera_fov": 66,  # 60, 66
            "sensors": dict(
                depth_camera=(DepthCamera, 800, 450),
                rgb_camera=(RGBCamera, 800, 450),
                seg_camera=(SemanticCamera, 800, 450),
            ),

            # ===== Remove useless items in the images =====
            "show_logo": False,
            "show_fps": False,
            "show_interface": True,
            "disable_collision": True,
            "force_destroy": True,
        }
    )

    # ===== Run the simulation =====
    region_candidates = [
        "Los Angeles, United States",
        "Beijing, China",
        "Pretoria, South Africa",
        "London, England",
        "Riyadh, Saudi Arabia",
        "Moscow, Russia",
        "Zurich, Switzerland",
        "Kyoto, Japan",
        "Vancouver, Canada",
        "Seoul, Korea",
        "Delhi, India",
    ]

    prefix_candidates = [
        # "",
        " in a lego style",
        " in a ukiyo-e style",
        " in a minecraft style",
        # " in a supermario style",
    ]

    for ep in range(9):
        frames = []
        env.reset()
        seed = np.random.randint(0, 100000)
        np_random = np.random.RandomState(seed)
        scenario = env.engine.data_manager.current_scenario
        scenario_id = scenario['id']
        print(
            "Current scenario ID {}, dataset version {}, len: {}".format(
                scenario_id, scenario['version'], scenario['length']
            )
        )
        horizon = scenario['length']
        for t in tqdm.trange(horizon):
            o, r, d, _, _ = env.step([1, 0.88])
            if t % skip_steps == 0:
                depth_img = Image.fromarray(o["depth"].repeat(3, axis=-1), mode="RGB")
                seg_img = Image.fromarray(o["seg"], mode="RGB")
                rgb_img = Image.fromarray(o["rgb"], mode="RGB")

                # Random select a city
                # Use np_random to avoid seed_everything in pipeline breaks the randomness of numpy.
                sampled_region_name = np_random.choice(region_candidates)
                # With 50% prob set prefix to ""
                sampled_prefix = np_random.choice(prefix_candidates)
                sampled_prefix = sampled_prefix if np_random.rand() < 0.5 else ""
                text_prompt = "An image of a city street in {}{}.".format(sampled_region_name, sampled_prefix)
                print("Text prompt: ", text_prompt)

                # Run SimGen
                output = pipeline(
                    depth_image=depth_img,
                    seg_image=seg_img,
                    content_image=rgb_img,
                    prompt=text_prompt,
                    seed=seed,
                    num_inference_steps=ddim_steps,
                )
                images = output.images
                image = images[0]
                vis = cv2.hconcat([o["seg"], o["depth"].repeat(3, axis=-1), o["rgb"]])
                h, w = image.shape[:2]
                vis_w = vis.shape[1]
                image = cv2.resize(image, (vis_w, int(h * vis_w / w)))
                image = add_text(image, text_prompt)
                vis = cv2.vconcat([vis, image])

                # Quick visualization:
                # import matplotlib.pyplot as plt;plt.imshow(vis);plt.show()

                for _ in range(skip_steps):
                    frames.append(vis)

        # Save mp4 video
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        mediapy.write_video(
            'video_{}_seed-{}_{}.mp4'.format(scenario_id, seed, time_str), frames,
            fps=fps
        )
