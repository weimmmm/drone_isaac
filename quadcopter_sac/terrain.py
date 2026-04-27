from isaaclab.terrains.height_field.utils import height_field_to_mesh
import numpy as np
import time


def sample_uniform_obstacle_layout(
    *,
    seed: int,
    num_obstacles: int,
    map_size: tuple[float, float],
    obstacle_width_range: tuple[float, float],
    obstacle_height_range: tuple[float, float],
    obstacles_distance: float,
    avoid_positions: list[list[float]] | None = None,
):
    rng = np.random.default_rng(seed)

    def is_good_position(obs_list, obs_pos, min_dist):
        for obs_pos_i in obs_list:
            dist = ((obs_pos[0] - obs_pos_i[0]) ** 2 + (obs_pos[1] - obs_pos_i[1]) ** 2) ** 0.5
            if dist < min_dist:
                return False
        return True

    half_x = map_size[0] * 0.5
    half_y = map_size[1] * 0.5
    existing_positions = [] if avoid_positions is None else [list(pos[:2]) for pos in avoid_positions]
    layout = []

    stop_sampling = False
    for _ in range(num_obstacles):
        width = float(rng.uniform(*obstacle_width_range))
        length = float(rng.uniform(*obstacle_width_range))
        height = float(rng.uniform(*obstacle_height_range))

        start_time = time.time()
        good_position = False
        x_pos = 0.0
        y_pos = 0.0
        while not good_position:
            x_pos = float(rng.uniform(-half_x, half_x))
            y_pos = float(rng.uniform(-half_y, half_y))
            good_position = is_good_position(existing_positions, [x_pos, y_pos], obstacles_distance)
            if (time.time() - start_time) > 0.2:
                stop_sampling = True
                break
        if stop_sampling:
            break

        existing_positions.append([x_pos, y_pos])
        layout.append(
            {
                "x": x_pos,
                "y": y_pos,
                "width": width,
                "length": length,
                "height": height,
            }
        )

    return layout


@height_field_to_mesh
def uniform_discrete_obstacles_terrain(difficulty: float, cfg) -> np.ndarray:
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))

    layout = cfg.obstacle_layout
    if layout is None:
        layout = sample_uniform_obstacle_layout(
            seed=int(cfg.seed),
            num_obstacles=cfg.num_obstacles,
            map_size=cfg.size,
            obstacle_width_range=cfg.obstacle_width_range,
            obstacle_height_range=cfg.obstacle_height_range,
            obstacles_distance=cfg.obstacles_distance,
            avoid_positions=cfg.avoid_positions,
        )

    for obstacle in layout:
        width = int(obstacle["width"] / cfg.horizontal_scale)
        length = int(obstacle["length"] / cfg.horizontal_scale)
        height = int(obstacle["height"] / cfg.vertical_scale)
        x_center = obstacle["x"] + 0.5 * cfg.size[0]
        y_center = obstacle["y"] + 0.5 * cfg.size[1]
        x_start = int((x_center - 0.5 * obstacle["width"]) / cfg.horizontal_scale)
        y_start = int((y_center - 0.5 * obstacle["length"]) / cfg.horizontal_scale)

        x_start = max(0, min(x_start, width_pixels - width))
        y_start = max(0, min(y_start, length_pixels - length))
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height


    # clip the terrain to the platform
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)
