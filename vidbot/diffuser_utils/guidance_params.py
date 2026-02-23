OPEN_PARAMS = {
    "goal_weight": 200.0,
    "goal_scale": 1.0,
    "smooth_weight": 15000.0,
    "noncollide_weight": 0.0,
    "normal_weight": 200.0,
    "normal_sign": 1,
    "contact_weight": 0.0,
    "set_goal_infinite": False,
    "fine_voxel_resolution": None,
}

CLOSE_PARAMS = {
    "goal_weight": 100.0,
    "goal_scale": 1.0,
    "smooth_weight": 15000.0,
    "noncollide_weight": 0.0,
    "normal_weight": 200.0,
    "normal_sign": -1,
    "contact_weight": 0.0,
    "set_goal_infinite": True,
    "fine_voxel_resolution": None,
}

TAKE_PARAMS = {
    "goal_weight": 100.0,
    "goal_scale": 1.0,
    "smooth_weight": 15000.0,
    "noncollide_weight": 200.0,
    "normal_weight": 10.0,
    "normal_sign": 1,
    "contact_weight": 0.0,
    "set_goal_infinite": False,
    "fine_voxel_resolution": 128,
}

PICKUP_PARAMS = {
    "goal_weight": 100.0,
    "goal_scale": 1.0,
    "smooth_weight": 15000.0,
    "noncollide_weight": 200.0,
    "normal_weight": 10.0,
    "normal_sign": 1,
    "contact_weight": 0.0,
    "set_goal_infinite": False,
    "fine_voxel_resolution": 128,
}

PLACE_PARAMS = {
    "goal_weight": 100.0,
    "goal_scale": 1.0,
    "smooth_weight": 15000.0,
    "noncollide_weight": 200.0,
    "normal_weight": 10.0,
    "normal_sign": -1,
    "contact_weight": 0.0,
    "set_goal_infinite": False,
    "fine_voxel_resolution": 128,
}

PRESS_PARAMS = {
    "goal_weight": 100.0,
    "goal_scale": 1.0,
    "smooth_weight": 15000.0,
    "noncollide_weight": 0.0,
    "normal_weight": 100.0,
    "normal_sign": -1,
    "contact_weight": 0.0,
    "set_goal_infinite": False,
    "fine_voxel_resolution": None,
}


GUIDANCE_PARAMS_DICT = {
    "open": OPEN_PARAMS,
    "close": CLOSE_PARAMS,
    "take": TAKE_PARAMS,
    "pickup": PICKUP_PARAMS,
    "place": PLACE_PARAMS,
    "press": PRESS_PARAMS,
}
