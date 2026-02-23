import numpy as np
import open3d as o3d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def pick_points_in_viewer(points, scene_colors=None, verbose=False):
    def pick_points(pcd):
        print("")
        print("1) Please pick at least three correspondences using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if scene_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    else:
        pcd = points

    picked_ids = pick_points(pcd)
    final_points = np.asarray(pcd.points)[picked_ids]

    if verbose:
        print("Final points: ")
        for i in range(len(final_points)):
            print(final_points[i])

    return final_points


def smooth_rotation_matrices(rotation_matrices, smooth_factor=0.1):
    """
    Smooth a sequence of rotation matrices using SLERP.

    Args:
        rotation_matrices: A numpy array of shape (N, 3, 3), where N is the number of matrices.
        smooth_factor: A float between 0 and 1 indicating how much to smooth. Lower values mean more smoothing.

    Returns:
        A numpy array of the smoothed rotation matrices.
    """
    # Convert rotation matrices to quaternions
    # quaternions = rotation_matrices_to_quaternions(rotation_matrices)
    quaternions = R.from_matrix(rotation_matrices).as_quat()
    # Interpolate quaternions with SLERP
    smoothed_quaternions = []
    for i in range(1, len(quaternions)):
        # r1 = R.from_quat(quaternions[i - 1])
        # r2 = R.from_quat(quaternions[i])
        # import pdb; pdb.set_trace()
        slerp = Slerp([0, 1], R.from_quat([quaternions[i - 1], quaternions[i]]))
        t = np.linspace(0, 1, num=int(1 / smooth_factor))
        interp_quats = slerp(t).as_quat()
        interp_quats_mid = interp_quats[len(interp_quats) // 2]
        smoothed_quaternions.append(interp_quats_mid)
    # import pdb; pdb.set_trace()
    # Flatten the list of interpolated quaternions
    smoothed_quaternions = np.stack(smoothed_quaternions, axis=0)

    # Convert smoothed quaternions back to rotation matrices
    # smoothed_rotation_matrices = quaternions_to_rotation_matrices(smoothed_quaternions)
    smoothed_rotation_matrices = R.from_quat(smoothed_quaternions).as_matrix()
    smoothed_rotation_matrices = np.concatenate(
        [rotation_matrices[:1], smoothed_rotation_matrices], axis=0
    )
    return smoothed_rotation_matrices


def simplify_trajectory(traj, end_id=5, start_id=0):
    traj_pos = traj[:, :3, 3]  # [H, 3]
    traj_length = np.linalg.norm(traj_pos[-1] - traj_pos[0], axis=0)
    traj_dir = traj_pos[end_id] - traj_pos[start_id]  # [3]
    traj_dir = traj_dir / np.linalg.norm(traj_dir)  # [3]
    traj_steps = np.linspace(0, traj_length, len(traj))
    traj_pos = traj_pos[start_id] + traj_steps[:, np.newaxis] * traj_dir  # [H, 3]
    traj_simp = traj.copy()
    traj_simp[:, :3, 3] = traj_pos
    return traj_simp


def smooth_trajectory(traj):
    # traj: [H, 4, 4]
    traj_pos = traj[:, :3, 3]  # [30, 3]
    traj_pos = savgol_filter(traj_pos, len(traj), len(traj) // 2, axis=0)  # [30, 3]
    traj[:, :3, 3] = traj_pos  # [30, 4, 4]

    traj_pos = traj[:, :3, 3]  # [H, 3]
    traj_dir = traj_pos[0:-1] - traj_pos[1:]  # [H-1, 3]
    traj_dir = traj_dir / np.linalg.norm(traj_dir, axis=1, keepdims=True)  # [H-1, 3]

    # Adjust the trajectory rotation
    traj_adjust = []
    for s in range(1, len(traj)):
        traj_rot = traj[s, :3, :3]
        # traj_rot_x = traj_dir[s - 1]
        # traj_rot_y = traj_rot[:, 1]
        # traj_rot_z = np.cross(traj_rot_x, traj_rot_y)
        # traj_rot_adjust = np.stack([traj_rot_x, traj_rot_y, traj_rot_z], axis=1)
        traj_rot_adjust = traj_rot
        traj_pose_adjust = traj[s].copy()
        traj_pose_adjust[:3, :3] = traj_rot_adjust
        traj_adjust.append(traj_pose_adjust)
    traj_adjust = np.stack(traj_adjust, axis=0)  # [H, 4, 4]

    # Smooth the rotation
    traj_adjust_rot = smooth_rotation_matrices(traj_adjust[:, :3, :3])
    traj_adjust[:, :3, :3] = traj_adjust_rot
    return traj_adjust


def smooth_trajectory_with_neighbor_direction(traj):
    traj_adjust = [traj[0]]
    traj_pos = traj[:, :3, 3]  # [30, 3]
    traj_dir = traj_pos[0:-1] - traj_pos[1:]  # [H-1, 3]
    traj_dir = traj_dir / np.linalg.norm(traj_dir, axis=1, keepdims=True)  # [H-1, 3]

    for s in range(1, len(traj)):
        traj_rot = traj[s, :3, :3]
        traj_rot_x = traj_dir[s - 1]
        traj_rot_z = traj_rot[:, 2]
        traj_rot_y = np.cross(traj_rot_z, traj_rot_x)
        traj_rot_adjust = np.stack([traj_rot_x, traj_rot_y, traj_rot_z], axis=1)
        traj_pose_adjust = traj[s].copy()
        traj_pose_adjust[:3, :3] = traj_rot_adjust
        traj_adjust.append(traj_pose_adjust)
    traj_adjust = np.stack(traj_adjust, axis=0)  # [H, 4, 4]

    # Smooth the rotation
    traj_adjust_rot = smooth_rotation_matrices(traj_adjust[:, :3, :3])
    traj_adjust[:, :3, :3] = traj_adjust_rot
    assert traj_adjust.shape == traj.shape
    return traj_adjust


def align_trajectory(traj, T_offset):
    # traj: [H, 4, 4]
    for i in range(len(traj)):
        traj[i] = traj[i] @ T_offset
    return traj


def scale_trajectory_length_manip(traj, scale, reciprocal=False):
    # traj: [H, 4, 4]
    if reciprocal:
        scale = 1 / scale

    if isinstance(traj, list):
        traj = np.array(traj)
    traj_pos = traj[:, :3, 3]  # [H, 3]
    traj_pos_init = traj_pos[0:1]  # [1, 3]
    traj_dist = traj_pos - traj_pos_init  # [H, 3]
    traj_pos = traj_pos_init + traj_dist * scale  # [H, 3]
    traj[:, :3, 3] = traj_pos
    return traj
