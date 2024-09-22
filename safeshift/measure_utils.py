import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d
from matplotlib import pyplot as plt

def save_delaunay(tri, path='tmp.png'):
    delaunay_plot_2d(tri)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    old_xlim = plt.xlim()
    d_xlim = max(np.abs(old_xlim[0]), np.abs(old_xlim[1]))
    plt.xlim([-d_xlim, d_xlim])
    old_ylim = plt.ylim()
    d_ylim = max(np.abs(old_ylim[0]), np.abs(old_ylim[1]))
    plt.ylim([-d_ylim, d_ylim])
    plt.savefig(path)

# From MotionCNN
def rot_matrix(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])

# From MotionCNN
def shift_rotate(x, shift, angle):
    return (x + shift) @ rot_matrix(angle).T

def rotate_shift(x, shift, angle):
    return (x) @ rot_matrix(angle).T + shift

def possible_leading(ego_track, other_tracks, heading_threshold=45, voronoi=False):
    # Now, time for Delauney triangularization
    possible_ego_lf = []
    for i in range(len(ego_track)):
        ego_info = ego_track[i]
        valid_ego = ego_info[-1]
        # In case invoked for a different "ego"
        if not valid_ego:
            possible_ego_lf.append(set())
            continue

        other_info = other_tracks[:, i]
        valid_other_info = other_info[other_info[..., -1] == 1]
        all_infos = np.concatenate([ego_info[np.newaxis, :], valid_other_info])

        # First, filter to ensure they're largely heading the same direction of travel.
        ego_heading_deg = np.rad2deg(ego_info[4])
        all_heading_deg = np.rad2deg(all_infos[:, 4])
        # Should be symmetric; just the difference between them
        # Cases: 
        # 1. Following: other = 100, ego = 110 -> 10
        # 2. Oncoming: other = 90, ego = 270 -> 180
        # 3. T bone: other = 0, ego = 70 -> 70
        # This ignores differences between lanes, e.g. in a passing scenario
        heading_diff = np.abs((ego_heading_deg - all_heading_deg + 540) % 360 - 180)
        all_infos = all_infos[heading_diff < heading_threshold]

        # Second, filter by Delaunay triangles, for multiple following scenarios
        all_pos = all_infos[:, :2] - ego_info[:2]
        if len(all_pos) >= 4 and voronoi:
            tri = Delaunay(all_pos)
            # check tri simplices for edges
            # only care about the ones to ego, which is index 0 of points
            edges = set()
            for triangle in tri.simplices:
                # Check if index 0 (i.e., ego) is in this
                if triangle[0]  == 0 or triangle[1]  == 0:
                    edges.add((triangle[0], triangle[1]))
                    edges.add((triangle[1], triangle[0]))
                if triangle[0]  == 0 or triangle[2]  == 0:
                    edges.add((triangle[0], triangle[2]))
                    edges.add((triangle[2], triangle[0]))
                if triangle[1]  == 0 or triangle[2]  == 0:
                    edges.add((triangle[1], triangle[2]))
                    edges.add((triangle[2], triangle[1]))
            keep_idx = set()
            [keep_idx.update(x) for x in edges]
            keep_idx = sorted(list(keep_idx))
            keep_ids = set(list(all_infos[keep_idx, -2]))
        else:
            keep_ids = set(list(all_infos[:, -2]))
        possible_ego_lf.append(keep_ids)
    return possible_ego_lf
    
# Returns true/false if ego is following other; non-symmetric relationship
def ego_follows(ego_track, other_track, possible_ego_lf, s_o=0.15, R_th=-2):
    ego_x, ego_y, ego_vx, ego_vy, ego_heading, ego_length, \
        ego_width, ego_crash, ego_id, ego_valid = ego_track.transpose()
    other_x, other_y, other_vx, other_vy, other_heading, other_length, \
        other_width, other_crash, other_id, other_valid = other_track.transpose()

    # Going to do the math from:  Nagahama et al., 2021: 
    # "Detection of leader–follower combinations frequently observed in mixed traffic with weak lane-discipline"

    # Need the lateral and longitudinal distances (which are non symmetric)
    rel_x, rel_y = other_x - ego_x, other_y - ego_y
    # Want to rotate these points by -center_heading
    ego_positions = np.stack([ego_x, ego_y], axis=-1)
    other_positions = np.stack([other_x, other_y], axis=-1)
    rel_pos = np.array([shift_rotate(other_pos, -ego_pos, -h) for ego_pos, other_pos, h \
            in zip(ego_positions, other_positions, ego_heading)])
    long_dist, lat_dist = rel_pos.transpose()

    # This filter accounts for whether they're roughly in the same direction + Delaunay edge okay
    e_ij_dd = np.array([other_id[0] in x for x in possible_ego_lf])
    # Now, accounting for longitudinal ahead vs. behind
    e_ij_forward = long_dist > 0

    # s_ij is the lateral separation from sides of the vehicle
    s_ij = np.abs(lat_dist) - (ego_width + other_width) / 2
    # Effectively, cut-off separation is (s_o * abs(R_th)) in meters
    a_ij = np.minimum(1, np.exp(-s_ij/s_o))
    e_ij_lat = a_ij > np.exp(R_th)

    # The final leader-follower relation true/false
    e_ij = e_ij_dd * e_ij_forward * e_ij_lat

    return e_ij

# Get interaction vals between ego and other
def interaction_measures(ego_track, other_track, possible_ego_lf, possible_other_lf, hz=10, eps=0.05, mttcp_threshold=1):
    

    times = np.arange(0, len(ego_track)/hz, 1/hz)

    joint_mask = (other_track[:, -1] * ego_track[:, -1]).astype(bool)
    other_track = other_track[joint_mask]
    ego_track = ego_track[joint_mask]
    times = times[joint_mask]

    # Represents which one (if any) is leading
    other_leading = ego_follows(ego_track, other_track, np.array(possible_ego_lf)[joint_mask])
    ego_leading = ego_follows(other_track, ego_track, np.array(possible_other_lf)[joint_mask])

    ego_x, ego_y, ego_vx, ego_vy, ego_heading, ego_length, \
        ego_width, ego_crash, ego_id, ego_valid = ego_track.transpose()
    other_x, other_y, other_vx, other_vy, other_heading, other_length, \
        other_width, other_crash, other_id, other_valid = other_track.transpose()

    ego_pos = np.stack([ego_x, ego_y], axis=-1)
    other_pos = np.stack([other_x, other_y], axis=-1)
    ego_vel = np.stack([ego_vx, ego_vy], axis=-1)
    other_vel = np.stack([other_vx, other_vy], axis=-1)
    speed_ego = np.linalg.norm(ego_vel, axis=-1) + eps
    speed_other = np.linalg.norm(other_vel, axis=-1) + eps

    # Conservative estimation of distance between vehicle centers is min(LENGTH, WIDTH)
    # If perfectly longitudinal collision, then length, if perfectly lateral then width; thus use width
    # Basically, models vehicles as disks with radius of width/2
    ego_offset = ego_width / 2
    other_offset = other_width / 2

    #ego_to_other = other_pos - ego_pos
    #ego_to_other /= np.linalg.norm(ego_to_other, axis=-1)[:, np.newaxis]
    #offset_dists = np.linalg.norm((ego_pos + ego_offset * ego_to_other) - (other_pos - other_offset * ego_to_other), axis=-1)
    #dists = offset_dists
    dists = np.linalg.norm(ego_pos - other_pos, axis=-1)
    offset_dists = np.maximum(eps, dists - (ego_offset + other_offset))

    ego_faster = speed_ego > speed_other
    other_faster = speed_other > speed_ego

    ego_leading_but_other_faster = ego_leading & other_faster
    other_leading_but_ego_faster = other_leading & ego_faster

    # TTC first
    ttc_into_ego = (offset_dists / (speed_other - speed_ego))[ego_leading_but_other_faster]
    ttc_into_other = (offset_dists / (speed_ego - speed_other))[other_leading_but_ego_faster]

    # THW next
    # If heading were ignored and speed maintained, time to other agent
    t_ego = offset_dists / speed_ego
    t_other = offset_dists / speed_other
    thw_into_ego = t_other[ego_leading]
    thw_into_other = t_ego[other_leading]

    # DRAC time
    drac_into_ego = (((speed_other - speed_ego) ** 2) / (2 * offset_dists))[ego_leading_but_other_faster]
    drac_into_other = (((speed_ego - speed_other) ** 2) / (2 * offset_dists))[other_leading_but_ego_faster]

    # From https://arxiv.org/pdf/2202.07438.pdf "Automated Analysis Framework"
    # Going to do a quick mTTCP, use the "actual path given"
    traj_dists = np.linalg.norm(ego_pos[:, None, :] - other_pos, axis=-1)
    # Index pairs where trajectories overlap
    ego_cp_idxs, other_cp_idxs = np.where(traj_dists <= mttcp_threshold)
    relative_mttcps = []
    for ego_cp_idx, other_cp_idx in zip(ego_cp_idxs, other_cp_idxs):
        cp = (ego_pos[ego_cp_idx] + other_pos[other_cp_idx]) / 2
        # assert np.linalg.norm(cp - other_pos[other_cp_idx]) <= mttcp_dist_threshold, 'The point of the threhold'
        # "the time a road user takes to travel to a conflict point by assuming a constant speed and using the actual path driven"

        # Also taking from here: 
        # "A microscopic simulation model for pedestrian-pedestrian and pedestrian-vehicle interactions at crosswalks", Liu et al. 2017
        # Equations 10-12

        # Dot products and projections
        ego_cp_vec = cp - ego_pos
        # TODO: fix divide by zero here.
        ego_proj_cp = (np.einsum('ij,ij->i', ego_vel, ego_cp_vec) / np.einsum('ij,ij->i', ego_cp_vec, ego_cp_vec))[:, None] * ego_cp_vec
        ego_proj_mag = np.linalg.norm(ego_proj_cp, axis=-1) + eps


        other_cp_vec = cp - other_pos
        other_proj_cp = (np.einsum('ij,ij->i', other_vel, other_cp_vec) / np.einsum('ij,ij->i', other_cp_vec, other_cp_vec))[:, None] * other_cp_vec
        other_proj_mag = np.linalg.norm(other_proj_cp, axis=-1) + eps

        downstream_i = np.einsum('ij,ij->i', ego_vel, cp - ego_pos) > 0
        downstream_j = np.einsum('ij,ij->i', other_vel, cp - other_pos) > 0

        joint_downstream_mask = downstream_i & downstream_j
        mttcp_i = np.linalg.norm(cp - ego_pos, axis=-1) / ego_proj_mag
        mttcp_j = np.linalg.norm(cp - other_pos, axis=-1) / other_proj_mag
        relative_mttcp = np.abs(mttcp_i - mttcp_j)[joint_downstream_mask]
        if len(relative_mttcp):
            relative_mttcps.append(relative_mttcp.min())

    # Reference from SafeShift:
    # def simple_interaction_score(feat, k=1):
    #     return min(2, feat['thw_val']) + \
    #         min(4, feat['scene_mttcp_val']) + \
    #         min(4, feat['agent_mttcp_val']) + \
    #         min(2*k, feat['ttc_val']) + \
    #         min(2*k, k/5 * feat['drac_val']) + \
    #         k*feat['collisions_val'] + \
    #         min(5, 0.1*feat['traj_pair_anomaly_val'])
    def safe_max(arr):
        return np.max(arr) if len(arr) else 0
    def safe_min(arr):
        return 1/np.min(arr) if len(arr) else 0
    interaction_score = \
        0.5 * min(2, safe_min(thw_into_ego)) + \
        0.5 * min(2, safe_min(thw_into_other)) + \
        0.5 * min(2, safe_min(ttc_into_ego)) + \
        0.5 * min(2, safe_min(ttc_into_other)) + \
        0.5 * min(2, 1/5 * safe_max(drac_into_ego)) + \
        0.5 * min(2, 1/5 * safe_max(drac_into_other)) + \
        1 * min(4, safe_min(relative_mttcps))
    
    return ttc_into_ego, thw_into_ego, drac_into_ego, ttc_into_other, thw_into_other, drac_into_other, relative_mttcps, [interaction_score]


# Get processed tracks from the cat_advgen save format (i.e., output/*/obs/*.npy)
def get_tracks(data, ego_key = 'default_agent'):
    tracks = data['tracks']

    ego_track = np.array([x[ego_key] for x in tracks])
    N = len(ego_track)
    # Add on a new ID (0) and valid bit (1) at end
    ego_track = np.concatenate([ego_track, np.zeros((N, 1)), np.ones((N, 1))], axis=-1)

    all_keys = set()
    [all_keys.update(x.keys()) for x in tracks]

    missing_val = np.array(([np.inf] * tracks[0][ego_key].shape[-1]))
    other_tracks = np.array([[x[k] if k in x else missing_val for x in tracks] \
                                for k in all_keys if k != ego_key])
    if len(other_tracks):
        invalid = (other_tracks == np.inf).any(axis = -1)
        valid = (1 - invalid)[:, :, np.newaxis]
        new_ids = 1 + np.arange(len(other_tracks))
        new_ids = np.repeat(new_ids[:, np.newaxis, np.newaxis], len(ego_track), axis=1)
        other_tracks = np.concatenate([other_tracks, new_ids, valid], axis=-1)
    else:
        other_tracks = np.empty((0, *ego_track.shape))

    return ego_track, other_tracks, np.array([k for k in all_keys if k != ego_key])


# Taken mostly from SafeShift codebase
def compute_leading_agent(ego_track, other_track,  mask = None):
    if mask is None:
        mask = np.arange(len(ego_track))
    
    ego_x, ego_y, ego_vx, ego_vy, ego_heading, ego_length, \
        ego_width, ego_crash, ego_id, ego_valid = ego_track.transpose()
    other_x, other_y, other_vx, other_vy, other_heading, other_length, \
        other_width, other_crash, other_id, other_valid = other_track.transpose()

    ego_x, ego_y = ego_x[mask], ego_y[mask]
    other_x, other_y = other_x[mask], other_y[mask]
    
    # heading in degrees, guaranteed to not differ from other person's heading by more than 45 degrees
    def angle_to(x1, y1, x2, y2, heading):
        # heading is already in radians here!
        #heading = np.deg2rad(heading)
        vector_to_other_point = np.array([x2 - x1, y2 - y1])
        
        angle_to_other_point = np.arctan2(vector_to_other_point[1], vector_to_other_point[0])
        angle_difference = angle_to_other_point - heading
        
        # Adjust the angle to be between -π and π
        while angle_difference > np.pi:
            angle_difference -= 2*np.pi
        while angle_difference < -np.pi:
            angle_difference += 2*np.pi
        
        return np.rad2deg(angle_difference)

    # Value is in degrees
    ego_to_other_angles = np.array([angle_to(x1, y1, x2, y2, h1) \
                                    for x1, y1, x2, y2, h1 in zip(ego_x, ego_y, other_x, other_y, ego_heading)])
    # Check if pos_j is "behind" pos_i
    ego_leading = np.abs(ego_to_other_angles) > 90
    # 0 means ego is leading, 1 means other is leading
    return (~ego_leading).astype(int)
