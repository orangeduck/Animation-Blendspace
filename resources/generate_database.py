import quat
import bvh
from scipy.interpolate import griddata
import scipy.signal as signal
import scipy.ndimage as ndimage
import struct
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

""" Basic function for mirroring animation data with this particular skeleton structure """

def animation_mirror(lrot, lpos, names, parents):

    joints_mirror = np.array([(
        names.index('Left'+n[5:]) if n.startswith('Right') else (
        names.index('Right'+n[4:]) if n.startswith('Left') else 
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([[-1, -1, 1], [1, 1, -1], [1, 1, -1]])

    grot, gpos = quat.fk(lrot, lpos, parents)

    gpos_mirror = mirror_pos * gpos[:,joints_mirror]
    grot_mirror = quat.from_xform(mirror_rot * quat.to_xform(grot[:,joints_mirror]))
    
    return quat.ik(grot_mirror, gpos_mirror, parents)

""" Looping Functionality """

def decay_cubic(
    x, 
    v, 
    blend_time, 
    dt):

    t = np.clip(dt / blend_time, 0, 1)

    d = x
    c = v * blend_time
    b = -3*d - 2*c
    a = 2*d + c
    
    return a*t*t*t + b*t*t + c*t + d


""" Files to Process """

files = [
    # Walking
    ('walk1_subject5.bvh', [
        # (175, 204), # Fwd
        
        # (204, 227), # Sharp Left
        # (289, 311), # Left
        # (463, 497), # Left
        # (529, 563), # Gentle Left
        # (589, 617), # Sharp Right
        # (650, 674), # Right
        # (781, 807), # Right
        # (844, 869), # Right
        
        (1188, 1219), # Spiral Right
        # (1219, 1252), # Spiral Right
        # (1252, 1282), # Spiral Right
        # (1282, 1312), # Spiral Right
        (1312, 1342), # Spiral Right
        (1342, 1372), # Spiral Right
        (1372, 1401), # Spiral Right
        
        (1434, 1465), # Spiral Left
        # (1465, 1498), # Spiral Left
        # (1498, 1526), # Spiral Left
        (1526, 1557), # Spiral Left
        # (1557, 1587), # Spiral Left
        # (1587, 1619), # Spiral Left
        (1619, 1649), # Spiral Left
        (1649, 1682), # Spiral Left
        
        (6230, 6255), # Turn on Spot Sharp Left
        (6366, 6397), # Turn on Spot Sharp Right
        (6707, 6737), # Turn on Sport Left
        # (6642, 6669), # Turn on Sport Right
        
        (1682, 1712), # Forward
    ]),
    # Running
    ('run1_subject5.bvh', [
        (187, 209), # Forward
        
        # (1715, 1737), # Spiral Left
        (1737, 1761), # Spiral Left
        (1761, 1785), # Spiral Left
        (1785, 1807), # Spiral Left
        # (1807, 1829), # Spiral Left
        (1829, 1853), # Spiral Left
        (1853, 1876), # Spiral Left
        (1876, 1898), # Spiral Left
        (1898, 1921), # Spiral Left
        
        (1945, 1969), # Spiral Right
        (1969, 1991), # Spiral Right
        (1991, 2013), # Spiral Right
        (2013, 2037), # Spiral Right
        # (2037, 2059), # Spiral Right
        # (2059, 2082), # Spiral Right
    ]),
    ('pushAndStumble1_subject5.bvh', [
        (215, 350), # Stand on spot
    ]),
]

""" We will accumulate data in these lists """

bone_positions = []
bone_velocities = []
bone_rotations = []
bone_angular_velocities = []
bone_parents = []
bone_names = []
    
range_starts = []
range_stops = []

contact_states = []

""" Loop Over Files """

for filename, intervals in files:
    
    # For each file we process it mirrored and not mirrored
    # for mirror in [False, True]:
    for mirror in [False]:
    
        """ Load Data """
        
        print('Loading "%s" %s...' % (filename, "(Mirrored)" if mirror else ""))
        
        bvh_data = bvh.load(filename)
        bvh_data['positions'] = bvh_data['positions']
        bvh_data['rotations'] = bvh_data['rotations']
        
        positions = bvh_data['positions']
        rotations = quat.unroll(quat.from_euler(np.radians(bvh_data['rotations']), order=bvh_data['order']))

        # Convert from cm to m
        positions *= 0.01
        
        if mirror:
            rotations, positions = animation_mirror(rotations, positions, bvh_data['names'], bvh_data['parents'])
            rotations = quat.unroll(rotations)
        
        """ Supersample """
        
        nframes = positions.shape[0]
        nbones = positions.shape[1]
        
        # Supersample data to 60 fps
        original_times = np.linspace(0, nframes - 1, nframes)
        # sample_times = np.linspace(0, nframes - 1, int(0.9 * (nframes * 2 - 1))) # Speed up data by 10%
        sample_times = np.linspace(0, nframes - 1, 2 * nframes)
        
        # This does a cubic interpolation of the data for supersampling and also speeding up by 10%
        positions = griddata(original_times, positions.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 3])
        rotations = griddata(original_times, rotations.reshape([nframes, -1]), sample_times, method='cubic').reshape([len(sample_times), nbones, 4])
        
        # Need to re-normalize after super-sampling
        rotations = quat.normalize(rotations)
        
        """ Extract Simulation Bone """
        
        # First compute world space positions/rotations
        global_rotations, global_positions = quat.fk(rotations, positions, bvh_data['parents'])
        
        # Specify joints to use for simulation bone 
        sim_position_joint = bvh_data['names'].index("Spine2")
        sim_rotation_joint = bvh_data['names'].index("Hips")
        
        # Position comes from spine joint
        sim_position = np.array([1.0, 0.0, 1.0]) * global_positions[:,sim_position_joint:sim_position_joint+1]
        sim_position = signal.savgol_filter(sim_position, 31, 3, axis=0, mode='interp')
        
        # Direction comes from projected hip forward direction
        sim_direction = np.array([1.0, 0.0, 1.0]) * quat.mul_vec(global_rotations[:,sim_rotation_joint:sim_rotation_joint+1], np.array([0.0, 1.0, 0.0]))

        # We need to re-normalize the direction after both projection and smoothing
        sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1))[...,np.newaxis]
        sim_direction = signal.savgol_filter(sim_direction, 61, 3, axis=0, mode='interp')
        sim_direction = sim_direction / np.sqrt(np.sum(np.square(sim_direction), axis=-1)[...,np.newaxis])
        
        # Extract rotation from direction
        sim_rotation = quat.normalize(quat.between(np.array([0, 0, 1]), sim_direction))

        # Transform first joints to be local to sim and append sim as root bone
        positions[:,0:1] = quat.mul_vec(quat.inv(sim_rotation), positions[:,0:1] - sim_position)
        rotations[:,0:1] = quat.mul(quat.inv(sim_rotation), rotations[:,0:1])
        
        positions = np.concatenate([sim_position, positions], axis=1)
        rotations = np.concatenate([sim_rotation, rotations], axis=1)
        
        bone_parents = np.concatenate([[-1], bvh_data['parents'] + 1])
        
        bone_names = ['Simulation'] + bvh_data['names']

        for start, stop in intervals:
            
            positions_int = positions[start*2:stop*2].copy()
            rotations_int = rotations[start*2:stop*2].copy()
            
            """ Make Loop """
            
            dt = 1.0 / 60.0
            
            positions_offset = positions_int[-1] - positions_int[0]
            positions_offset_vel = (positions_int[-1] - positions_int[-2]) / dt - (positions_int[1] - positions_int[0]) / dt 
            
            rotations_offset = quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations_int[-1], rotations_int[0])))
            rotations_offset_vel = (
                quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations_int[-1], rotations_int[-2]))) / dt -
                quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations_int[1], rotations_int[0]))) / dt)
            
            dts = dt * np.arange(len(positions_int))[:,None,None]
            
            positions_int[:,1:] = (decay_cubic(positions_offset, positions_offset_vel, 0.15, dts) + positions_int)[:,1:]
            rotations_int[:,1:] = quat.mul(quat.from_scaled_angle_axis(decay_cubic(rotations_offset, rotations_offset_vel, 0.15, dts)), rotations_int)[:,1:]
            
            # Root Adjustment
            
            root_start_rot = rotations_int[0,0].copy()
            root_end_rot = rotations_int[-1,0].copy()
            
            pos_end_vel = quat.inv_mul_vec(root_end_rot, (positions_int[-1,0] - positions_int[-2,0]) / dt)
            pos_start_vel = quat.inv_mul_vec(root_start_rot, (positions_int[1,0] - positions_int[0,0]) / dt)
            pos_diff_vel = pos_end_vel - pos_start_vel
            
            rot_end_vel = quat.inv_mul_vec(root_end_rot, quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations_int[-1,0], rotations_int[-2,0]))) / dt)
            rot_start_vel = quat.inv_mul_vec(root_start_rot, quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rotations_int[1,0], rotations_int[0,0]))) / dt)
            rot_diff_vel = rot_end_vel - rot_start_vel
            
            positions_int[:,0] = (decay_cubic(np.zeros(3), quat.mul_vec(root_start_rot, pos_diff_vel), 0.5, dts[:,0]) + positions_int[:,0])
            rotations_int[:,0] = quat.mul(quat.from_scaled_angle_axis(decay_cubic(np.zeros(3), quat.mul_vec(root_start_rot, rot_diff_vel), 0.5, dts[:,0])), rotations_int[:,0])
            
            bvh.save('loop_%i.bvh' % len(range_starts), dict(
                positions=positions_int,
                rotations=np.degrees(quat.to_euler(rotations_int)),
                offsets=positions_int[0],
                names=bone_names,
                parents=bone_parents,
                order=bvh_data['order'],
            ))
            
            """ Append to Database """
            
            bone_positions.append(positions_int)
            bone_rotations.append(rotations_int)
            
            offset = 0 if len(range_starts) == 0 else range_stops[-1] 

            range_starts.append(offset)
            range_stops.append(offset + len(positions_int))
    
    
""" Concatenate Data """
    
bone_positions = np.concatenate(bone_positions, axis=0).astype(np.float32)
bone_rotations = np.concatenate(bone_rotations, axis=0).astype(np.float32)
bone_parents = bone_parents.astype(np.int32)

range_starts = np.array(range_starts).astype(np.int32)
range_stops = np.array(range_stops).astype(np.int32)
    
""" Write Database """

print("Writing Database...")

with open('database.bin', 'wb') as f:
    
    nframes = bone_positions.shape[0]
    nbones = bone_positions.shape[1]
    nranges = range_starts.shape[0]
    
    f.write(struct.pack('II', nframes, nbones) + bone_positions.ravel().tobytes())
    f.write(struct.pack('II', nframes, nbones) + bone_rotations.ravel().tobytes())
    f.write(struct.pack('I', nbones) + bone_parents.ravel().tobytes())
    
    f.write(struct.pack('I', nranges) + range_starts.ravel().tobytes())
    f.write(struct.pack('I', nranges) + range_stops.ravel().tobytes())

    
    