extern "C"
{
#include "raylib.h"
#include "raymath.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
}
#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#include "common.h"
#include "vec.h"
#include "mat.h"
#include "quat.h"
#include "array.h"
#include "character.h"
#include "database.h"
#include "triangulate.h"
#include "nnet.h"
#include "spring.h"

#include <initializer_list>
#include <vector>
#include <functional>

//--------------------------------------

static inline Vector3 to_Vector3(vec3 v)
{
    return (Vector3){ v.x, v.y, v.z };
}

//--------------------------------------

// Perform linear blend skinning and copy 
// result into mesh data. Update and upload 
// deformed vertex positions and normals to GPU
void deform_character_mesh(
  Mesh& mesh, 
  const character& c,
  const slice1d<vec3> bone_anim_positions,
  const slice1d<quat> bone_anim_rotations,
  const slice1d<int> bone_parents)
{
    linear_blend_skinning_positions(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.vertices),
        c.positions,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_positions,
        c.bone_rest_rotations,
        bone_anim_positions,
        bone_anim_rotations);
    
    linear_blend_skinning_normals(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.normals),
        c.normals,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_rotations,
        bone_anim_rotations);
    
    UpdateMeshBuffer(mesh, 0, mesh.vertices, mesh.vertexCount * 3 * sizeof(float), 0);
    UpdateMeshBuffer(mesh, 2, mesh.normals, mesh.vertexCount * 3 * sizeof(float), 0);
}

Mesh make_character_mesh(character& c)
{
    Mesh mesh = { 0 };
    
    mesh.vertexCount = c.positions.size;
    mesh.triangleCount = c.triangles.size / 3;
    mesh.vertices = (float*)MemAlloc(c.positions.size * 3 * sizeof(float));
    mesh.texcoords = (float*)MemAlloc(c.texcoords.size * 2 * sizeof(float));
    mesh.normals = (float*)MemAlloc(c.normals.size * 3 * sizeof(float));
    mesh.indices = (unsigned short*)MemAlloc(c.triangles.size * sizeof(unsigned short));
    
    memcpy(mesh.vertices, c.positions.data, c.positions.size * 3 * sizeof(float));
    memcpy(mesh.texcoords, c.texcoords.data, c.texcoords.size * 2 * sizeof(float));
    memcpy(mesh.normals, c.normals.data, c.normals.size * 3 * sizeof(float));
    memcpy(mesh.indices, c.triangles.data, c.triangles.size * sizeof(unsigned short));
    
    UploadMesh(&mesh, true);
    
    return mesh;
}

//--------------------------------------

float orbit_camera_update_azimuth(
    const float azimuth, 
    const float mouse_dx,
    const float dt)
{
    return azimuth + 1.0f * dt * -mouse_dx;
}

float orbit_camera_update_altitude(
    const float altitude, 
    const float mouse_dy,
    const float dt)
{
    return clampf(altitude + 1.0f * dt * mouse_dy, 0.0, 0.4f * PIf);
}

float orbit_camera_update_distance(
    const float distance, 
    const float dt)
{
    return clampf(distance +  20.0f * dt * -GetMouseWheelMove(), 0.1f, 100.0f);
}

// Updates the camera using the orbit cam controls
void orbit_camera_update(
    Camera3D& cam, 
    float& camera_azimuth,
    float& camera_altitude,
    float& camera_distance,
    const vec3 target,
    const float mouse_dx,
    const float mouse_dy,
    const float dt)
{
    camera_azimuth = orbit_camera_update_azimuth(camera_azimuth, mouse_dx, dt);
    camera_altitude = orbit_camera_update_altitude(camera_altitude, mouse_dy, dt);
    camera_distance = orbit_camera_update_distance(camera_distance, dt);
    
    quat rotation_azimuth = quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0));
    vec3 position = quat_mul_vec3(rotation_azimuth, vec3(0, 0, camera_distance));
    vec3 axis = normalize(cross(position, vec3(0, 1, 0)));
    
    quat rotation_altitude = quat_from_angle_axis(camera_altitude, axis);
    
    vec3 eye = target + quat_mul_vec3(rotation_altitude, position);

    cam.target = (Vector3){ target.x, target.y, target.z };
    cam.position = (Vector3){ eye.x, eye.y, eye.z };
}

//--------------------------------------

// Fits a Delauney triangulation to the given set of points
// int the parameter space.
void fit_triangulation(
    array1d<delauney_tri>& parameter_tris,
    const slice2d<float> animation_parameters)
{
    assert(animation_parameters.cols == 2);
  
    // Allocate triangles array
    parameter_tris.resize(animation_parameters.rows * 3);
    parameter_tris.zero();
    
    // Allocate space for points which includes index
    // so we do not lose ordering after sorting
    array1d<delauney_point> delauney_points(animation_parameters.rows + 3);
    delauney_points.zero();
    
    for (int i = 0; i < animation_parameters.rows; i++)
    {
        delauney_points(i).index = i;
        delauney_points(i).x = animation_parameters(i, 0);
        delauney_points(i).y = animation_parameters(i, 1);
    }
    
    // Fit triangulation
    int tri_num = 0;
    int status = delauney_triangulate(
        &tri_num, 
        parameter_tris, 
        animation_parameters.rows, 
        delauney_points);
    
    assert(status == 0);
    
    // Copy found points into triangles array
    parameter_tris.resize(tri_num);
    for (int i = 0; i < parameter_tris.size; i++)
    {
        parameter_tris(i).p1 = delauney_points(parameter_tris(i).p1).index;
        parameter_tris(i).p2 = delauney_points(parameter_tris(i).p2).index;
        parameter_tris(i).p3 = delauney_points(parameter_tris(i).p3).index;
    }
}

//--------------------------------------

// Computes a weighted average of bone positions
void weighted_average_positions(
    slice1d<vec3> out_positions,
    slice2d<vec3> anim_positions,
    slice1d<float> blend_weights)
{
    out_positions.zero();
  
    for (int i = 0; i < blend_weights.size; i++)
    {
        for (int j = 0; j < out_positions.size; j++)
        {
            out_positions(j) += blend_weights(i) * anim_positions(i, j);
        }
    }
}

// Computes a weighted average of bone rotations
// See: https://theorangeduck.com/page/quaternion-weighted-average
void weighted_average_rotations(
    slice1d<quat> out_rotations,
    slice1d<mat4> accum_rotations,
    const slice2d<quat> anim_rotations, 
    const slice1d<float> blend_weights)
{
    assert(anim_rotations.rows == blend_weights.size);
    assert(anim_rotations.cols == out_rotations.size);
    assert(anim_rotations.cols == accum_rotations.size);
    
    accum_rotations.zero();
    
    for (int i = 0; i < anim_rotations.rows; i++)
    {
        for (int j = 0; j < anim_rotations.cols; j++)
        {
            quat q = anim_rotations(i, j);
          
            accum_rotations(j) = accum_rotations(j) + blend_weights(i) * mat4(
                q.w*q.w, q.w*q.x, q.w*q.y, q.w*q.z,
                q.x*q.w, q.x*q.x, q.x*q.y, q.x*q.z,
                q.y*q.w, q.y*q.x, q.y*q.y, q.y*q.z,
                q.z*q.w, q.z*q.x, q.z*q.y, q.z*q.z);
        }
    }
    
    for (int j = 0; j < anim_rotations.cols; j++)
    {
        vec4 guess = vec4(1, 0, 0, 0);
        vec4 u = mat4_svd_dominant_eigen(accum_rotations(j), guess, 64, 1e-8f);
        vec4 v = normalize(mat4_transpose_mul_vec4(accum_rotations(j), u));
        
        out_rotations(j) = quat_abs(quat(v.x, v.y, v.z, v.w));
    }
}

// Computes a cheaper weighted average of bone rotations 
// using the reference pose
void weighted_average_rotations_ref(
    slice1d<quat> out_rotations, 
    const slice1d<quat> reference_rotations,
    const slice2d<quat> anim_rotations, 
    const slice1d<float> blend_weights)
{
    assert(anim_rotations.rows == blend_weights.size);
    assert(anim_rotations.cols == out_rotations.size);
    assert(anim_rotations.cols == reference_rotations.size);
    
    out_rotations.zero();
    
    for (int i = 0; i < anim_rotations.rows; i++)
    {
        for (int j = 0; j < anim_rotations.cols; j++)
        {
            out_rotations(j) = out_rotations(j) + blend_weights(i) * 
                quat_abs(quat_inv_mul(
                    reference_rotations(j), anim_rotations(i, j)));
        }
    }
    
    for (int j = 0; j < anim_rotations.cols; j++)
    {
        out_rotations(j) = quat_abs(quat_mul(
            reference_rotations(j), quat_normalize(out_rotations(j))));
    }
}

// Samples a pose from the database for the given animation
// at the given normalized time value between 0 and 1
void database_sample(
    slice1d<vec3> sample_positions,
    slice1d<quat> sample_rotations,
    database& db,
    int anim,
    float time)
{
    float frame_time = time * (db.range_stops(anim) - db.range_starts(anim)) + db.range_starts(anim);
    int i0 = clamp((int)frame_time + 0, db.range_starts(anim), db.range_stops(anim) - 1);
    int i1 = clamp((int)frame_time + 1, db.range_starts(anim), db.range_stops(anim) - 1);
    float alpha = fmod(frame_time, 1.0f);
    
    for (int j = 0; j < sample_positions.size; j++)
    {
        sample_positions(j) = lerp(db.bone_positions(i0, j), db.bone_positions(i1, j), alpha);
        sample_rotations(j) = quat_slerp_shortest(db.bone_rotations(i0, j), db.bone_rotations(i1, j), alpha);
    }
}

//--------------------------------------

// Computes the average animation parameters for 
// speed and turning angle (average angular velocity) 
void compute_average_animation_parameters_speed_turn(
    slice2d<float> animation_parameters, 
    database& db, 
    float dt)
{
    for (int i = 0; i < db.nranges(); i++)
    {
        int range_frame_num = db.range_stops(i) - db.range_starts(i) - 1;
        
        // Compute average angular velocity
        float angvel = 0.0f;
        for (int j = db.range_starts(i); j < db.range_stops(i) - 1; j++)
        {
            angvel += quat_to_scaled_angle_axis(quat_abs(quat_mul_inv(
                db.bone_rotations(j + 1, 0),
                db.bone_rotations(j + 0, 0)))).y / (range_frame_num * dt);
        }
        
        // Map into range 0 to 1
        animation_parameters(i, 0) = angvel / 9.0f + 0.5f; 
    }
    
    for (int i = 0; i < db.nranges(); i++)
    {
        int range_frame_num = db.range_stops(i) - db.range_starts(i) - 1;

        // Compute average speed
        float speed = 0.0f;
        for (int j = db.range_starts(i); j < db.range_stops(i) - 1; j++)
        {
            speed += length(
                db.bone_positions(j + 1, 0) - 
                db.bone_positions(j + 0, 0)) / (range_frame_num * dt);
        }
      
        // Map into range 0 to 1
        animation_parameters(i, 1) = 1.0f - speed / 4.0f;
    }
}

// Computes the animation root rotation and location accounting 
// for looping (and so can be provided a large frame parameter)
static inline void animation_root_looped(
    vec3& out_pos,
    quat& out_rot,
    slice2d<vec3> bone_positions, 
    slice2d<quat> bone_rotations,
    int frame)
{
    int nframes = bone_positions.rows;
  
    out_pos = vec3();
    out_rot = quat();
    while (frame >= nframes)
    {
        // given frame is larger than the number of frames insert 
        // full change in location and rotation for the whole anim
        vec3 pos_diff = quat_inv_mul_vec3(bone_rotations(0,0), bone_positions(nframes - 1,0) - bone_positions(0,0));
        quat rot_diff = quat_mul_inv(bone_rotations(nframes - 1,0), bone_rotations(0,0));
      
        out_pos = quat_mul_vec3(out_rot, pos_diff) + out_pos;
        out_rot = quat_mul(rot_diff, out_rot);
        frame -= nframes;
    }
    
    // Get the change in location and rotation up to the given frame
    vec3 pos_diff = quat_inv_mul_vec3(bone_rotations(0,0), bone_positions(frame,0) - bone_positions(0,0));
    quat rot_diff = quat_mul_inv(bone_rotations(frame,0), bone_rotations(0,0));
    
    out_pos = quat_mul_vec3(out_rot, pos_diff) + out_pos;
    out_rot = quat_mul(rot_diff, out_rot);
}

// Compute the per-frame animation parameters for speed and turn rate
void compute_frame_animation_parameters_speed_turn(
    slice2d<float> animation_parameters, 
    database& db, 
    float dt, 
    float normalized_time)
{
    for (int i = 0; i < db.nranges(); i++)
    {
        int range_frame_num = db.range_stops(i) - db.range_starts(i);

        int frame = db.range_starts(i) + clamp((int)(normalized_time * (range_frame_num - 1)), 0, range_frame_num - 2);
      
        float angvel = quat_to_scaled_angle_axis(quat_abs(quat_mul_inv(
                db.bone_rotations(frame + 0, 0),
                db.bone_rotations(frame + 1, 0)))).y / dt;

        animation_parameters(i, 0) = angvel / 9.0f + 0.5f; 
    }
    
    for (int i = 0; i < db.nranges(); i++)
    {
        int range_frame_num = db.range_stops(i) - db.range_starts(i);

        int frame = db.range_starts(i) + clamp((int)(normalized_time * (range_frame_num - 1)), 0, range_frame_num - 2);

        float speed = length(
                db.bone_positions(frame + 0, 0) - 
                db.bone_positions(frame + 1, 0)) / dt;
        
        animation_parameters(i, 1) = 1.0f - speed / 4.0f;
    }
}

// Compute the average future trajectory position and direction animation parameters
void compute_average_animation_parameters_trajectory(
    slice2d<float> animation_parameters, 
    database& db)
{
    for (int i = 0; i < db.nranges(); i++)
    {
        int range_frame_num = db.range_stops(i) - db.range_starts(i);
        
        for (int s = 0; s < 3; s++)
        {
            vec3 pos = vec3();
            vec3 dir = vec3();
            
            for (int j = 0; j < range_frame_num; j++)
            {
                vec3 base_pos;
                quat base_rot;
                animation_root_looped(
                    base_pos, base_rot,
                    db.bone_positions.slice(db.range_starts(i), db.range_stops(i)),
                    db.bone_rotations.slice(db.range_starts(i), db.range_stops(i)),
                    j);
              
                vec3 off_pos;
                quat off_rot;
                animation_root_looped(
                    off_pos, off_rot,
                    db.bone_positions.slice(db.range_starts(i), db.range_stops(i)),
                    db.bone_rotations.slice(db.range_starts(i), db.range_stops(i)),
                    j + (s + 1) * 20);
              
                vec3 loc_pos = quat_inv_mul_vec3(base_rot, off_pos - base_pos);
                vec3 loc_dir = quat_inv_mul_vec3(base_rot, quat_mul_vec3(off_rot, vec3(0,0,1)));
              
                pos += loc_pos / range_frame_num;
                dir += loc_dir / range_frame_num;
            }
            
            dir = normalize(dir);

            animation_parameters(i, s * 4 + 0) = pos.x;
            animation_parameters(i, s * 4 + 1) = pos.z;
            animation_parameters(i, s * 4 + 2) = dir.x;
            animation_parameters(i, s * 4 + 3) = dir.z;
        }
    }
}

// Compute the per-frame future trajectory positions and directions
void compute_frame_animation_parameters_trajectory(
    slice2d<float> animation_parameters, 
    database& db, 
    float normalized_time)
{
    for (int i = 0; i < db.nranges(); i++)
    {
        int range_frame_num = db.range_stops(i) - db.range_starts(i);
        int frame = db.range_starts(i) + clamp((int)(normalized_time * (range_frame_num - 1)), 0, range_frame_num - 2);
        
        for (int s = 0; s < 3; s++)
        {
            vec3 base_pos;
            quat base_rot;
            animation_root_looped(
                base_pos, base_rot,
                db.bone_positions.slice(db.range_starts(i), db.range_stops(i)),
                db.bone_rotations.slice(db.range_starts(i), db.range_stops(i)),
                frame);
          
            vec3 off_pos;
            quat off_rot;
            animation_root_looped(
                off_pos, off_rot,
                db.bone_positions.slice(db.range_starts(i), db.range_stops(i)),
                db.bone_rotations.slice(db.range_starts(i), db.range_stops(i)),
                frame + (s + 1) * 20);
          
            vec3 pos = quat_inv_mul_vec3(base_rot, off_pos - base_pos);
            vec3 dir = quat_inv_mul_vec3(base_rot, quat_mul_vec3(off_rot, vec3(0,0,1)));

            animation_parameters(i, s * 4 + 0) = pos.x;
            animation_parameters(i, s * 4 + 1) = pos.z;
            animation_parameters(i, s * 4 + 2) = dir.x;
            animation_parameters(i, s * 4 + 3) = dir.z;
        }
    }
}

// Clamp and normalize blend weights so that they are greater than
// zero and sum to one
void clamp_normalize_blend_weights(slice1d<float> blend_weights)
{
    float total_blend_weight = 0.0f;
    for (int i = 0; i < blend_weights.size; i++)
    {
        blend_weights(i) = maxf(blend_weights(i), 0.0f);
        total_blend_weight += blend_weights(i);
    }
    
    for (int i = 0; i < blend_weights.size; i++)
    {
        blend_weights(i) /= total_blend_weight;
    }
}

// Compute the distances in the parameter space from the 
// current query to all of the animation parameters
void compute_query_distances(
    slice1d<float> query_distances, 
    const slice2d<float> animation_parameters, 
    const slice1d<float> query_parameters)
{
    assert(query_distances.size == animation_parameters.rows);
  
    for (int i = 0; i < animation_parameters.rows; i++)
    {
        query_distances(i) = 0.0f;
        for (int j = 0; j < animation_parameters.cols; j++)
        {
            query_distances(i) += squaref(
                query_parameters(j) - animation_parameters(i, j));
        }
        query_distances(i) = sqrtf(query_distances(i));
    }
}

// Compute blend weights given a blend matrix and distances
void compute_blend_weights(
    slice1d<float> blend_weights,
    const slice1d<float> query_distances,
    const slice2d<float> blend_matrix)
{
    return mat_mul_vec(blend_weights, blend_matrix, query_distances);
}

// Fit the blend matrix to the given animation parameters
void fit_blend_matrix(
    slice2d<float> blend_matrix,
    const slice2d<float> animation_parameters)
{    
    int nanims = animation_parameters.rows;
    int nparams = animation_parameters.cols;
    
    // Compute Pairwise Distances

    array2d<float> distances(nanims, nanims);
    
    for (int i = 0; i < nanims; i++)
    {
        for (int j = 0; j < nanims; j++)
        {
            distances(i, j) = 0.0f;
            for (int k = 0; k < nparams; k++)
            {
                distances(i, j) += squaref(
                    animation_parameters(i, k) - 
                    animation_parameters(j, k));
            }
            distances(i, j) = sqrtf(distances(i, j));
        }
    }
    
    // Subtract epsilon from diagonal this helps the stability
    // of the decomposition and solve
    
    for (int i = 0; i < nanims; i++)
    {
        distances(i, i) -= 1e-4f;
    }
    
    // Decompose in place
    
    array1d<int> row_order(nanims);
    array1d<float> row_scale(nanims);
    
    bool success = mat_lu_decompose_inplace(distances, row_order, row_scale);
    assert(success);
    
    // Write associated blend weights into blend matrix
    
    for (int i = 0; i < nanims; i++)
    {
        for (int j = 0; j < nanims; j++)
        {
            blend_matrix(i, j) = i == j ? 1.0f : 0.0f;
        }
    }
    
    // Solve for blend matrix in-place
    
    for (int i = 0; i < nanims; i++)
    {
        mat_lu_solve_inplace(blend_matrix(i), distances, row_order);
    }
}

//--------------------------------------

// Project a point onto a line segment
static inline vec2 triangulation_edge_proj(vec2 p, vec2 a, vec2 b)
{
    float l2 = dot(a - b, a - b);
    if (l2 == 0.0f) { return a; }
    
    float t = clampf(dot(p - a, b - a) / l2, 0.0f, 1.0f);
    return a + t * (b - a);
}

// Project a point onto the triangulation convex hull
void project_onto_convex_hull(
    slice1d<float> query_parameters,
    const slice1d<float> target_parameters,
    const slice2d<float> animation_parameters,
    const slice1d<delauney_tri> parameter_tris)
{
    float best_dist = FLT_MAX;
    vec2 best_point = vec2();
    
    for (int i = 0; i < parameter_tris.size; i++)
    {
        int p1 = parameter_tris(i).p1;
        int p2 = parameter_tris(i).p2;
        int p3 = parameter_tris(i).p3;
      
        vec2 a = vec2(animation_parameters(p1,0), animation_parameters(p1,1));
        vec2 b = vec2(animation_parameters(p2,0), animation_parameters(p2,1));
        vec2 c = vec2(animation_parameters(p3,0), animation_parameters(p3,1));
        vec2 p = vec2(target_parameters(0), target_parameters(1));

        vec2 v0 = b - a, v1 = c - a, v2 = p - a;
        float d00 = dot(v0, v0);
        float d01 = dot(v0, v1);
        float d11 = dot(v1, v1);
        float d20 = dot(v2, v0);
        float d21 = dot(v2, v1);
        float denom = d00 * d11 - d01 * d01;
        
        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0f - v - w;
        
        if ((u >= 0.0f) && (v >= 0.0f) && (u + v < 1.0f))
        {
            query_parameters(0) = target_parameters(0);
            query_parameters(1) = target_parameters(1);
            return;
        }

        vec2 e1 = triangulation_edge_proj(p, a, b);
        vec2 e2 = triangulation_edge_proj(p, b, c);
        vec2 e3 = triangulation_edge_proj(p, c, a);
        float d1 = length(e1 - p);
        float d2 = length(e2 - p);
        float d3 = length(e3 - p);
        
        if (d1 < best_dist)
        {
            best_dist = d1;
            best_point = e1;
        }
        
        if (d2 < best_dist)
        {
            best_dist = d2;
            best_point = e2;
        }
        
        if (d3 < best_dist)
        {
            best_dist = d3;
            best_point = e3;
        }
    }
    
    query_parameters(0) = best_point.x;
    query_parameters(1) = best_point.y;
}

// Compute the blend weights for the query point given
// a triangulation of the space. Assumes the query point
// is within the convex hull.
void compute_blend_weights_triangulation(
    slice1d<float> blend_weights,
    const slice2d<float> animation_parameters,
    const slice1d<float> query_parameters,
    const slice1d<delauney_tri> parameter_tris)
{  
    blend_weights.zero();

    for (int i = 0; i < parameter_tris.size; i++)
    {
        int p1 = parameter_tris(i).p1;
        int p2 = parameter_tris(i).p2;
        int p3 = parameter_tris(i).p3;
      
        vec2 a = vec2(animation_parameters(p1,0), animation_parameters(p1,1));
        vec2 b = vec2(animation_parameters(p2,0), animation_parameters(p2,1));
        vec2 c = vec2(animation_parameters(p3,0), animation_parameters(p3,1));
        vec2 p = vec2(query_parameters(0), query_parameters(1));
      
        vec2 v0 = b - a, v1 = c - a, v2 = p - a;
        float d00 = dot(v0, v0);
        float d01 = dot(v0, v1);
        float d11 = dot(v1, v1);
        float d20 = dot(v2, v0);
        float d21 = dot(v2, v1);
        float denom = d00 * d11 - d01 * d01;
        
        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0f - v - w;
        float eps = 1e-3f;
        
        if ((u >= 0.0f - eps) && (v >= 0.0f - eps) && (u + v < 1.0f + eps))
        {
            blend_weights(p1) = clampf(u, 0.0f, 1.0f);
            blend_weights(p2) = clampf(v, 0.0f, 1.0f);
            blend_weights(p3) = clampf(w, 0.0f, 1.0f);
            
            float sum = blend_weights(p1) + blend_weights(p2) + blend_weights(p3);
            blend_weights(p1) /= sum;
            blend_weights(p2) /= sum;
            blend_weights(p3) /= sum;
            return;
        }
    }
    
    blend_weights(0) = 1.0f;
    return;
}

//--------------------------------------

void draw_axis(const vec3 pos, const quat rot, const float scale = 1.0f)
{
    vec3 axis0 = pos + quat_mul_vec3(rot, scale * vec3(1.0f, 0.0f, 0.0f));
    vec3 axis1 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 1.0f, 0.0f));
    vec3 axis2 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 0.0f, 1.0f));
    
    DrawLine3D(to_Vector3(pos), to_Vector3(axis0), RED);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis1), GREEN);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis2), BLUE);
}

void draw_axis(const vec3 pos, const mat3 rot, const float scale = 1.0f)
{
    vec3 axis0 = pos + mat3_mul_vec3(rot, scale * vec3(1.0f, 0.0f, 0.0f));
    vec3 axis1 = pos + mat3_mul_vec3(rot, scale * vec3(0.0f, 1.0f, 0.0f));
    vec3 axis2 = pos + mat3_mul_vec3(rot, scale * vec3(0.0f, 0.0f, 1.0f));
    
    DrawLine3D(to_Vector3(pos), to_Vector3(axis0), RED);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis1), GREEN);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis2), BLUE);
}

void draw_trajectory_parameters(slice1d<float> parameters, Color color)
{
    vec3 pos0 = vec3(parameters(0), 0, parameters(1));
    vec3 dir0 = vec3(parameters(2), 0, parameters(3));
    vec3 pos1 = vec3(parameters(4), 0, parameters(5));
    vec3 dir1 = vec3(parameters(6), 0, parameters(7));
    vec3 pos2 = vec3(parameters(8), 0, parameters(9));
    vec3 dir2 = vec3(parameters(10), 0, parameters(11));
    
    DrawSphereWires(to_Vector3(pos0), 0.05f, 4, 10, color);
    DrawSphereWires(to_Vector3(pos1), 0.05f, 4, 10, color);
    DrawSphereWires(to_Vector3(pos2), 0.05f, 4, 10, color);
    
    DrawLine3D(to_Vector3(pos0), to_Vector3(pos0 + 0.6f * dir0), color);
    DrawLine3D(to_Vector3(pos1), to_Vector3(pos1 + 0.6f * dir1), color);
    DrawLine3D(to_Vector3(pos2), to_Vector3(pos2 + 0.6f * dir2), color);

    DrawLine3D(to_Vector3(pos0), to_Vector3(pos1), color);
    DrawLine3D(to_Vector3(pos1), to_Vector3(pos2), color);    
}

//--------------------------------------

static float viridis_data[][3] = {
      {0.267004, 0.004874, 0.329415},
      {0.268510, 0.009605, 0.335427},
      {0.269944, 0.014625, 0.341379},
      {0.271305, 0.019942, 0.347269},
      {0.272594, 0.025563, 0.353093},
      {0.273809, 0.031497, 0.358853},
      {0.274952, 0.037752, 0.364543},
      {0.276022, 0.044167, 0.370164},
      {0.277018, 0.050344, 0.375715},
      {0.277941, 0.056324, 0.381191},
      {0.278791, 0.062145, 0.386592},
      {0.279566, 0.067836, 0.391917},
      {0.280267, 0.073417, 0.397163},
      {0.280894, 0.078907, 0.402329},
      {0.281446, 0.084320, 0.407414},
      {0.281924, 0.089666, 0.412415},
      {0.282327, 0.094955, 0.417331},
      {0.282656, 0.100196, 0.422160},
      {0.282910, 0.105393, 0.426902},
      {0.283091, 0.110553, 0.431554},
      {0.283197, 0.115680, 0.436115},
      {0.283229, 0.120777, 0.440584},
      {0.283187, 0.125848, 0.444960},
      {0.283072, 0.130895, 0.449241},
      {0.282884, 0.135920, 0.453427},
      {0.282623, 0.140926, 0.457517},
      {0.282290, 0.145912, 0.461510},
      {0.281887, 0.150881, 0.465405},
      {0.281412, 0.155834, 0.469201},
      {0.280868, 0.160771, 0.472899},
      {0.280255, 0.165693, 0.476498},
      {0.279574, 0.170599, 0.479997},
      {0.278826, 0.175490, 0.483397},
      {0.278012, 0.180367, 0.486697},
      {0.277134, 0.185228, 0.489898},
      {0.276194, 0.190074, 0.493001},
      {0.275191, 0.194905, 0.496005},
      {0.274128, 0.199721, 0.498911},
      {0.273006, 0.204520, 0.501721},
      {0.271828, 0.209303, 0.504434},
      {0.270595, 0.214069, 0.507052},
      {0.269308, 0.218818, 0.509577},
      {0.267968, 0.223549, 0.512008},
      {0.266580, 0.228262, 0.514349},
      {0.265145, 0.232956, 0.516599},
      {0.263663, 0.237631, 0.518762},
      {0.262138, 0.242286, 0.520837},
      {0.260571, 0.246922, 0.522828},
      {0.258965, 0.251537, 0.524736},
      {0.257322, 0.256130, 0.526563},
      {0.255645, 0.260703, 0.528312},
      {0.253935, 0.265254, 0.529983},
      {0.252194, 0.269783, 0.531579},
      {0.250425, 0.274290, 0.533103},
      {0.248629, 0.278775, 0.534556},
      {0.246811, 0.283237, 0.535941},
      {0.244972, 0.287675, 0.537260},
      {0.243113, 0.292092, 0.538516},
      {0.241237, 0.296485, 0.539709},
      {0.239346, 0.300855, 0.540844},
      {0.237441, 0.305202, 0.541921},
      {0.235526, 0.309527, 0.542944},
      {0.233603, 0.313828, 0.543914},
      {0.231674, 0.318106, 0.544834},
      {0.229739, 0.322361, 0.545706},
      {0.227802, 0.326594, 0.546532},
      {0.225863, 0.330805, 0.547314},
      {0.223925, 0.334994, 0.548053},
      {0.221989, 0.339161, 0.548752},
      {0.220057, 0.343307, 0.549413},
      {0.218130, 0.347432, 0.550038},
      {0.216210, 0.351535, 0.550627},
      {0.214298, 0.355619, 0.551184},
      {0.212395, 0.359683, 0.551710},
      {0.210503, 0.363727, 0.552206},
      {0.208623, 0.367752, 0.552675},
      {0.206756, 0.371758, 0.553117},
      {0.204903, 0.375746, 0.553533},
      {0.203063, 0.379716, 0.553925},
      {0.201239, 0.383670, 0.554294},
      {0.199430, 0.387607, 0.554642},
      {0.197636, 0.391528, 0.554969},
      {0.195860, 0.395433, 0.555276},
      {0.194100, 0.399323, 0.555565},
      {0.192357, 0.403199, 0.555836},
      {0.190631, 0.407061, 0.556089},
      {0.188923, 0.410910, 0.556326},
      {0.187231, 0.414746, 0.556547},
      {0.185556, 0.418570, 0.556753},
      {0.183898, 0.422383, 0.556944},
      {0.182256, 0.426184, 0.557120},
      {0.180629, 0.429975, 0.557282},
      {0.179019, 0.433756, 0.557430},
      {0.177423, 0.437527, 0.557565},
      {0.175841, 0.441290, 0.557685},
      {0.174274, 0.445044, 0.557792},
      {0.172719, 0.448791, 0.557885},
      {0.171176, 0.452530, 0.557965},
      {0.169646, 0.456262, 0.558030},
      {0.168126, 0.459988, 0.558082},
      {0.166617, 0.463708, 0.558119},
      {0.165117, 0.467423, 0.558141},
      {0.163625, 0.471133, 0.558148},
      {0.162142, 0.474838, 0.558140},
      {0.160665, 0.478540, 0.558115},
      {0.159194, 0.482237, 0.558073},
      {0.157729, 0.485932, 0.558013},
      {0.156270, 0.489624, 0.557936},
      {0.154815, 0.493313, 0.557840},
      {0.153364, 0.497000, 0.557724},
      {0.151918, 0.500685, 0.557587},
      {0.150476, 0.504369, 0.557430},
      {0.149039, 0.508051, 0.557250},
      {0.147607, 0.511733, 0.557049},
      {0.146180, 0.515413, 0.556823},
      {0.144759, 0.519093, 0.556572},
      {0.143343, 0.522773, 0.556295},
      {0.141935, 0.526453, 0.555991},
      {0.140536, 0.530132, 0.555659},
      {0.139147, 0.533812, 0.555298},
      {0.137770, 0.537492, 0.554906},
      {0.136408, 0.541173, 0.554483},
      {0.135066, 0.544853, 0.554029},
      {0.133743, 0.548535, 0.553541},
      {0.132444, 0.552216, 0.553018},
      {0.131172, 0.555899, 0.552459},
      {0.129933, 0.559582, 0.551864},
      {0.128729, 0.563265, 0.551229},
      {0.127568, 0.566949, 0.550556},
      {0.126453, 0.570633, 0.549841},
      {0.125394, 0.574318, 0.549086},
      {0.124395, 0.578002, 0.548287},
      {0.123463, 0.581687, 0.547445},
      {0.122606, 0.585371, 0.546557},
      {0.121831, 0.589055, 0.545623},
      {0.121148, 0.592739, 0.544641},
      {0.120565, 0.596422, 0.543611},
      {0.120092, 0.600104, 0.542530},
      {0.119738, 0.603785, 0.541400},
      {0.119512, 0.607464, 0.540218},
      {0.119423, 0.611141, 0.538982},
      {0.119483, 0.614817, 0.537692},
      {0.119699, 0.618490, 0.536347},
      {0.120081, 0.622161, 0.534946},
      {0.120638, 0.625828, 0.533488},
      {0.121380, 0.629492, 0.531973},
      {0.122312, 0.633153, 0.530398},
      {0.123444, 0.636809, 0.528763},
      {0.124780, 0.640461, 0.527068},
      {0.126326, 0.644107, 0.525311},
      {0.128087, 0.647749, 0.523491},
      {0.130067, 0.651384, 0.521608},
      {0.132268, 0.655014, 0.519661},
      {0.134692, 0.658636, 0.517649},
      {0.137339, 0.662252, 0.515571},
      {0.140210, 0.665859, 0.513427},
      {0.143303, 0.669459, 0.511215},
      {0.146616, 0.673050, 0.508936},
      {0.150148, 0.676631, 0.506589},
      {0.153894, 0.680203, 0.504172},
      {0.157851, 0.683765, 0.501686},
      {0.162016, 0.687316, 0.499129},
      {0.166383, 0.690856, 0.496502},
      {0.170948, 0.694384, 0.493803},
      {0.175707, 0.697900, 0.491033},
      {0.180653, 0.701402, 0.488189},
      {0.185783, 0.704891, 0.485273},
      {0.191090, 0.708366, 0.482284},
      {0.196571, 0.711827, 0.479221},
      {0.202219, 0.715272, 0.476084},
      {0.208030, 0.718701, 0.472873},
      {0.214000, 0.722114, 0.469588},
      {0.220124, 0.725509, 0.466226},
      {0.226397, 0.728888, 0.462789},
      {0.232815, 0.732247, 0.459277},
      {0.239374, 0.735588, 0.455688},
      {0.246070, 0.738910, 0.452024},
      {0.252899, 0.742211, 0.448284},
      {0.259857, 0.745492, 0.444467},
      {0.266941, 0.748751, 0.440573},
      {0.274149, 0.751988, 0.436601},
      {0.281477, 0.755203, 0.432552},
      {0.288921, 0.758394, 0.428426},
      {0.296479, 0.761561, 0.424223},
      {0.304148, 0.764704, 0.419943},
      {0.311925, 0.767822, 0.415586},
      {0.319809, 0.770914, 0.411152},
      {0.327796, 0.773980, 0.406640},
      {0.335885, 0.777018, 0.402049},
      {0.344074, 0.780029, 0.397381},
      {0.352360, 0.783011, 0.392636},
      {0.360741, 0.785964, 0.387814},
      {0.369214, 0.788888, 0.382914},
      {0.377779, 0.791781, 0.377939},
      {0.386433, 0.794644, 0.372886},
      {0.395174, 0.797475, 0.367757},
      {0.404001, 0.800275, 0.362552},
      {0.412913, 0.803041, 0.357269},
      {0.421908, 0.805774, 0.351910},
      {0.430983, 0.808473, 0.346476},
      {0.440137, 0.811138, 0.340967},
      {0.449368, 0.813768, 0.335384},
      {0.458674, 0.816363, 0.329727},
      {0.468053, 0.818921, 0.323998},
      {0.477504, 0.821444, 0.318195},
      {0.487026, 0.823929, 0.312321},
      {0.496615, 0.826376, 0.306377},
      {0.506271, 0.828786, 0.300362},
      {0.515992, 0.831158, 0.294279},
      {0.525776, 0.833491, 0.288127},
      {0.535621, 0.835785, 0.281908},
      {0.545524, 0.838039, 0.275626},
      {0.555484, 0.840254, 0.269281},
      {0.565498, 0.842430, 0.262877},
      {0.575563, 0.844566, 0.256415},
      {0.585678, 0.846661, 0.249897},
      {0.595839, 0.848717, 0.243329},
      {0.606045, 0.850733, 0.236712},
      {0.616293, 0.852709, 0.230052},
      {0.626579, 0.854645, 0.223353},
      {0.636902, 0.856542, 0.216620},
      {0.647257, 0.858400, 0.209861},
      {0.657642, 0.860219, 0.203082},
      {0.668054, 0.861999, 0.196293},
      {0.678489, 0.863742, 0.189503},
      {0.688944, 0.865448, 0.182725},
      {0.699415, 0.867117, 0.175971},
      {0.709898, 0.868751, 0.169257},
      {0.720391, 0.870350, 0.162603},
      {0.730889, 0.871916, 0.156029},
      {0.741388, 0.873449, 0.149561},
      {0.751884, 0.874951, 0.143228},
      {0.762373, 0.876424, 0.137064},
      {0.772852, 0.877868, 0.131109},
      {0.783315, 0.879285, 0.125405},
      {0.793760, 0.880678, 0.120005},
      {0.804182, 0.882046, 0.114965},
      {0.814576, 0.883393, 0.110347},
      {0.824940, 0.884720, 0.106217},
      {0.835270, 0.886029, 0.102646},
      {0.845561, 0.887322, 0.099702},
      {0.855810, 0.888601, 0.097452},
      {0.866013, 0.889868, 0.095953},
      {0.876168, 0.891125, 0.095250},
      {0.886271, 0.892374, 0.095374},
      {0.896320, 0.893616, 0.096335},
      {0.906311, 0.894855, 0.098125},
      {0.916242, 0.896091, 0.100717},
      {0.926106, 0.897330, 0.104071},
      {0.935904, 0.898570, 0.108131},
      {0.945636, 0.899815, 0.112838},
      {0.955300, 0.901065, 0.118128},
      {0.964894, 0.902323, 0.123941},
      {0.974417, 0.903590, 0.130215},
      {0.983868, 0.904867, 0.136897},
      {0.993248, 0.906157, 0.143936}};

// Update the heatmap texture
void update_heatmap(
    Texture2D heatmap_texture, 
    Image heatmap_image, 
    slice2d<float> animation_parameters,
    slice1d<delauney_tri> parameter_tris,
    slice2d<float> blend_matrix,
    nnet& blender,
    nnet_evaluation& blender_evaluation,
    int animation,
    int interpolation_method,
    bool project_target)
{
    assert(animation_parameters.cols == 2);
  
    array1d<float> blend_weights(animation_parameters.rows);
    array1d<float> query_parameters(2);
    array1d<float> target_parameters(2);
    array1d<float> query_distances(animation_parameters.rows);

    for (int x = 0; x < heatmap_image.width; x++)
    {
        for (int y = 0; y < heatmap_image.height; y++)
        {            
            target_parameters(0) = ((float)y + 0.5f) / (heatmap_image.height - 2);
            target_parameters(1) = ((float)x + 0.5f) / (heatmap_image.width - 2);
      
            if (interpolation_method == 0)
            {
                project_onto_convex_hull(
                    query_parameters, 
                    target_parameters, 
                    animation_parameters, 
                    parameter_tris);
              
                compute_blend_weights_triangulation(
                    blend_weights,
                    animation_parameters,
                    query_parameters,
                    parameter_tris);
            }
            else if (interpolation_method == 1)
            {
                if (project_target)
                {
                    project_onto_convex_hull(
                        query_parameters, 
                        target_parameters, 
                        animation_parameters, 
                        parameter_tris);
                }
                else
                {
                    query_parameters = target_parameters;
                }
              
                compute_query_distances(
                    query_distances,
                    animation_parameters,
                    query_parameters);
                
                compute_blend_weights(
                    blend_weights,
                    query_distances,
                    blend_matrix);
            }
            else if (interpolation_method == 2)
            {
                query_parameters = target_parameters;
              
                compute_query_distances(
                    query_distances,
                    animation_parameters,
                    query_parameters);
              
                blender_evaluation.layers[0] = query_distances;
                nnet_evaluate(blender_evaluation, blender); 
                blend_weights = blender_evaluation.layers.back();
            }
            else
            {
                assert(false);
            }

            clamp_normalize_blend_weights(blend_weights);
            
            unsigned char viridis_lookup = 255 * blend_weights(animation);
            
            ((unsigned char*)heatmap_image.data)[x * heatmap_image.height * 4 + y * 4 + 0] = 255 * viridis_data[viridis_lookup][0];
            ((unsigned char*)heatmap_image.data)[x * heatmap_image.height * 4 + y * 4 + 1] = 255 * viridis_data[viridis_lookup][1];
            ((unsigned char*)heatmap_image.data)[x * heatmap_image.height * 4 + y * 4 + 2] = 255 * viridis_data[viridis_lookup][2];
            ((unsigned char*)heatmap_image.data)[x * heatmap_image.height * 4 + y * 4 + 3] = 255;
        }
    }

    UpdateTexture(heatmap_texture, heatmap_image.data);
}

//--------------------------------------

static void update_trajectory(
    vec3& traj_pos,
    vec3& traj_vel,
    vec3& traj_acc,
    quat& traj_rot,
    vec3& traj_ang,
    vec3& traj_aac,
    float desired_fwd_vel,
    float desired_up_ang,
    float dt,
    float halflife)
{
    trajectory_spring_damper(
        traj_rot,
        traj_ang,
        traj_aac,
        vec3(0, desired_up_ang, 0),
        halflife,
        dt);
        
    trajectory_spring_damper(
        traj_pos,
        traj_vel,
        traj_acc,
        quat_mul_vec3(traj_rot, vec3(0, 0, desired_fwd_vel)),
        halflife,
        dt);
}

static void predict_trajectory(
    slice1d<vec3> traj_pos,
    slice1d<vec3> traj_vel,
    slice1d<vec3> traj_acc,
    slice1d<quat> traj_rot,
    slice1d<vec3> traj_ang,
    slice1d<vec3> traj_aac,
    float desired_fwd_vel,
    float desired_up_ang,
    float dt,
    float halflife)
{
    int ntraj = traj_pos.size;
  
    for (int i = 1; i < ntraj; i++)
    {
        traj_pos(i) = traj_pos(i - 1);
        traj_vel(i) = traj_vel(i - 1);
        traj_acc(i) = traj_acc(i - 1);
        traj_rot(i) = traj_rot(i - 1);
        traj_ang(i) = traj_ang(i - 1);
        traj_aac(i) = traj_aac(i - 1);
        
        update_trajectory(
            traj_pos(i),
            traj_vel(i),
            traj_acc(i),
            traj_rot(i),
            traj_ang(i),
            traj_aac(i),
            desired_fwd_vel,
            desired_up_ang,
            dt * 20,
            halflife);
    }
}

//--------------------------------------

void update_callback(void* args)
{
    ((std::function<void()>*)args)->operator()();
}

int main(void)
{
    // Init Window
    
    const int screen_width = 1280;
    const int screen_height = 720;
    // const int screen_width = 640;
    // const int screen_height = 480;
    
    SetConfigFlags(FLAG_VSYNC_HINT);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screen_width, screen_height, "raylib [blendspace]");
    SetTargetFPS(60);
    
    // Camera

    Camera3D camera = { 0 };
    camera.position = (Vector3){ 5.0f, 3.0f, 5.0f };
    camera.target = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    
    float camera_azimuth = 0.0f;
    float camera_altitude = 0.4f;
    float camera_distance = 4.0f;
    
    // Ground Plane
    
    Shader ground_plane_shader = LoadShader("./resources/checkerboard.vs", "./resources/checkerboard.fs");
    Mesh ground_plane_mesh = GenMeshPlane(20.0f, 20.0f, 10, 10);
    Model ground_plane_model = LoadModelFromMesh(ground_plane_mesh);
    ground_plane_model.materials[0].shader = ground_plane_shader;
    
    // Character
    
    character character_data;
    character_load(character_data, "./resources/character.bin");
    
    Shader character_shader = LoadShader("./resources/character.vs", "./resources/character.fs");
    Mesh character_mesh = make_character_mesh(character_data);
    Model character_model = LoadModelFromMesh(character_mesh);
    character_model.materials[0].shader = character_shader;
    
    // Load Animation Data
    
    const float dt = 1.0f / 60.0f;

    database db;
    database_load(db, "./resources/database.bin");
    
    // Trajectory
    
    bool use_trajectory = false;
    // bool use_trajectory = true;
    float trajectory_halflife = 0.2f;
    
    array1d<vec3> trajectory_pos(4);
    array1d<vec3> trajectory_vel(4);
    array1d<vec3> trajectory_acc(4);
    trajectory_pos.zero();
    trajectory_vel.zero();
    trajectory_acc.zero();
    
    array1d<quat> trajectory_rot(4);
    array1d<vec3> trajectory_ang(4);
    array1d<vec3> trajectory_aac(4);
    trajectory_rot.set(quat());
    trajectory_ang.zero();
    trajectory_aac.zero();
    
    // Blend Parameters
    
    array1d<float> blend_weights(db.nranges());
    blend_weights.zero();
    
    int nparams = use_trajectory ? 12 : 2;
    
    array1d<float> target_parameters(nparams);    
    target_parameters.set(0.5f);

    array1d<float> query_parameters(nparams);
    query_parameters = target_parameters;

    array1d<float> curr_parameters(nparams);
    curr_parameters.zero();
  
    array2d<float> animation_parameters(db.nranges(), nparams);
    
    if (use_trajectory)
    {
        compute_average_animation_parameters_trajectory(animation_parameters, db);
    }
    else
    {
        compute_average_animation_parameters_speed_turn(animation_parameters, db, dt);
    }
    
    FILE* f = fopen(use_trajectory ? 
        "./resources/parameters_traj.bin" : 
        "./resources/parameters_speedturn.bin", "wb");
    assert(f != NULL);
    array2d_write(animation_parameters, f);
    fclose(f);
    
    int interpolation_method = 0;
    // int interpolation_method = 1;
    // int interpolation_method = 2;
    
    // Delauney Triangulation
    
    array1d<delauney_tri> parameter_tris;
    bool display_triangulation = true;
    // bool display_triangulation = false;
    bool project_target = true;
    // bool project_target = false;
    bool display_current = true;
    // bool display_current = false;

    if (!use_trajectory)
    {
        fit_triangulation(parameter_tris, animation_parameters);
    }
    else
    {
        interpolation_method = 1;
        display_triangulation = false;
        project_target = false;
    }
    
    // Blend Matrix
    
    array2d<float> blend_matrix(db.nranges(), db.nranges());    
    fit_blend_matrix(blend_matrix, animation_parameters);
    
    array1d<float> query_distances(db.nranges());
    query_distances.zero();

    // Neural Network
    
    nnet blender;
    nnet_load(blender, use_trajectory ? 
        "./resources/network_traj.bin" : 
        "./resources/network_speedturn.bin");
    
    nnet_evaluation blender_evaluation;
    blender_evaluation.resize(blender);

    // Pose Data
    
    array2d<vec3> sample_positions(db.nranges(), db.nbones());
    array2d<quat> sample_rotations(db.nranges(), db.nbones());
    
    array1d<vec3> reference_positions(db.nbones());
    array1d<quat> reference_rotations(db.nbones());
    
    backward_kinematics_full(
        reference_positions,
        reference_rotations,
        character_data.bone_rest_positions,
        character_data.bone_rest_rotations,
        db.bone_parents);
    
    array1d<vec3> local_bone_positions(db.nbones());
    array1d<quat> local_bone_rotations(db.nbones());
    array1d<mat4> local_bone_accumulator(db.nbones());
    
    array1d<vec3> global_bone_positions(db.nbones());
    array1d<quat> global_bone_rotations(db.nbones());
    
    // Heatmap
    
    const int paramspace_width = 200;
    const int paramspace_height = 200;
    
    Image heatmap_image;
    heatmap_image.data = calloc(paramspace_width * paramspace_height, 4);
    heatmap_image.width = paramspace_width;
    heatmap_image.height = paramspace_height;
    heatmap_image.mipmaps = 1;
    heatmap_image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    
    Texture2D heatmap_texture = LoadTextureFromImage(heatmap_image);
    
    bool display_heatmap = true;
    // bool display_heatmap = false;
    int heatmap_index = 0;
    // int heatmap_index = 15;
    
    if (!use_trajectory)
    {
        update_heatmap(
            heatmap_texture, 
            heatmap_image, 
            animation_parameters,
            parameter_tris,
            blend_matrix,
            blender,
            blender_evaluation,
            heatmap_index,
            interpolation_method,
            project_target);
    }
    else
    {
        display_heatmap = false;
        heatmap_index = -1;
    }
    
    char heatmap_name_str[512];
    char heatmap_tmpstr[512];
    heatmap_name_str[0] = '\0';
    for (int i = 0; i < db.nranges(); i++)
    {
        if (i != db.nranges() - 1)
        {
            sprintf(heatmap_tmpstr, "Anim %i;", i);
        }
        else
        {
            sprintf(heatmap_tmpstr, "Anim %i", i);
        }
        strcat(heatmap_name_str, heatmap_tmpstr);
    }
    
    // bool dynamic_blendspace = true;
    bool dynamic_blendspace = false;
    int anim_selected = -1;
    
    // Playback State
    
    float normalized_time = 0.0f;

    // Go
    
    auto update_func = [&]()
    {        
        // Camera
    
        orbit_camera_update(
            camera, 
            camera_azimuth,
            camera_altitude,
            camera_distance,
            vec3(0, 1, 0),
            (IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(0)) ? GetMouseDelta().x : 0.0f,
            (IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(0)) ? GetMouseDelta().y : 0.0f,
            dt);
        
        // User Control
        
        float desired_fwd_vel = 0.0f;
        float desired_up_ang = 0.0f;
        
        vec2 gamepad = vec2(
            GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_X),
            GetGamepadAxisMovement(0, GAMEPAD_AXIS_LEFT_Y));
        
        float gamepad_length = length(gamepad);
        float deadzone = 0.1f;
        if (gamepad_length > deadzone)
        {
            vec2 gamepad_dir = normalize(gamepad);
            float gamepad_norm_length = powf((gamepad_length - deadzone) / (1.0 - deadzone), 2.0f);
            vec2 gamepad_norm = gamepad_norm_length * gamepad_dir;
            desired_fwd_vel = 4.0f * maxf(gamepad_norm.x, 0.0f);
            desired_up_ang = PIf * -gamepad_norm.y;
        }

        if (anim_selected != -1)
        {
            assert(!use_trajectory);
            fit_triangulation(parameter_tris, animation_parameters);
            fit_blend_matrix(blend_matrix, animation_parameters);
        }
        
        if (use_trajectory)
        {
            predict_trajectory(
                trajectory_pos,
                trajectory_vel,
                trajectory_acc,
                trajectory_rot,
                trajectory_ang,
                trajectory_aac,
                desired_fwd_vel,
                desired_up_ang,
                dt,
                trajectory_halflife);
                
            for (int i = 0; i < 3; i++)
            {
                target_parameters(i * 4 + 0) = trajectory_pos(i + 1).x;
                target_parameters(i * 4 + 1) = trajectory_pos(i + 1).z;
                target_parameters(i * 4 + 2) = quat_mul_vec3(trajectory_rot(i + 1), vec3(0,0,1)).x;
                target_parameters(i * 4 + 3) = quat_mul_vec3(trajectory_rot(i + 1), vec3(0,0,1)).z;
            }
        }
        
        // Recompute Blend Space
        
        if (dynamic_blendspace)
        {
            if (use_trajectory)
            {
                compute_frame_animation_parameters_trajectory(animation_parameters, db, normalized_time);                            
            }
            else
            {
                compute_frame_animation_parameters_speed_turn(animation_parameters, db, dt, normalized_time);              
                fit_triangulation(parameter_tris, animation_parameters);              
            }
            
            fit_blend_matrix(blend_matrix, animation_parameters);
        }
    
        // Compute Blend Weights
        
        if (interpolation_method == 0)
        {
            assert(!use_trajectory);
          
            project_onto_convex_hull(
                query_parameters, 
                target_parameters, 
                animation_parameters, 
                parameter_tris);
          
            compute_blend_weights_triangulation(
                blend_weights,
                animation_parameters,
                query_parameters,
                parameter_tris);
        }
        else if (interpolation_method == 1)
        {
            if (project_target && !use_trajectory)
            {
                project_onto_convex_hull(
                    query_parameters, 
                    target_parameters, 
                    animation_parameters, 
                    parameter_tris);
            }
            else
            {
                query_parameters = target_parameters;
            }
          
            compute_query_distances(
                query_distances,
                animation_parameters,
                query_parameters);
            
            compute_blend_weights(
                blend_weights,
                query_distances,
                blend_matrix);
        }
        else if (interpolation_method == 2)
        {
            query_parameters = target_parameters;

            compute_query_distances(
                query_distances,
                animation_parameters,
                query_parameters);
          
            blender_evaluation.layers[0] = query_distances;
            nnet_evaluate(blender_evaluation, blender); 
            blend_weights = blender_evaluation.layers.back();
        }
        else
        {
            assert(false);
        }
        
        clamp_normalize_blend_weights(blend_weights);
        
        // Sample Animations
        
        for (int i = 0; i < db.nranges(); i++)
        {
            database_sample(
                sample_positions(i),
                sample_rotations(i),
                db,
                i,
                normalized_time);
        }
        
        // Compute Weighted Average
        
        weighted_average_positions(
            local_bone_positions,
            sample_positions,
            blend_weights);
        
        weighted_average_rotations_ref(
            local_bone_rotations,
            reference_rotations,
            sample_rotations,
            blend_weights);
        
        // Accumulate Weighted Average of Normalized Delta Time
        
        float normalized_dt = 0.0f;
        for (int i = 0; i < db.nranges(); i++)
        {
            normalized_dt += blend_weights(i) * (1.0 / (db.range_stops(i) - db.range_starts(i)));
        }
        
        normalized_time = fmod(normalized_time + normalized_dt, 1.0f);
  
        // Done!
        
        local_bone_positions(0) = vec3();
        local_bone_rotations(0) = quat();
        
        forward_kinematics_full(
            global_bone_positions,
            global_bone_rotations,
            local_bone_positions,
            local_bone_rotations,
            db.bone_parents);
        
        // Update Trajectory
        
        if (use_trajectory)
        {
            update_trajectory(
                trajectory_pos(0),
                trajectory_vel(0),
                trajectory_acc(0),
                trajectory_rot(0),
                trajectory_ang(0),
                trajectory_aac(0),
                desired_fwd_vel,
                desired_up_ang,
                dt,
                trajectory_halflife);
            
            quat old_rot = trajectory_rot(0);
            
            trajectory_pos(0) = vec3();
            trajectory_rot(0) = quat();
            trajectory_vel(0) = quat_inv_mul_vec3(old_rot, trajectory_vel(0));
            trajectory_acc(0) = quat_inv_mul_vec3(old_rot, trajectory_acc(0));
            trajectory_ang(0) = quat_inv_mul_vec3(old_rot, trajectory_ang(0));
            trajectory_aac(0) = quat_inv_mul_vec3(old_rot, trajectory_aac(0));
        }
        
        // Render
        
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        BeginMode3D(camera);
        
        // Draw Character
        
        deform_character_mesh(
            character_mesh, 
            character_data, 
            global_bone_positions, 
            global_bone_rotations,
            db.bone_parents);
        
        DrawModel(character_model, (Vector3){0.0f, 0.0f, 0.0f}, 1.0f, RAYWHITE);
        
        // Draw Ground Plane
        
        DrawModel(ground_plane_model, (Vector3){0.0f, -0.01f, 0.0f}, 1.0f, WHITE);
        DrawGrid(20, 1.0f);
        draw_axis(vec3(), quat());
        
        // Draw Trajectory
        
        if (use_trajectory)
        {
            draw_trajectory_parameters(target_parameters, GREEN);
            
            if (display_current)
            {
                mat_transpose_mul_vec(curr_parameters, animation_parameters, blend_weights);
                draw_trajectory_parameters(curr_parameters, BLUE);
            }   
        }
        
        EndMode3D();

        // UI
        
        //---------
        
        if (!use_trajectory)
        {
            Vector2 mouse_point = GetMousePosition();
            
            if (IsKeyPressed(KEY_LEFT_ALT))
            {
                float mouse_x = (mouse_point.x - 20) / paramspace_width;
                float mouse_y = (mouse_point.y - 20) / paramspace_height;
              
                float best_dist = FLT_MAX;
                for (int i = 0; i < db.nranges(); i++)
                {
                    float dist = sqrtf(
                        squaref(mouse_x - animation_parameters(i, 0)) +
                        squaref(mouse_y - animation_parameters(i, 1)));
                        
                    if (dist < best_dist)
                    {
                        best_dist = dist;
                        anim_selected = i;
                    }
                }
            }
            
            if (IsKeyDown(KEY_LEFT_ALT) && anim_selected != -1 &&
                mouse_point.x >= 20 && mouse_point.x <= paramspace_width + 20 &&
                mouse_point.y >= 20 && mouse_point.y <= paramspace_height + 20)
            {
                if (anim_selected != -1)
                {
                    animation_parameters(anim_selected, 0) = (mouse_point.x - 20) / paramspace_width;
                    animation_parameters(anim_selected, 1) = (mouse_point.y - 20) / paramspace_height;
                }
            }
            
            if (!IsKeyDown(KEY_LEFT_ALT))
            {
                anim_selected = -1;
            }
            
            if (IsMouseButtonDown(0) && 
                mouse_point.x >= 20 && mouse_point.x <= paramspace_width + 20 &&
                mouse_point.y >= 20 && mouse_point.y <= paramspace_height + 20)
            {
                target_parameters(0) = (mouse_point.x - 20) / paramspace_width;
                target_parameters(1) = (mouse_point.y - 20) / paramspace_height;
            }
            
            GuiGroupBox((Rectangle){ 10, 10, paramspace_width + 20, paramspace_height + 20 }, "parameter space");
            
            if (!dynamic_blendspace && display_heatmap)
            {
                DrawTexture(heatmap_texture, 20, 20, WHITE);
                DrawRectangleLines(20, 20, paramspace_width, paramspace_height, RAYWHITE);
            }
            else
            {
                GuiDrawRectangle((Rectangle){ 20, 20, paramspace_width, paramspace_height }, 1, RAYWHITE, DARKGRAY);
            }
            
            if (display_triangulation)
            {
                for (int i = 0; i < parameter_tris.size; i++)
                {
                    int p0_x = 20 + (int)(animation_parameters(parameter_tris(i).p1, 0) * paramspace_width);
                    int p0_y = 20 + (int)(animation_parameters(parameter_tris(i).p1, 1) * paramspace_height);
                  
                    int p1_x = 20 + (int)(animation_parameters(parameter_tris(i).p2, 0) * paramspace_width);
                    int p1_y = 20 + (int)(animation_parameters(parameter_tris(i).p2, 1) * paramspace_height);
                  
                    int p2_x = 20 + (int)(animation_parameters(parameter_tris(i).p3, 0) * paramspace_width);
                    int p2_y = 20 + (int)(animation_parameters(parameter_tris(i).p3, 1) * paramspace_height);
                    
                    Color col = RAYWHITE;
                    col.a = 128;
                    
                    DrawLine(p0_x, p0_y, p1_x, p1_y, col);
                    DrawLine(p1_x, p1_y, p2_x, p2_y, col);
                    DrawLine(p2_x, p2_y, p0_x, p0_y, col);
                }
            }
            
            { 
                for (int i = 0; i < db.nranges(); i++)
                {
                    float weight = powf(blend_weights(i), 1.0f / 2.0f);
                    Color col = {
                        255, 
                        (unsigned char)((1.0f - weight) * 255), 
                        (unsigned char)((1.0f - weight) * 255), 
                        255 };
                  
                    DrawCircle(
                        20 + (int)(animation_parameters(i, 0) * paramspace_width), 
                        20 + (int)(animation_parameters(i, 1) * paramspace_height), 
                        3, col);
                    DrawCircleLines(
                        20 + (int)(animation_parameters(i, 0) * paramspace_width), 
                        20 + (int)(animation_parameters(i, 1) * paramspace_height), 
                        3, 
                        LIGHTGRAY);
                }
                
                DrawCircle(
                    20 + (int)(target_parameters(0) * paramspace_width), 
                    20 + (int)(target_parameters(1) * paramspace_height), 
                    3, 
                    GREEN);
                DrawCircleLines(
                    20 + (int)(target_parameters(0) * paramspace_width), 
                    20 + (int)(target_parameters(1) * paramspace_height), 
                    3, 
                    LIGHTGRAY);
                    
                if (display_current)
                {
                    mat_transpose_mul_vec(curr_parameters, animation_parameters, blend_weights);
                 
                    DrawCircle(
                        20 + (int)(curr_parameters(0) * paramspace_width), 
                        20 + (int)(curr_parameters(1) * paramspace_height), 
                        3, 
                        BLUE);
                    DrawCircleLines(
                        20 + (int)(curr_parameters(0) * paramspace_width), 
                        20 + (int)(curr_parameters(1) * paramspace_height), 
                        3, 
                        LIGHTGRAY);
                }
            }
            
        }
        
        for (int i = 0; i < db.nranges(); i++)
        {
            float offset_x = 20 + (float)(i / 9) * 70;
            float offset_y = (use_trajectory ? 20 : paramspace_height + 40) + (float)(i % 9) * 20;
            GuiLabel((Rectangle){ offset_x, offset_y, 100, 20 }, TextFormat("Anim %2i:", i));
            
            float weight = powf(blend_weights(i), 1.0f / 2.0f);
            Color col = {
                255, 
                (unsigned char)((1.0f - weight) * 255), 
                (unsigned char)((1.0f - weight) * 255), 
                255 };
          
            DrawCircle(offset_x + 50, offset_y + 10, 4, col);
            DrawCircleLines(offset_x + 50, offset_y + 10, 4, LIGHTGRAY);
        }
        
        float ui_ctrl_height = 10;
        float ui_ctrl_left = screen_width - 260;
        
        GuiGroupBox((Rectangle){ ui_ctrl_left, ui_ctrl_height, 250, 80 }, "controls");
        
        GuiLabel((Rectangle){ ui_ctrl_left + 20, ui_ctrl_height + 10, 200, 20 }, "Ctrl + Left Click - Move Camera");
        GuiLabel((Rectangle){ ui_ctrl_left + 20, ui_ctrl_height + 30, 200, 20 }, "Mouse Wheel - Zoom");
        GuiLabel((Rectangle){ ui_ctrl_left + 20, ui_ctrl_height + 50, 200, 20 }, "Right Click - Move target");
        
        float ui_settings_height = 120;
        // float ui_settings_height = 10;
        float ui_settings_left = screen_width - 260;
        
        GuiGroupBox((Rectangle){ ui_settings_left, ui_settings_height, 250, 200 }, "settings");
        // GuiGroupBox((Rectangle){ ui_settings_left, ui_settings_height, 250, 80 }, "settings");
        // GuiGroupBox((Rectangle){ ui_settings_left, ui_settings_height, 250, 150 }, "settings");

        GuiCheckBox(
            (Rectangle){ ui_settings_left + 20, ui_settings_height + 20, 20, 20 }, 
            "display triangulation", 
            &display_triangulation);
        
        GuiLabel((Rectangle){ ui_settings_left + 20, ui_settings_height + 50, 100, 20 }, "method");
        
        int prev_method = interpolation_method;
        if (dynamic_blendspace)
        {
            GuiComboBox((Rectangle){ ui_settings_left + 80, ui_settings_height + 50, 140, 20 }, "Triangulation;Interpolation", &interpolation_method);
        }
        else
        {
            GuiComboBox((Rectangle){ ui_settings_left + 80, ui_settings_height + 50, 140, 20 }, "Triangulation;Interpolation;Network", &interpolation_method);          
            // GuiComboBox((Rectangle){ ui_settings_left + 80, ui_settings_height + 50, 140, 20 }, "Triangulation;Interpolation", &interpolation_method);          
        }
        
        if (use_trajectory && interpolation_method == 0)
        {
            interpolation_method = 1;
        }
        
        GuiLabel((Rectangle){ ui_settings_left + 20, ui_settings_height + 80, 100, 20 }, "heatmap");

        int heatmap_index_prev = heatmap_index;
        GuiComboBox((Rectangle){ ui_settings_left + 80, ui_settings_height + 80, 140, 20 }, heatmap_name_str, &heatmap_index);
        
        GuiCheckBox(
            (Rectangle){ ui_settings_left + 20, ui_settings_height + 110, 20, 20 }, 
            "display heatmap", 
            &display_heatmap);
        
        GuiCheckBox(
            (Rectangle){ ui_settings_left + 20, ui_settings_height + 140, 20, 20 }, 
            "display result", 
            &display_current);
            
        bool project_target_prev = project_target;
        GuiCheckBox(
            (Rectangle){ ui_settings_left + 20, ui_settings_height + 170, 20, 20 }, 
            "project convex hull", 
            &project_target);
            
        if (dynamic_blendspace)
        {
            display_heatmap = false;
            if (interpolation_method == 2) { interpolation_method = 0; }
        }
        
        if (!use_trajectory && (
            interpolation_method != prev_method || 
            heatmap_index_prev != heatmap_index ||
            project_target_prev != project_target))
        // if (!use_trajectory && (
            // interpolation_method != prev_method || 
            // heatmap_index_prev != heatmap_index))
        {
            update_heatmap(
                heatmap_texture, 
                heatmap_image, 
                animation_parameters,
                parameter_tris,
                blend_matrix,
                blender,
                blender_evaluation,
                heatmap_index,
                interpolation_method,
                project_target);
        }
        
        // Done
        
        EndDrawing();
    };
    
#if defined(PLATFORM_WEB)
    std::function<void()> u{update_func};
    emscripten_set_main_loop_arg(update_callback, &u, 0, 1);
#else
    while (!WindowShouldClose())
    {
        update_func();
    }
#endif

    // Unload stuff and finish
    UnloadImage(heatmap_image);
    UnloadTexture(heatmap_texture);
    UnloadModel(character_model);
    UnloadModel(ground_plane_model);
    UnloadShader(character_shader);
    UnloadShader(ground_plane_shader);

    CloseWindow();

    return 0;
}