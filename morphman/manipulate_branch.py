##   Copyright (c) Aslak W. Bergersen, Henrik A. Kjeldsberg. All rights reserved.
##   See LICENSE file for details.

##      This software is distributed WITHOUT ANY WARRANTY; without even 
##      the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
##      PURPOSE.  See the above copyright notices for more information.

from argparse import ArgumentParser, RawDescriptionHelpFormatter

from scipy.signal import argrelextrema

# Local import
from morphman.common.argparse_common import *
from morphman.common.surface_operations import *

MAX_BIFURCATION_LENGTH = 25


def manipulate_branch(input_filepath, output_filepath, smooth, smooth_factor, poly_ball_size, no_smooth,
                      no_smooth_point, resampling_step, polar_angle, azimuth_angle, remove_branch,
                      branch_to_manipulate_number, branch_location, translation_method):
    """
    Primary script for moving or removing a selected branch of any blood vessel.
    Relies on a surface geometry of a 3D blood vessel network.

    Defines in- and out-put files, computes voronoi diagram, and
    moves voronoi diagram according to user selected points.
    Used selects two points defining a diverging branch, and the
    one point defining the new location on the surface.
    Proceedes with either removal of a single branch,
    or traslationg and rotation of a single branch.

    Args:
        remove_branch (bool: If true, removes selected branch completely.
        input_filepath (str): Path to input surface.
        output_filepath (str): Path to output the manipulated surface.
        smooth (bool): Smooth the Voronoi diagram.
        smooth_factor (float): Smoothing factor used for Voronoi diagram smoothing.
        poly_ball_size (list): Resolution of polyballs used to create surface.
        no_smooth (bool): True of part of the model is not to be smoothed.
        no_smooth_point (ndarray): Point which is untouched by smoothing.
        resampling_step (float): Resampling length when resampling centerline.
        azimuth_angle (float): Angle to rotate around new surface normal vector. Default is no rotation.
        polar_angle (float): Angle to rotate around the axis spanned by cross product of new normal and frenet normal.
        branch_to_manipulate_number (int): Number of branch to manipulate, ordered from up- to down-stream.
        branch_location (ndarray): Point where branch to manipulate will be moved. Closest point on surface is selected.
        translation_method (str): Method for translating the branch to be manipulated.
    """

    # Input and output filenames
    base_path = get_path_names(input_filepath)
    centerlines_path = base_path + "_centerline.vtp"
    unprepared_output_filepath = base_path + "_unprepared_output.vtp"

    # Clean and capp / uncapp surface
    surface, capped_surface = prepare_surface(base_path, input_filepath)
    inlet, outlets = compute_centers(surface, base_path)

    print("-- Computing centerlines and Voronoi diagram")
    centerlines_complete, voronoi, pole_ids = compute_centerlines(inlet, outlets, centerlines_path,
                                                                  capped_surface, resampling=resampling_step,
                                                                  smooth=False, base_path=base_path)

    if smooth:
        print("-- Smoothing Voronoi diagram")
        voronoi = prepare_voronoi_diagram(capped_surface, centerlines_complete, base_path,
                                          smooth, smooth_factor, no_smooth,
                                          no_smooth_point, voronoi, pole_ids, resampling_step)

    # Select diverging branch from Brancher and compare with diverging branch found manually
    centerlines_branched = vmtk_compute_branch_extractor(centerlines_complete)
    branches_complete = get_all_branches(centerlines_branched, get_sorted_lines(centerlines_complete))

    # Select branch to manipulate
    if branch_to_manipulate_number is not None:
        check_branch_number(branch_to_manipulate_number, branches_complete)
        branch_to_manipulate = branches_complete[branch_to_manipulate_number - 1]
    else:
        branch_to_manipulate = pick_branch(capped_surface, branches_complete)

    if not remove_branch:
        # Get new position of branch on model surface
        if translation_method == 'manual':
            new_branch_pos_id, new_branch_pos = pick_new_branch_position(capped_surface)
        elif translation_method == 'commandline':
            new_branch_pos_id, new_branch_pos = get_new_branch_position(branch_location, capped_surface)
        elif translation_method == 'no_translation':
            new_branch_pos_id, new_branch_pos = None, None

    branch_to_manipulate_end_point = get_end_point(branch_to_manipulate)
    centerlines = filter_centerlines(centerlines_complete, branch_to_manipulate_end_point)

    print("-- Clipping Voronoi diagram")
    voronoi_branch, voronoi_remaining = get_split_voronoi_diagram(voronoi, [branch_to_manipulate, centerlines])
    voronoi_branch, voronoi_remaining_2 = filter_voronoi(voronoi_branch, branch_to_manipulate)
    voronoi_remaining = vtk_merge_polydata([voronoi_remaining, voronoi_remaining_2])

    if remove_branch:
        print("-- Removing branch")
        detach_branch(voronoi_remaining, centerlines, poly_ball_size,
                      unprepared_output_filepath, surface, output_filepath, centerlines_complete, base_path)
    else:
        move_and_rotate_branch(polar_angle, azimuth_angle, capped_surface, centerlines, centerlines_complete,
                               branch_to_manipulate, new_branch_pos, new_branch_pos_id, output_filepath,
                               poly_ball_size, surface, unprepared_output_filepath, voronoi_branch, voronoi_remaining,
                               base_path, translation_method)


def detach_branch(voronoi_remaining, centerlines, poly_ball_size, unprepared_output_filepath, surface, output_filepath,
                  centerlines_complete, base_path):
    """
    Reconstructs the Voronoi diagram, creating a new surface,
    without the selected branch, utimatley removing it.
    Proceeds by computing the corresponding centerline in the new
    geometries, used for postprocessing, geometric analysis and
    meshing.

    Args:
        centerlines (vtkPolyData): Relevant centerlines in new geometry
        centerlines_complete (vtkPolyData): Complete set of centerlines
        surface (vtkPolyData): Surface model
        unprepared_output_filepath (string): Path to save unprepared surface
        voronoi_remaining (vtkPolyData): Voronoi diagram of remaining surface model
        base_path (string): Path to save location
        output_filepath (str): Path to output the manipulated surface.
        poly_ball_size (list): Resolution of polyballs used to create surface.
    """
    new_centerlines_path = base_path + "_centerline_removed_branch.vtp"
    new_voronoi_path = base_path + "_voronoi_removed_branch.vtp"

    # Create new voronoi diagram and new centerlines
    write_polydata(voronoi_remaining, new_voronoi_path)

    write_polydata(centerlines, new_centerlines_path)

    new_surface = create_new_surface(voronoi_remaining, poly_ball_size=poly_ball_size)
    write_polydata(new_surface, unprepared_output_filepath)

    print("-- Smoothing, cleaning, and checking surface")
    new_surface = prepare_output_surface(new_surface, surface,
                                         centerlines, output_filepath,
                                         test_merge=False, changed=False,
                                         old_centerline=centerlines_complete)

    print("\n-- Writing new surface to {}.".format(output_filepath))
    write_polydata(new_surface, output_filepath)


def move_and_rotate_branch(polar_angle, azimuth_angle, capped_surface, centerlines, centerlines_complete,
                           diverging_centerline_branch, new_branch_pos, new_branch_pos_id, output_filepath,
                           poly_ball_size, surface, unprepared_output_filepath, voronoi_branch, voronoi_remaining,
                           base_path, method):
    """
    Moves, and/or performs rotation of the voronoi diagram, then
    reconstructs the Voronoi diagram, creating a new surface.
    Proceeds by computing the corresponding centerline in the new
    geometries, used for postprocessing, geometric analysis and
    meshing.

    Args:
        method (str): Translation method for selecting new branch position.
        capped_surface (vtkPolyData): Capped surface model
        centerlines (vtkPolyData): Relevant centerlines in new geometry
        centerlines_complete (vtkPolyData): Complete set of centerlines
        diverging_centerline_branch (vtkPolyData): Diverging centerline
        new_branch_pos (vtkPoint): Point where branch is moved
        new_branch_pos_id (int): ID of point where branch is moved
        surface (vtkPolyData): Surface model
        unprepared_output_filepath (string): Path to save unprepared surface
        voronoi_branch (vtkPolyData): Voronoi diagram of branch
        voronoi_remaining (vtkPolyData): Voronoi diagram of remaining surface model
        base_path (string): Path to save location
        output_filepath (str): Path to output the manipulated surface.
        poly_ball_size (list): Resolution of polyballs used to create surface.
        azimuth_angle (float): Angle to rotate around new surface normal vector. Default is no rotation.
        polar_angle (float): Angle to rotate around the axis spanned by cross product of new normal and frenet normal.
    """
    if method in ['commandline', 'manual']:
        if polar_angle != 0 or azimuth_angle != 0:
            description = "moved_and_rotated"
        else:
            description = "moved"
    elif method == 'no_translation':
        description = 'rotated'

    new_centerlines_path = base_path + "_centerline_{}.vtp".format(description)
    new_voronoi_path = base_path + "_voronoi_{}.vtp".format(description)

    # Get surface normals and setup of paramters
    old_normal = get_estimated_surface_normal(diverging_centerline_branch)
    manipulated_centerline = diverging_centerline_branch
    manipulated_voronoi = voronoi_branch
    if method == 'no_translation':
        new_normal = old_normal
        origin = np.asarray(diverging_centerline_branch.GetPoint(0))

    # Perform manipulation
    if method in ['commandline', 'manual']:
        print("-- Translating branch")
        new_normal = get_exact_surface_normal(capped_surface, new_branch_pos_id)
        manipulated_voronoi, manipulated_centerline, origin = move_branch(centerlines, manipulated_centerline,
                                                                          new_branch_pos, old_normal, new_normal,
                                                                          manipulated_voronoi)
    if azimuth_angle != 0:
        print("-- Rotating branch")
        manipulated_voronoi, manipulated_centerline = rotate_branch(azimuth_angle, manipulated_centerline,
                                                                    manipulated_voronoi, origin, new_normal)
    if polar_angle != 0:
        print("-- Rotating branch")
        axis_of_rotation = get_axis_of_rotation(manipulated_centerline, new_normal)
        manipulated_voronoi, manipulated_centerline = rotate_branch(polar_angle, manipulated_centerline,
                                                                    manipulated_voronoi, origin, axis_of_rotation)

    # Create new voronoi diagram and new centerlines
    new_voronoi = vtk_merge_polydata([voronoi_remaining, manipulated_voronoi])
    write_polydata(new_voronoi, new_voronoi_path)

    new_centerlines = vtk_merge_polydata([centerlines, manipulated_centerline])
    write_polydata(new_centerlines, new_centerlines_path)

    new_surface = create_new_surface(new_voronoi, poly_ball_size=poly_ball_size)
    write_polydata(new_surface, unprepared_output_filepath)

    print("-- Smoothing, cleaning, and checking surface")
    new_surface = prepare_output_surface(new_surface, surface,
                                         new_centerlines, output_filepath,
                                         test_merge=False, changed=True,
                                         old_centerline=centerlines_complete)

    print("\n-- Writing new surface to {}.".format(output_filepath))
    write_polydata(new_surface, output_filepath)


def get_axis_of_rotation(centerline, normal_vector):
    """
    Compute axis of rotation for Azimuthal rotation

    Args:
        centerline (vtkPolyData): Centerline of branch to manipulate
        normal_vector (vtkPolyData): Surface normal vector at rotation origin

    Returns:
        ndarray: Axis of rotation vector
    """
    centerline = vmtk_compute_geometric_features(centerline, True)
    frenet_normal = centerline.GetPointData().GetArray("FrenetNormal").GetTuple3(0)
    axis_of_rotation = np.cross(normal_vector, frenet_normal)

    return axis_of_rotation


def move_branch(centerlines, diverging_centerline_branch, new_branch_pos,
                old_normal, new_normal, voronoi_branch):
    """
    Translates and rotates the voronoi diagram and centerline from one point
    on the surface model to a selected point, defined by
    new branch position.

    Args:
        new_normal (ndarray): New surface normal
        old_normal (ndarray): Old surface normal
        centerlines (vtkPolyData): Relevant centerlines in new geometry
        diverging_centerline_branch (vtkPolyData): Diverging centerline
        new_branch_pos (vtkPoint): Point where branch is moved
        voronoi_branch (vtkPolyData): Voronoi diagram of branch

    Returns:
        vtkPolyData: Translated Voronoi diagram
        vtkPolyData: Translated centerline
        ndarray: Origin of new position
    """
    # Define rotation between surface normal vectors
    u, surface_normals_angle = get_rotation_axis_and_angle(new_normal, old_normal)
    R = get_rotation_matrix(u, surface_normals_angle)

    # Define translation parameters
    dx, origin = get_translation_parameters(centerlines, diverging_centerline_branch, new_branch_pos)

    # Move branch centerline and voronoi diagram
    moved_voronoi = manipulate_voronoi_branch(voronoi_branch, dx, R, origin, None, None, 0.0, 'translate')
    moved_centerline = manipulate_centerline_branch(diverging_centerline_branch, origin, R, dx, None, 0.0, 'translate')

    return moved_voronoi, moved_centerline, origin


def rotate_branch(angle, diverging_centerline_branch, voronoi_branch, origin, axis_of_rotation):
    """
    Perform rotation of the voronoi diagram and the centerline
    around a given axis defined by the input normal vector.

    Args:
        angle (float): Angle to rotate the branch, in radians
        origin (ndarray): Origin of the centerline
        axis_of_rotation (ndarray): Vector defing axis to rotate around
        diverging_centerline_branch (vtkPolyData): Diverging centerline
        voronoi_branch (vtkPolyData): Voronoi diagram of branch

    Returns:
        vtkPolyData: Rotated Voronoi diagram
        vtkPolyData: Rotated centerline
    """
    # Define rotation around new surface normal
    R = get_rotation_matrix(-axis_of_rotation, angle)

    # Move branch centerline and voronoi diagram
    rotated_voronoi = manipulate_voronoi_branch(voronoi_branch, 0.0, R, origin, diverging_centerline_branch,
                                                axis_of_rotation, angle, 'rotate')
    rotated_centerline = manipulate_centerline_branch(diverging_centerline_branch, origin, R, 0.0, axis_of_rotation,
                                                      angle, 'rotate')

    return rotated_voronoi, rotated_centerline


def check_branch_number(branch_to_manipulate_number, branches_complete):
    """
    Check if branch number provided by user is larger than number of centerlines.
    Raises RuntimeError if number exceeds limit.

    Args:
        branch_to_manipulate_number (int): Input number, supplied by user
        branches_complete (list): All centerline branches
    """
    num_lines = len(branches_complete)
    if branch_to_manipulate_number > num_lines:
        raise RuntimeError("\nERROR: Branch number cannot exceed number of valid branches." +
                           " Number of selectable branches for this model is {}.".format(num_lines))


def get_new_branch_position(branch_location, capped_surface):
    """
    Get point on surface closest to branch_location. Returns
    both the point on the surface and it's ID.

    Args:
        branch_location (ndarray): User selected point.
        capped_surface (vtkPolyData): Input surface

    Returns:
        new_branch_pos_id (int): Point closest to branch location ID on surface.
        new_branch_pos (ndarray): Point closest to branch location on surface.
    """
    surface_locator = get_vtk_point_locator(capped_surface)
    new_branch_pos_id = surface_locator.FindClosestPoint(branch_location)
    new_branch_pos = capped_surface.GetPoint(new_branch_pos_id)

    return new_branch_pos_id, new_branch_pos


def pick_new_branch_position(capped_surface):
    """
    Select (by manually chosing) where branch is translated to on the surface.

    Args:
        capped_surface (vtkPolyData): Input surface

    Returns:
        new_branch_pos_id (int): Point closest to branch location ID on surface.
        new_branch_pos (ndarray): Point closest to branch location on surface.
    """
    print("\nPlease select new point to move branch in the render window.")
    seed_selector = vmtkPickPointSeedSelector()
    seed_selector.SetSurface(capped_surface)
    seed_selector.text = "Press space to select the point where you want to move the branch to, 'u' to undo.\n"
    seed_selector.Execute()
    new_branch_pos_id = seed_selector.GetTargetSeedIds().GetId(0)
    new_branch_pos = capped_surface.GetPoint(new_branch_pos_id)

    print("-- New branch location is:", np.asarray(new_branch_pos))

    return new_branch_pos_id, new_branch_pos


def pick_branch(capped_surface, centerlines):
    """
    Select (by manually chosing) which branch is translated to on the surface.

    Args:
        capped_surface (vtkPolyData): Input surface
        centerlines (vtkPolyData): Set of valid centerline branches throughout model

    Returns:
          branch_to_manipulate (vtkPolyData): Branch selected to be manipulated
    """
    print("\nPlease select the branch you with to move in the render window.")
    # Select point on surface
    seed_selector = vmtkPickPointSeedSelector()
    seed_selector.SetSurface(capped_surface)
    seed_selector.text = "Press space to select the branch you want to move, 'u' to undo.\n"
    seed_selector.Execute()
    branch_surface_point_id = seed_selector.GetTargetSeedIds().GetId(0)
    surface_point = capped_surface.GetPoint(branch_surface_point_id)

    # Find closest centerlines
    closest_dist = 1e9
    valid_branches_to_manipulate = []
    for line_to_compare in centerlines:
        locator = get_vtk_point_locator(line_to_compare)
        closest_point_id = locator.FindClosestPoint(surface_point)
        closest_point = line_to_compare.GetPoint(closest_point_id)
        dist = get_distance(closest_point, surface_point)
        if dist <= closest_dist:
            closest_dist = dist
            valid_branches_to_manipulate.append(line_to_compare)

    # Select shortest centerline of valid branches
    branch_to_manipulate = valid_branches_to_manipulate[0]
    for branch in valid_branches_to_manipulate[1:]:
        branch_length = len(get_curvilinear_coordinate(branch))
        current_length = len(get_curvilinear_coordinate(branch_to_manipulate))
        if branch_length < current_length:
            branch_to_manipulate = branch

    return branch_to_manipulate


def filter_voronoi(voronoi, diverging_centerline_branch):
    """
    Filter away voronoi points too far away from relevant branch

    Args:
        voronoi (vtkPolyData): Voronoi diagram to be filtered
        diverging_centerline_branch (vtkPolyData): Relevant centerlines of the branch

    Returns:
        vtkPolyData: Voronoi diagram, diverging part
    Returns:
        vtkPolyData: Voronoi diagram, remaining part
    """
    misr = get_point_data_array(radiusArrayName, diverging_centerline_branch)
    locator = get_vtk_point_locator(diverging_centerline_branch)

    n = voronoi.GetNumberOfPoints()
    diverging_voronoi = vtk.vtkPolyData()
    remaining_voronoi = vtk.vtkPolyData()
    div_cell_array = vtk.vtkCellArray()
    rem_cell_array = vtk.vtkCellArray()
    div_points = vtk.vtkPoints()
    div_radius = np.zeros(n)
    rem_points = vtk.vtkPoints()
    rem_radius = np.zeros(n)

    radius_array_data = voronoi.GetPointData().GetArray(radiusArrayName).GetTuple1

    div_count = 0
    rem_count = 0
    for i in range(n):
        point = voronoi.GetPoint(i)
        closest_point_id = locator.FindClosestPoint(point)
        closest_point = diverging_centerline_branch.GetPoint(closest_point_id)
        if get_distance(closest_point, point) > max(misr):
            rem_count = set_voronoi_point_data(i, point, radius_array_data, rem_cell_array, rem_count, rem_points,
                                               rem_radius)
        else:
            div_count = set_voronoi_point_data(i, point, radius_array_data, div_cell_array, div_count, div_points,
                                               div_radius)

    set_voronoi_data(div_cell_array, div_count, div_points, div_radius, diverging_voronoi)
    set_voronoi_data(rem_cell_array, rem_count, rem_points, rem_radius, remaining_voronoi)

    return diverging_voronoi, remaining_voronoi


def set_voronoi_point_data(i, point, radius_array_data, cell_array, count, points, radius):
    """
    Set point data to a single Voronoi diagram point

    Args:
        i (int): Counter
        point (vtkPoint): Single point of Voronoi diagram
        radius_array_data (ndarray): MISR Radius array
        cell_array (vtkCellArray): Cell array
        count (int): Specific counter
        points (vtkPoints): Point array
        radius (ndarray):  Radius array

    Returns:
        int: Incremented counter
    """
    points.InsertNextPoint(point)
    cell_array.InsertNextCell(1)
    cell_array.InsertCellPoint(count)
    value = radius_array_data(i)
    radius[count] = value
    count += 1

    return count


def set_voronoi_data(cell_array, count, points, radius, voronoi):
    """
    Apply points and data to voronoi object

    Args:
        cell_array (vtkCellArray): Cell array
        count (int): Specific counter
        points (vtkPoints): Point array
        radius (ndarray):  Radius array
        voronoi (vtkPolyData): Voronoi diagram to be filtered
    """
    radius_array = get_vtk_array(radiusArrayName, 1, count)
    for i in range(count):
        radius_array.SetTuple(i, [float(radius[i])])

    voronoi.SetPoints(points)
    voronoi.SetVerts(cell_array)
    voronoi.GetPointData().AddArray(radius_array)


def get_translation_parameters(centerlines, diverging_centerline_branch, new_branch_pos):
    """
    Get distance to translate branch, and the new position location for branch

    Args:
        centerlines (vtkPolyData): Relevant centerlines
        diverging_centerline_branch (vtkPolyData): Diverging centerline
        new_branch_pos (vtkPoint):  Position where branch is moved.

    Returns:
        dx (float): Distance to translate branch
    Returns:
        origin (ndarray): Adjusted origin / position of new branch position
    """
    locator = get_vtk_point_locator(centerlines)
    new_branch_pos_on_cl = centerlines.GetPoint(locator.FindClosestPoint(new_branch_pos))
    adjusted_branch_pos = (np.asarray(new_branch_pos) - np.asarray(new_branch_pos_on_cl)) * 0.8 + np.asarray(
        new_branch_pos_on_cl)
    dx = np.asarray(adjusted_branch_pos) - np.asarray(diverging_centerline_branch.GetPoint(0))
    origin = np.asarray(adjusted_branch_pos)

    return dx, origin


def get_exact_surface_normal(capped_surface, new_branch_pos_id):
    """
    Compute normal out of surface at given point.

    Args:
        capped_surface (vtkPolyData): Capped surface model
        new_branch_pos_id (int): ID of point where branch is moved, on surface

    Returns:
        new_normal (ndarray): Normal vector out of surface
    """
    capped_surface_with_normals = vmtk_compute_surface_normals(capped_surface)

    new_normal = capped_surface_with_normals.GetPointData().GetNormals().GetTuple(new_branch_pos_id)
    new_normal /= la.norm(new_normal)

    return new_normal


def get_estimated_surface_normal(centerline):
    """
    Estimate the surface normal at initial diverging centerline branch.

    Args:
        centerline (vtkPolyData): Diverging centerline to be moved.

    Returns:
        normal_vector (ndarray): Estimated normal vector at diverging centerline
    """
    centerline = vmtk_compute_geometric_features(centerline, True)
    curvature = get_point_data_array("Curvature", centerline)
    first_local_maxima_id = argrelextrema(curvature, np.greater)[0][0]

    # TODO: Generalize choice of end point factor
    factor = 0.4
    start_point = 0
    end_point = int(first_local_maxima_id * factor)
    normal_vector = np.asarray(centerline.GetPoint(end_point)) - np.asarray(
        centerline.GetPoint(start_point))

    normal_vector /= la.norm(normal_vector)

    return normal_vector


def manipulate_centerline_branch(centerline_branch, origin, R, dx, normal, angle, manipulation):
    """
    Depending on manipulation method, either translates or
    rotates the selected branch, represented as a centerline.

    Args:
        manipulation (str): Type of manipulation, either 'rotate' or 'translate'
        centerline_branch (vtkPolyData): Centerline branch to be manipulated
        dx (float): Distance to translate branch
        R (ndarray): Rotation matrix, rotation from old to new surface normal or around new surface normal
        origin (ndarray): Adjusted origin / position of new branch position
        angle (float): Angle to rotate in radians
        normal (ndarray): Normal vector at manipulation location

    Returns:
        centerline (vtkPolyData): Manipulated centerline
    """
    # Locator to find closest point on centerline
    number_of_points = centerline_branch.GetNumberOfPoints()

    centerline = vtk.vtkPolyData()
    centerline_points = vtk.vtkPoints()
    centerline_cell_array = vtk.vtkCellArray()
    radius_array = get_vtk_array(radiusArrayName, 1, number_of_points)

    centerline_cell_array.InsertNextCell(number_of_points)
    radius_array_data = centerline_branch.GetPointData().GetArray(radiusArrayName).GetTuple1

    # Transition from base to branch for 30% of branch creating a smoother rotation transition
    base_end = int(number_of_points * 0.3)

    for p in range(centerline_branch.GetNumberOfPoints()):
        point = centerline_branch.GetPoint(p)

        if manipulation == 'translate':
            # Translate and rotate branch upright
            point = np.asarray(point) + dx
            point = np.dot(R, point - origin) + origin

        elif manipulation == 'rotate':
            # Rotate branch around axis
            if p <= base_end:
                transition_angle = angle * (p / base_end) ** 0.2
                transition_rotation_matrix = get_rotation_matrix(-normal, transition_angle)
                point = np.dot(transition_rotation_matrix, point - origin) + origin
            else:
                point = np.dot(R, point - origin) + origin

        centerline_points.InsertNextPoint(point)
        radius_array.SetTuple1(p, radius_array_data(p))
        centerline_cell_array.InsertCellPoint(p)

    centerline.SetPoints(centerline_points)
    centerline.SetLines(centerline_cell_array)
    centerline.GetPointData().AddArray(radius_array)

    return centerline


def get_rotation_axis_and_angle(new_normal, old_normal):
    """
    Compute axis vector and angle between normal vectors (input)

    Args:
        old_normal (ndarray): Normal vector at initial position
        new_normal (ndarray): Normal vector at new position

    Returns:
        u (ndarray): Normal vector corresponding to rotation axis
    Returns:
        angle (float): Angle between normal vectors
    """
    z = np.asarray(old_normal)
    z_prime = np.asarray(new_normal)

    u = np.cross(z, z_prime)
    u /= la.norm(u)

    angle = np.arccos(z.dot(z_prime) / (la.norm(z) * la.norm(z_prime)))

    return u, angle


def manipulate_voronoi_branch(voronoi, dx, R, origin, centerline, normal, angle, manipulation):
    """
    Depending on manipulation method, either translates or
    rotates the selected branch, represented as a Voronoi diagram.

    Args:
        manipulation (str): Type of manipulation, either 'rotate' or 'translate'
        voronoi (vtkPolyData): Voronoi diagram of surface
        dx (float): Distance to translate branch
        R (ndarray): Rotation matrix, rotation from old to new surface normal or around new surface normal
        origin (ndarray): Adjusted origin / position of new branch position
        angle (float): Angle to rotate in radians
        normal (ndarray): Normal vector at manipulation location
        centerline (vtkPolyData): Centerline of branch to be manipulated

    Returns:
        new_voronoi (vtkPolyData): Manipulated Voronoi diagram
    """
    # Voronoi diagram
    n = voronoi.GetNumberOfPoints()
    new_voronoi = vtk.vtkPolyData()
    voronoi_points = vtk.vtkPoints()
    cell_array = vtk.vtkCellArray()
    radius_array = get_vtk_array(radiusArrayName, 1, n)
    misr_array = voronoi.GetPointData().GetArray(radiusArrayName).GetTuple1

    if manipulation == 'rotate':
        # Centerline locator
        m = centerline.GetNumberOfPoints()
        centerline_locator = get_vtk_point_locator(centerline)

        # Transition from base to branch for 30% of branch creating a smoother rotation transition
        base_end = int(m * 0.3)

    # Iterate through Voronoi diagram and manipulate
    for i in range(n):
        point = voronoi.GetPoint(i)
        misr = misr_array(i)

        if manipulation == 'translate':
            # Translate voronoi points
            point = np.asarray(point) + dx
            point = np.dot(R, point - origin) + origin

        elif manipulation == 'rotate':
            # Rotate voronoi points
            cl_id = centerline_locator.FindClosestPoint(point)
            if cl_id <= base_end:
                transition_angle = angle * (cl_id / base_end) ** 0.2
                transition_rotation_matrix = get_rotation_matrix(-normal, transition_angle)
                point = np.dot(transition_rotation_matrix, point - origin) + origin
            else:
                point = np.dot(R, point - origin) + origin

        # Change radius
        radius_array.SetTuple1(i, misr)
        voronoi_points.InsertNextPoint(point)
        cell_array.InsertNextCell(1)
        cell_array.InsertCellPoint(i)

    new_voronoi.SetPoints(voronoi_points)
    new_voronoi.SetVerts(cell_array)
    new_voronoi.GetPointData().AddArray(radius_array)

    return new_voronoi


def get_all_branches(branched_centerlines, centerlines):
    """
    Extract and combine all branches of the surface model.
    Removes first part of centerlines after combining,
    excluding the bifurcating part.

    Args:
        branched_centerlines (vktPolyData): Complete centerlines divided into branches
        centerlines (list, ndarray): List containing sorted centerlines

    Returns:
        list: All possible branches in the geometry
    """
    branch_segments = extract_all_branches(branched_centerlines, centerlines)
    all_branches_combined = combine_all_branches(branch_segments)
    all_branches = remove_bifurcation_lines(all_branches_combined)
    sorted_branches = sort_branches(all_branches, centerlines)

    return sorted_branches


def extract_all_branches(branched_centerlines, centerlines):
    """
    Find all unique branches of a network of centerlines.

    Args:
        branched_centerlines (vtkPolyData): Segmented centerlines from vmtkBranchExtractor
        centerlines (list): List of all centerlines in geometry

    Returns:
        list: Contains all unique branches
    """
    n = branched_centerlines.GetNumberOfLines()
    branch_segments = []
    branches_tmp = []
    add_branch = False
    for i in range(n):
        line = extract_single_line(branched_centerlines, i)
        line_length = len(get_curvilinear_coordinate(line))

        if line_length < MAX_BIFURCATION_LENGTH:
            branches_tmp.append(line)
            add_branch = True

        if add_branch:
            for j, bif in enumerate(branches_tmp):
                branches_tmp[j] = vtk_merge_polydata([branches_tmp[j], line])

        for cl in centerlines:
            if get_end_point(cl) == get_end_point(line):
                add_branch = False
                branch_segments.append(branches_tmp)
                branches_tmp = []

    return branch_segments


def combine_all_branches(branch_segments):
    """
    Combine branches which share a common bifurcation point

    Args:
        branch_segments (list): Contains all unique branching segments

    Returns:
        list: Contains combined branches which share bifurcation point
    """
    all_branches_combined = []
    for i, segment in enumerate(branch_segments):
        for j, branch in enumerate(segment):
            if branch is None:
                continue

            if branch.GetNumberOfLines() > 2:
                bif_point = extract_single_line(branch, 1).GetPoint(0)

                for k, segment_to_search in enumerate(branch_segments):
                    # Skip current segment
                    if k == i:
                        continue

                    for l, branch_to_check in enumerate(segment_to_search):
                        if branch_to_check is None:
                            continue

                        bif_point_to_check = extract_single_line(branch_to_check, 1).GetPoint(0)
                        if bif_point == bif_point_to_check:
                            branch = vtk_merge_polydata([branch, branch_to_check])
                            branch_segments[k][l] = None

                all_branches_combined.append(branch)
            else:
                all_branches_combined.append(branch)

            branch_segments[i][j] = None

    return all_branches_combined


def remove_bifurcation_lines(all_branches_combined):
    """
    Removed first part of each centerline, containing
    tiny bifurcation segment of centerline.

    Args:
        all_branches_combined (list): Combined unique branches

    Returns:
        list: Contains all unique branches in centerline network
    """
    # Remove bifurcation part of branch
    all_branches = []
    for branch in all_branches_combined:
        branch_without_bifurcation = [extract_single_line(branch, i) for i in range(2, branch.GetNumberOfLines())]
        all_branches.append(vtk_merge_polydata(branch_without_bifurcation))

    return all_branches


def sort_branches(branches, centerlines):
    """
    Sort combined branches based on sorted centerlines

    Args:
        branches (list): Contains branches to be sorted
        centerlines (list): Contains sorted centerlines

    Returns:
        list: Sorted branches
    """

    sorted_branches_dict = {k: [] for k in range(len(centerlines))}

    # Compare branches end point with sorted centerlines
    for i, branch in enumerate(branches):
        branch_end_point = get_end_point(branch)
        for j, line in enumerate(centerlines):
            line_end_point = get_end_point(line)
            if branch_end_point == line_end_point:
                sorted_branches_dict[j].append(branch)
                continue

    # Collect and merge all branches
    sorted_branches = []
    for _, lines in sorted_branches_dict.items():
        for line in lines:
            sorted_branches.append(line)

    return sorted_branches


def read_command_line_branch(input_path=None, output_path=None):
    """
    Read arguments from commandline and return all values in a dictionary.
    If input_path and output_path are not None, then do not parse command line, but
    only return default values.

    Args:
        input_path (str): Input file path, positional argument with default None.
        output_path (str): Output file path, positional argument with default None.
    """
    # Description of the script
    description = "Moves a selected part of a tubular geometry, " + \
                  "in two (horizontal, vertical) geometry-specific directions. " + \
                  "Magnitude of movement is defined by the parameters alpha and beta" + \
                  "Primary script used for application in blood vessels."

    parser = ArgumentParser(description=description, formatter_class=RawDescriptionHelpFormatter)

    # Add common arguments
    required = not (input_path is not None and output_path is not None)
    add_common_arguments(parser, required=required)

    parser.add_argument("-tm", "--translation-method", type=str, default="manual",
                        choices=["manual", "commandline", "no_translation"],
                        help="Defines the method of translation of the branch to be manipulated." +
                             " The parameter provides three options: 'manual', 'commandline' and 'no_translation'. In" +
                             " 'manual' the user will be provided with a visualization of the input surface, and " +
                             "asked to provide the new position of the branch on the surface model." +
                             " If 'commandline' is provided, then '--branch-location'" +
                             " is expected to be provided. Selecting 'no_translation' will " +
                             "result in no translation; any manipulation performed on the " +
                             "branch will happen at the branch's current position. ")
    parser.add_argument('-bl', "--branch-location", nargs="+", type=float, default=None, metavar="branch_location",
                        help="If this parameter is provided, the branch to be manipulated will be moved to the point "
                             "on the surface closest to this point. Example providing the point (1, 5, -1):" +
                             " --branch-loc 1 5 -1")

    # Arguments for rotation
    parser.add_argument('-aa', '--azimuth-angle', type=float, default=0,
                        help="The manipulated branch is rotated an angle 'aa' around the old or new" +
                             " surface normal vector. 'aa' is assumed to be in degrees," +
                             " and not radians. Default is no rotation.", metavar="surface_normal_axis_angle")
    parser.add_argument('-pa', '--polar-angle', type=float, default=0,
                        help="The manipulated branch is rotated an angle 'pa' around the" +
                             " surface tangent vector, constructed by the cross product of the surface normal vector" +
                             " and the Frenet normal vector. 'pa' is assumed to be in degrees," +
                             " and not radians. Default is no rotation.", metavar="surface_tangent_axis_angle")

    # Argument for selecting branch
    parser.add_argument('-bn', '--branch-number', type=int, default=None,
                        help="The number corresponding the branch to manipulate. " +
                             "The branches are ordered from 1 to N, " +
                             "from upstream to downstream, relative to the inlet. " +
                             "If not selected, the user must manually select the branch "
                             "to manipulate. ", metavar="branch_number")

    # Argument for selecting branch
    parser.add_argument('-rb', '--remove-branch', type=str2bool, default=False,
                        help="If true, will remove selected branch and perform no manipulation")

    # Parse paths to get default values
    if required:
        args = parser.parse_args()
    else:
        args = parser.parse_args(["-i" + input_path, "-o" + output_path])

    if not 0 <= args.azimuth_angle <= 360:
        raise ArgumentTypeError("The azimuth angle is limited to be within [0, 360] degrees, cannot have value" +
                                " {}".format(args.azimuth_angle))

    if not -180 <= args.polar_angle <= 180:
        raise ArgumentTypeError("The polar angle is limited to be within [-180, 180] degrees, cannot have value" +
                                " {}".format(args.polar_angle))

    # Convert from deg to rad and invert rotation if exceeding 180 degrees
    polar_angle_to_radians = args.polar_angle * math.pi / 180

    azimuth_angle_to_radians = args.azimuth_angle * math.pi / 180
    if azimuth_angle_to_radians > np.pi:
        azimuth_angle_to_radians -= 2 * np.pi

    if args.branch_number is not None:
        if args.branch_number < 1:
            raise ValueError("ERROR: Branch number cannot be 0 or negative. Please select a positive number")

    if args.no_smooth_point is not None and len(args.no_smooth_point):
        if len(args.no_smooth_point) % 3 != 0:
            raise ValueError("ERROR: Please provide the no smooth point(s) as a multiple of 3")

    return dict(input_filepath=args.ifile, smooth=args.smooth, smooth_factor=args.smooth_factor,
                output_filepath=args.ofile, poly_ball_size=args.poly_ball_size, no_smooth=args.no_smooth,
                no_smooth_point=args.no_smooth_point, resampling_step=args.resampling_step,
                polar_angle=polar_angle_to_radians, azimuth_angle=azimuth_angle_to_radians,
                remove_branch=args.remove_branch, branch_to_manipulate_number=args.branch_number,
                branch_location=args.branch_location, translation_method=args.translation_method)


def main_branch():
    manipulate_branch(**read_command_line_branch())


if __name__ == "__main__":
    manipulate_branch(**read_command_line_branch())
