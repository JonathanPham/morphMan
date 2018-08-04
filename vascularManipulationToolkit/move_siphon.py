from argparse import ArgumentParser
from os import path, listdir
import sys
import numpy.linalg as la

# Local import
from common import *
from patchandinterpolatecenterlines import *
from clipvoronoidiagram import *
from paralleltransportvoronoidiagram import *
from moveandmanipulatetools import *

def read_command_line():
    """
    Read arguments from commandline
    """
    parser = ArgumentParser()

    parser.add_argument('-d', '--dir_path', type=str, default=".",
                        help="Path to the folder with all the cases")
    parser.add_argument('--case','-c', type=str, default=None, help="Choose case")
    parser.add_argument('-s', '--smooth', type=bool, default=True,
                        help="If the original voronoi diagram (surface) should be" + \
                        "smoothed before it is manipulated", metavar="smooth")
    parser.add_argument("-a", "--alpha", type=float, default=0.0,
                        help="Compression factor in vertical direction, ranging from -1.0 to 1.0")
    parser.add_argument("-b", "--beta", type=float, default=0.0,
                        help="Compression factor in horizontal direction, ranging from -1.0 to 1.0")

    args = parser.parse_args()

    return args.smooth, args.dir_path, args.case, args.alpha, args.beta


def move_vessel(dirpath, smooth, name, point_path, alpha=0.0, beta=0.0):
    """
    Primary script for moving a selected part of any blood vessel.
    Relies on an input centerline, a surface geometry of a 3D blood vessel network,
    and the magnitude of movement (alpha, beta) in horizontal and vertical direction respectively.

    Defines in- and out-put files, computes voronoi diagram, and
    moves voronoi diagram in the horizontal direction based on the
    definitions in "Investigating the Interaction Between Morphology
    of the Anterior Bend and Aneurysm Initiation" (2018).
    Procedes by computing the corresponding centerline in the new
    geometries, used for postprocessing, geometric analysis and
    meshing.

    Continuation in the move_vessel_vertically-method for vertical movement.

    Args:
        dirpath (str): Directory where case is located.
        smooth (bool): Adjusts smoothing of the voronoi diagram.
        name (str): Directory where surface models are located.
        point_path (str): Location of boundary points of the vessel to be manipulated.
        alpha (float): Extension / Compression factor in vertical direction.
        beta (float): Extension / Compression factor in horizontal direction.
    """

    # Input filenames
    model_path = path.join(dirpath, name, "model.vtp")

    # Output names
    model_smoothed_path = path.join(dirpath, name, "model_smoothed.vtp")
    model_new_surface = path.join(dirpath, name, "new_model_alpha_%s_beta_%s.vtp" % (alpha,beta))
    model_new_surface_tmp = path.join(dirpath, name, "new_model_alpha_%s_beta_%s_tmp.vtp" % (alpha, beta))

    # Centerlines
    centerline_complete_path = path.join(dirpath, name, "centerline_complete.vtp")
    centerline_clipped_path = path.join(dirpath, name,  "centerline_clipped.vtp")
    centerline_clipped_part_path = path.join(dirpath, name,  "centerline_clipped_part.vtp")
    new_centerlines_path = path.join(dirpath, name, "new_centerlines_alpha_%s_beta_%s.vtp" % (alpha, beta))
    new_centerlines_path_tmp = path.join(dirpath, name, "new_centerlines_alpha_%s_beta_%s_tmp.vtp" % (alpha, beta))

    # Voronoi diagrams
    voronoi_path = path.join(dirpath, name, "model_voronoi.vtp")
    voronoi_smoothed_path = path.join(dirpath, name,  "model_voronoi_smoothed.vtp")
    voronoi_remaining_path = path.join(dirpath, name, "model_voronoi_remaining.vtp")
    voronoi_siphon_path = path.join(dirpath, name, "model_voronoi_siphon.vtp")

    # Extract Clipping points
    clipping_points = get_clipping_points(dirpath, point_path)

    # Read and check model
    if not path.exists(model_path):
        RuntimeError("The given directory: %s did not contain the file: model.vtp" % dirpath)

    # Clean surface
    surface = read_polydata(model_path)
    surface = surface_cleaner(surface)
    surface = triangulate_surface(surface)

    #Get a capped and uncapped version of the surface
    open_surface = surface
    capped_surface = capp_surface(surface)

    # Get inlet and outlets
    inlet, outlets = get_centers(open_surface, dirpath)

    # Compute all centerlines
    centerlines_complete = vmtk_compute_centerlines(inlet, outlets,
                                               centerline_complete_path,
                                               capped_surface, resampling=0.1)
    centerlines_in_order = sort_centerlines(centerlines_complete)

    # Set clipping points in order and make VTK objects
    line = extract_single_line(centerlines_in_order, 0)
    p1, p2, ID1, ID2, vtk_clipping_points, clipping_points = get_vtk_clipping_points(line, clipping_points)

    print("Compute Voronoi diagram")
    voronoi = make_voronoi_diagram(surface, voronoi_path)
    if not path.exists(voronoi_smoothed_path) and smooth:
        voronoi_smoothed = SmoothClippedVoronoiDiagram(voronoi, centerlines_complete, 0.25)
        write_polydata(voronoi_smoothed, voronoi_smoothed_path)
        surface_smoothed = create_new_surface(voronoi_smoothed)
        write_polydata(surface_smoothed, model_smoothed_path)

    # Check if case includs the opthalmic artery
    eye, clip_ID, centerlines_in_order, eyeline = find_ophthalmic_artery(centerlines_in_order, clipping_points)
    if eye == True:
        print("Clipping opthamlic artery")
        patch_eye = clip_eyeline(eyeline, clipping_points[0], clip_ID)

    # Clip centerline
    print("Clipping centerlines.")
    centerline_remaining = CreateParentArteryPatches(centerlines_in_order, vtk_clipping_points, siphon=True)
    centerline_siphon = extract_single_line(centerlines_in_order, 0, startID=ID1, endID=ID2)
    if eye == True:
        eyeline_end = extract_single_line(patch_eye,1)
        centerline_siphon = merge_data([centerline_siphon, eyeline_end])

    write_polydata(centerline_remaining, centerline_clipped_path)
    write_polydata(centerline_siphon, centerline_clipped_part_path)

    # Clip Voronoi diagram into
    # siphon and remaining part of geometry
    print("Clipping Voronoi diagrams")
    if not path.exists(voronoi_siphon_path):
        masked_voronoi_clip  = MaskVoronoiDiagram(voronoi, centerline_siphon)
        voronoi_siphon = ExtractMaskedVoronoiPoints(voronoi, masked_voronoi_clip)
        write_polydata(voronoi_siphon, voronoi_siphon_path)
    else:
        voronoi_siphon = read_polydata(voronoi_siphon_path)

    if not path.exists(voronoi_remaining_path):
        masked_voronoi  = MaskVoronoiDiagram(voronoi, centerline_remaining)
        voronoi_remaining = ExtractMaskedVoronoiPoints(voronoi, masked_voronoi)
        write_polydata(voronoi_remaining, voronoi_remaining_path)
    else:
        voronoi_remaining = read_polydata(voronoi_remaining_path)

    # Extract translation vectors
    print("Computing translation directions.")
    direction = "horizont"
    middle_points, middleIds = get_spline_points(line, beta, direction, vtk_clipping_points)
    dx_p1 = middle_points[0] - p1
    dx_p2 = middle_points[-1] - p2
    middle_points = middle_points[1:-1]

    # Move centerline manually for updating inlet
    # and outlet positions used to compute
    # new centerlines
    print("Adjusting line manually")
    line1 = extract_single_line(centerline_remaining, 0)
    lines = []
    newpoints = []

    from IPython import embed
    embed()


    n = centerline_remaining.GetNumberOfCells()
    for i in range(1,n):
        lines.append(extract_single_line(centerline_remaining, i))

    centerline_remaining_start = move_line_horizontally(line1, ID1, ID2, dx_p1, clip=False, side="left") if beta != 0.0 else line1
    centerline_remaining_ends = []
    for line_ in lines:
        if beta != 0.0:
            centerline_remaining_end = move_line_horizontally(line_, ID1, ID2, dx_p1, clip=False, side="right")
            newpoints.append(centerline_remaining_end.GetPoint(centerline_remaining_end.GetNumberOfCells() - 1))
        else:
            centerline_remaining_end = line_
            newpoints.append(centerline_remaining_end.GetPoint(centerline_remaining_end.GetNumberOfPoints() - 1))
        centerline_remaining_ends.append(centerline_remaining_end)


    centerline_siphon_new = move_line_horizontally(centerline_siphon, ID1, ID2, dx_p1, clip=True, eye=eye) if beta != 0.0 else centerline_siphon

    if eye:
        newpoints.append(centerline_siphon_new.GetPoint(centerline_siphon_new.GetNumberOfCells() - 1))

    newpoints.append(centerline_remaining_start.GetPoint(0))


    if beta != 0.0:
        # Move anterior bend horizontally.
        # Iterate over points P from Voronoi diagram,
        # and move them
        print("Adjusting Voronoi diagram")
        voronoi_remaining = move_voronoi_horizontally(dx_p1, dx_p2, voronoi_remaining, centerline_remaining, ID1,ID2, clip_ID, clip=False)
        voronoi_siphon = move_voronoi_horizontally(dx_p1, dx_p2, voronoi_siphon, centerline_siphon, ID1,ID2, clip_ID, clip=True, eye=eye)

        newVoronoi = merge_data([voronoi_remaining,voronoi_siphon])
        new_surface = create_new_surface(newVoronoi)
        write_polydata(new_surface, model_new_surface_tmp)

        print "Creating new_centerline_complete.vtp of horizontally moved model"
        new_centerline = make_centerline(model_new_surface_tmp, new_centerlines_path_tmp, smooth=False, resampling=False, newpoints=newpoints, recompute=True, store_points=False)
    else:
        print("No horizontal movement. Initiating vertical movement")
        new_centerline = centerlines_complete
        write_polydata(new_centerline, new_centerlines_path_tmp)

    if alpha != 0.0:
        # Vertical movement
        print("Moving geometry vertically")
        move_vessel_vertically(dirpath, name, newpoints, alpha, voronoi_remaining, voronoi_siphon, new_centerline, eye, vtk_clipping_points)
    else:
        write_polydata(new_centerline, new_centerlines_path)
        write_polydata(new_surface, model_new_surface)


def move_vessel_vertically(dirpath, name, oldpoints, alpha,  voronoi_remaining, voronoi_siphon, centerline, eye, vtk_clipping_points):
    """
    Secondary script used for vertical displacement of
    the blood vessel. Moves the input voronoi diagram and
    centerline in the vertical direction.

    Args:
        dirpath (str): Directory where case is located.
        name (str): Directory where surface models are located.
        oldpoints (list): Points defining the inlet/outlets of the geometry.
        centerline (vtkPolyData): Centerline through the geometry.
        alpha (float): Extension / Compression factor in vertical direction.
        voronoi (vtkPolyData): Complete Voronoi diagram.
        voronoi_remaining (vtkPolyData): Voronoi diagram excluding siphon.
        voronoi_siphon (vtkPolyData): Voronoi diagram representing siphon.
        centerline (vtkPolyData): Centerline through the geometry.
        eye (bool): Determines if opthalmic artery is present.
        vtk_clipping_points (vtkPoints): Points defining the siphon to be manipulated.
    """
    # Filenames
    new_centerlines_path = path.join(dirpath, name, "new_centerlines_alpha_%s_beta_%s.vtp" % (alpha, beta))
    model_new_surface = path.join(dirpath, name, "new_model_alpha_%s_beta_%s.vtp" % (alpha,beta))

    # Set clipping points in order and make VTK objects
    p1      = vtk_clipping_points.GetPoint(0)
    p2      = vtk_clipping_points.GetPoint(1)
    centerline_clippedipping_points = [p1,p2]
    line = extract_single_line(centerline, 0)
    p1, p2, ID1, ID2, vtk_clipping_points, clipping_points = get_vtk_clipping_points(line, centerline_clippedipping_points)

    # Sort centerlines
    centerlines_in_order = sort_centerlines(centerline)

    # Special cases including the ophthalmic artery
    clip_ID = None
    if eye == True:
        eye, clip_ID, centerlines_in_order, eyeline = find_ophthalmic_artery(centerlines_in_order, clipping_points)

    if eye == True:
        print("Clipping opthamlic artery")
        patch_eye = clip_eyeline(eyeline, clipping_points[0], clip_ID)


    # Get clipped curve
    print("Clipping centerlines.")
    centerline_remaining = CreateParentArteryPatches(centerlines_in_order, vtk_clipping_points, siphon=True)
    centerline_siphon = extract_single_line(centerlines_in_order, 0, startID=ID1, endID=ID2)

    if eye == True:
        eyeline_end = extract_single_line(patch_eye,1)
        centerline_siphon = merge_data([centerline_siphon, eyeline_end])

    # Find ID of middle pooint:
    print("Finding points to spline through.")
    line = extract_single_line(centerlines_in_order, 0)
    loc = get_locator(line)
    ID1 = loc.FindClosestPoint(p1)

    direction = "vertical"
    middle_points, middleIds, dx = get_spline_points(line, alpha, direction, vtk_clipping_points)

    # Iterate over points P from Voronoi diagram,
    # and move them
    print("Adjust voronoi diagram")
    voronoi_siphon =  move_voronoi_vertically(voronoi_siphon,centerline_siphon, ID1,  clip_ID, dx,  eye)
    newVoronoi = merge_data([voronoi_remaining, voronoi_siphon])

    # Move centerline manually for postprocessing
    print("Adjusting line manually")
    if eye:
        newpoints = oldpoints[:-2] + [oldpoints[-1]]
    else:
        newpoints = oldpoints
    centerline_siphon_new = move_line_vertically(centerline_siphon, dx, ID1, clip_ID, eye)

    if eye:
        newpoints.insert(0, centerline_siphon_new.GetPoint(centerline_siphon_new.GetNumberOfCells() - 1))

    # Write a new surface from the new voronoi diagram
    print("Writing new surface")
    new_surface = create_new_surface(newVoronoi)
    write_polydata(new_surface, model_new_surface)

    print "Creating new_centerline_complete.vtp of vertically moved model"
    new_centerline = make_centerline(model_new_surface, new_centerlines_path, smooth=False, resampling=False, newpoints=newpoints, recompute=True)
    write_polydata(new_centerline, new_centerlines_path)


def move_voronoi_horizontally(dx_p1, dx_p2, voronoi_clipped, centerline_clipped, ID1,ID2, clip_ID, clip=False, eye=False):
    """
    Iterate through voronoi diagram and move based on a profile
    for horizontal movement. Includes special treatment of
    opthalmic artery if present.

    Args:
        dx_p1 (ndarray): Direction to move upstream.
        dx_p2 (ndarray): Direction to move downstream.
        voronoi_clipped (vtkPolyData): Voronoi diagram to be moved.
        centerline_clipped (vtkPolyData): Centerline corresponding voronoi diagram.
        ID1 (int): Index of first clipping point.
        ID2 (int): Index of second clipping point.
        clip_ID (int): Index where opthamlic artery is located (if present)
        clip (bool): Determines which part of geometry is being moved, True if siphon.
        eye (bool): Determines presence of opthamlic artery.

    Returns:
        newDataSet (vtkPolyData): Manipulated Voronoi diagram.
    """

    centerline_loc = get_locator(centerline_clipped)
    newDataSet = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()

    if clip == True:
        # Find boundaries
        idmid_0 = int((ID1 + ID2)/2.)
        ID1_0 = ID1
        ID1 = 0
        if eye == True:
            l1 = extract_single_line(centerline_clipped, 0)
            ID2 = len(get_curvilinear_coordinate(l1))
        else:
            ID2 = len(get_curvilinear_coordinate(centerline_clipped))
        idmid = int( (ID1 + ID2)/2.)

        # Manpipulation of voronoi diagram..
        if eye == True:
            # ..with opthalmic artery
            for p in range(voronoi_clipped.GetNumberOfPoints()):
                cl_id = centerline_loc.FindClosestPoint(voronoi_clipped.GetPoint(p))

                if cl_id < idmid:
                    dist = dx_p1 * (idmid**2-cl_id**2) / (idmid**2-ID1**2)
                else:
                    if cl_id <= (ID2-1):
                        dist = -dx_p1 * (cl_id-idmid)**(0.5) / (ID2-idmid)**(0.5)
                    else:
                        # Move opthalmic artery based on its discovery ID
                        if clip_ID < idmid_0:
                            cl_id =  clip_ID - ID1_0
                            dist = dx_p1 * (idmid**2-cl_id**2) / (idmid**2-ID1**2)
                        else:
                            cl_id = clip_ID - ID1_0
                            dist = -dx_p1 * (cl_id-idmid)**(0.5) / (ID2-idmid)**(0.5)

                points.InsertNextPoint(np.asarray(voronoi_clipped.GetPoint(p)) + dist)
                verts.InsertNextCell(1)
                verts.InsertCellPoint(p)

        else:
            # ..without opthalmic artery
            for p in range(voronoi_clipped.GetNumberOfPoints()):
                cl_id = centerline_loc.FindClosestPoint(voronoi_clipped.GetPoint(p))

                if cl_id < idmid:
                    dist = dx_p1 * (idmid**2-cl_id**2) / (idmid**2-ID1**2)
                else:
                    dist = -dx_p1 * (cl_id-idmid)**(0.5) / (ID2-idmid)**(0.5)

                points.InsertNextPoint(np.asarray(voronoi_clipped.GetPoint(p)) + dist)
                verts.InsertNextCell(1)
                verts.InsertCellPoint(p)

    else:
        # Move reamining part of the voronoi diagram
        # representing the geometry excluding the siphon to be moved
        for p in range(voronoi_clipped.GetNumberOfPoints()+1):
            cl_id = centerline_loc.FindClosestPoint(voronoi_clipped.GetPoint(p))

            if cl_id <= ID1:
                dist = dx_p1
            else:
                dist =  -dx_p1

            points.InsertNextPoint(np.asarray(voronoi_clipped.GetPoint(p)) + dist)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(p)

    # Insert points and create new voronoi diagram
    newDataSet.SetPoints(points)
    newDataSet.SetVerts(verts)
    newDataSet.GetPointData().AddArray(voronoi_clipped.GetPointData().GetArray(radiusArrayName))
    return newDataSet


def move_voronoi_vertically(voronoi_clipped, centerline_clipped, ID1_0, clip_ID, dx,eye=False):
    """
    Iterate through voronoi diagram and move based on a profile
    for vertical movement. Includes special treatment of
    opthalmic artery if present.

    Args:
        voronoi_clipped (vtkPolyData): Voronoi diagram to be moved.
        centerline_clipped (vtkPolyData): Centerline corresponding voronoi diagram.
        ID1 (int): Index of first clipping point.
        clip_ID (int): Index where opthamlic artery is located (if present)
        dx (ndarray): Direction to move.
        eye (bool): Determines presence of opthamlic artery.

    Returns:
        newDataSet (vtkPolyData): Manipulated Voronoi diagram.
    """

    centerline_loc = get_locator(centerline_clipped)
    newDataSet = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()

    # Manpipulation of voronoi diagram..
    if eye == True:
        # ..with opthalmic artery
        ID1 = I1 = 0
        l1 = extract_single_line(centerline_clipped, 0)
        ID2 = len(get_curvilinear_coordinate(l1))
        I2 = ID2 - 1
        IDmid = int( (ID1 + ID2)/2.)
        for p in range(voronoi_clipped.GetNumberOfPoints()):
            cl_id = centerline_loc.FindClosestPoint(voronoi_clipped.GetPoint(p))

            if cl_id <= I2:
                dist = 4 * dx * (cl_id - I1)*(I2 - cl_id) / ( I2 - I1)**2
            else:
                cl_id = clip_ID - ID1_0
                dist = 4 * dx * (cl_id - ID1)*(ID2 - cl_id) / ( ID2 - ID1)**2

            points.InsertNextPoint(np.asarray(voronoi_clipped.GetPoint(p)) + dist)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(p)

    else:
        # ..witout opthalmic artery
        ID1 = 0
        ID2 = len(get_curvilinear_coordinate(centerline_clipped)) - 1
        IDmid = int( (ID1 + ID2)/2.)
        for p in range(voronoi_clipped.GetNumberOfPoints()):
            cl_id = centerline_loc.FindClosestPoint(voronoi_clipped.GetPoint(p))

            dist = 4 * dx * (cl_id - ID1)*(ID2 - cl_id) / ( ID2 - ID1)**2

            points.InsertNextPoint(np.asarray(voronoi_clipped.GetPoint(p)) + dist)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(p)

    # Insert points and create new voronoi diagram
    newDataSet.SetPoints(points)
    newDataSet.SetVerts(verts)
    newDataSet.GetPointData().AddArray(voronoi_clipped.GetPointData().GetArray(radiusArrayName))
    return newDataSet


if  __name__ == "__main__":
    smooth, basedir, case, alpha, beta = read_command_line()
    folders = sorted([folder for folder in listdir(basedir) if folder[:2] in ["P0"]])
    name = "surface"
    point_path = "carotid_siphon_points.particles"

    if case is not None:
        print("==== Working on case %s ====" % case)
        dirpath = path.join(basedir, case)
        move_vessel(dirpath,smooth, name, point_path, alpha, beta)
    else:
        for folder in folders:
            print("==== Working on case %s ====" % folder)
            dirpath = path.join(basedir,folder)
            move_vessel(dirpath,smooth, name, point_path, alpha, beta)
