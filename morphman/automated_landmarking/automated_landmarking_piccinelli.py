##   Copyright (c) Aslak W. Bergersen, Henrik A. Kjeldsberg. All rights reserved.
##   See LICENSE file for details.

##      This software is distributed WITHOUT ANY WARRANTY; without even
##      the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##      PURPOSE.  See the above copyright notices for more information.

import os

import matplotlib.pyplot as plt
from matplotlib import rc

from morphman.automated_landmarking.automated_landmarking_tools import *

# Local import

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})


def landmarking_piccinelli(centerline, base_path, approximation_method, algorithm, resampling_step,
                           smooth_line, nknots, smoothing_factor_curv, smoothing_factor_torsion,
                           iterations):
    """
    Perform landmarking of an input centerline to
    identify different segments along the vessel.
    Landmarking algorithm based on Bogunovic et al. (2012).
    Uses curvature and torsion to objectively subdivide
    the siphon into bends.
    Subdivision of individual siphon bends is
    performed by identifying locations of curvature and torsion peaks along
    the siphon and defining a bend for each curvature peak delimited by 2
    enclosing (proximal and distal) torsion peaks.

    Args:
        centerline (vtkPolyData): Centerline data points.
        base_path (str): Location of case to landmark.
        approximation_method (str): Method used for computing curvature.
        algorithm (str): Name of landmarking algorithm.
        resampling_step (float): Resampling step. Is None if no resampling.
        smooth_line (bool): Smooths centerline with VMTK if True.
        nknots (int): Number of knots for B-splines.
        smoothing_factor_curv (float): Smoothing factor for computing curvature.
        smoothing_factor_torsion (float): Smoothing factor for computing torsion.
        iterations (int): Number of smoothing iterations.

    Returns:
        landmarks (dict): Landmarking interfaces as points.
    """
    bogunovic_models = False
    pts_ids = []
    plotline = True
    if bogunovic_models:
        a = np.asarray([23.25749969482422, 31.970500946044922, 38.20930099487305])
        b = np.asarray([23.860700607299805, 27.085100173950195, 39.113399505615234])
        c = np.asarray([19.331199645996094, 20.83060073852539, 35.12670135498047])
        d = np.asarray([10.315199851989746, 6.9745001792907715, 31.156400680541992])
        points = [a, b, c, d]

    if not bogunovic_models:
        a = np.asarray([43.57447814941406, 46.041900634765625, 44.09065246582031])
        b = np.asarray([44.350345611572266, 40.11917495727539, 43.808135986328125])
        c = np.asarray([46.78703689575195, 32.10150146484375, 51.94873809814453])
        d = np.asarray([55.90689468383789, 24.68092155456543, 57.486629486083984])
        e = np.asarray([61.60901641845703, 14.373627662658691, 65.34747314453125])

        points = [a, b, c, d, e]

    ax1 = plt.subplot(3, 1, 3)
    ax2 = plt.subplot(2, 3, 1)
    ax3 = plt.subplot(2, 3, 2)
    ax4 = plt.subplot(2, 3, 3)

    # PLot empty figure
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_axis_off()

    peaks_torsion = []
    peaks_curvature = []
    resampling_steps = [0.05, 0.075, 0.1, 0.25, 0.5]
    colors = [
        "#003f5c",
        "#58508d",
        "#bc5090",
        "#ff6361",
        "#ffa600"
    ]
    for i, resampling_step in enumerate(resampling_steps):
        if resampling_step is not None:
            centerline = vmtk_resample_centerline(centerline, resampling_step)

        if approximation_method == "vmtk":
            line = centerline
            line_curv = vmtk_compute_geometric_features(centerline, True, outputsmoothed=False,
                                                        factor=smoothing_factor_curv, iterations=iterations)
            line_tor = vmtk_compute_geometric_features(centerline, True, outputsmoothed=False,
                                                       factor=smoothing_factor_torsion, iterations=iterations)
            # Get curvature and torsion, find peaks
            curvature = get_point_data_array("Curvature", line_curv)
            torsion = get_point_data_array("Torsion", line_tor)

            # Smooth torsion curve to remove noise
            torsion_smooth = gaussian_filter(torsion, 25)
            curv_smooth = gaussian_filter(curvature, 5)
            max_point_ids = list(argrelextrema(curv_smooth, np.greater)[0])
            max_point_tor_ids = list(argrelextrema(abs(torsion_smooth), np.greater)[0])
            peaks_curvature.append(len(max_point_ids))
            peaks_torsion.append(len(max_point_tor_ids))

        else:
            raise ValueError("ERROR: Selected method for computing curvature / torsion not available" +
                             "\nPlease select between 'spline' and 'vmtk'")

        abscissa = -get_curvilinear_coordinate(centerline)[::-1]
        linewidth = 2
        fontsize = 25
        labelsize = 20

        ax3.plot(curv_smooth, abscissa, '-', label=r"$r$=%.3f" % resampling_step, linewidth=linewidth, color=colors[i])
        ax4.plot(torsion_smooth, abscissa, '-', linewidth=linewidth, color=colors[i])
        if plotline:
            locator = get_vtk_point_locator(centerline)
            for point in points:
                id = locator.FindClosestPoint(point)
                pts_ids.append(id)

            for ID in pts_ids:
                ax3.axhline(abscissa[ID], color='k', linewidth=2)
                ax4.axhline(abscissa[ID], color='k', linewidth=2)
        plotline = False

    markers = ["s", "o", "v", "D"]
    # Curvature
    ax3.set_xlabel(r'Curvature, $\kappa$', fontsize=fontsize)
    ax3.set_xlim([-0.02, 0.7])
    ax3.set_ylim([-82, 0])
    ax3.set_ylabel(r'Abscissa [mm]', fontsize=fontsize)

    # Torsion
    ax4.set_xlabel(r'Torsion, $\tau$', fontsize=fontsize)
    ax4.set_yticks([])
    ax4.set_xlim([-0.75, 0.9])
    ax4.set_ylim([-82, 0])

    # Plot resamp vs n peaks
    l_peaks_torsion = np.log(peaks_torsion)
    l_peaks_curv = np.log(peaks_curvature)
    l_resam_step = np.log(resampling_steps)
    colors_peaks = ["#4B878BFF", "#D01C1FFF"]
    ax1.plot(l_resam_step, l_peaks_torsion, label="Torsion", linewidth=linewidth, marker=markers[2], markersize=8,
             color=colors_peaks[0])
    ax1.plot(l_resam_step, l_peaks_curv, label="Curvature", linewidth=linewidth, marker=markers[3], markersize=8,
             color=colors_peaks[1])
    ax1.set_xlabel(r'$\ln ( r ) $', fontsize=fontsize)
    ax1.set_ylabel(r'$\ln ( \# $Peaks$ )$', fontsize=fontsize)

    ax4.set_yticks([])

    # Legendes
    ax1.legend(loc='upper right', fontsize=labelsize)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.29), ncol=3, fontsize=labelsize)

    # Tick params
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)
    ax3.tick_params(axis='both', which='major', labelsize=labelsize)
    ax4.tick_params(axis='both', which='major', labelsize=labelsize)

    plt.show()
    return

    if resampling_step is not None:
        centerline = vmtk_resample_centerline(centerline, resampling_step)

    if approximation_method == "spline":
        line, max_point_ids, _ = spline_centerline_and_compute_geometric_features(centerline, smooth_line, nknots)

        # Get curvature and torsion, find peaks
        curvature = get_point_data_array("Curvature", line)
        torsion = get_point_data_array("Torsion", line)
        torsion_smooth = gaussian_filter(torsion, 10)
        max_point_tor_ids = list(argrelextrema(abs(torsion_smooth), np.greater)[0])

    elif approximation_method == "vmtk":
        line = centerline
        line_curv = vmtk_compute_geometric_features(centerline, True, outputsmoothed=False,
                                                    factor=smoothing_factor_curv, iterations=iterations)
        line_tor = vmtk_compute_geometric_features(centerline, True, outputsmoothed=False,
                                                   factor=smoothing_factor_torsion, iterations=iterations)
        # Get curvature and torsion, find peaks
        curvature = get_point_data_array("Curvature", line_curv)
        torsion = get_point_data_array("Torsion", line_tor)

        # Smooth torsion curve to remove noise
        torsion_smooth = gaussian_filter(torsion, 25)
        max_point_ids = list(argrelextrema(curvature, np.greater)[0])
        max_point_tor_ids = list(argrelextrema(abs(torsion_smooth), np.greater)[0])

    else:
        raise ValueError("ERROR: Selected method for computing curvature / torsion not available" +
                         "\nPlease select between 'spline' and 'vmtk'")

    # Extract local curvature minimums
    length = get_curvilinear_coordinate(line)

    # Remove points too close to the ends of the siphon
    for i in max_point_ids:
        if length[i] in length[-10:] or length[i] in length[:10]:
            max_point_ids.remove(i)

    # Remove curvature and torsion peaks too close to each other
    tolerance = 70
    dist = []
    dist_tor = []
    for i in range(len(max_point_ids) - 1):
        dist.append(max_point_ids[i + 1] - max_point_ids[i])
    for i in range(len(max_point_tor_ids) - 1):
        dist_tor.append(max_point_tor_ids[i + 1] - max_point_tor_ids[i])

    curv_remove_ids = []
    for i, dx in enumerate(dist):
        if dx < tolerance:
            curv1 = curvature[max_point_ids[i]]
            curv2 = curvature[max_point_ids[i + 1]]
            if curv1 > curv2:
                curv_remove_ids.append(max_point_ids[i + 1])
            else:
                curv_remove_ids.append(max_point_ids[i])

    tor_remove_ids = []
    for i, dx in enumerate(dist_tor):
        if dx < tolerance:
            tor1 = torsion_smooth[max_point_tor_ids[i]]
            tor2 = torsion_smooth[max_point_tor_ids[i + 1]]
            if tor1 > tor2:
                tor_remove_ids.append(max_point_tor_ids[i + 1])
            else:
                tor_remove_ids.append(max_point_tor_ids[i])

    max_point_ids = [ID for ID in max_point_ids if ID not in curv_remove_ids]
    max_point_tor_ids = [ID for ID in max_point_tor_ids if ID not in tor_remove_ids]

    # Define bend interfaces based on Piccinelli et al.
    def find_interface():
        found = False
        interface = {}
        k = 0
        start_id = 0
        for c in max_point_ids:
            for i in range(start_id, len(max_point_tor_ids) - 1):
                if max_point_tor_ids[i] < c < max_point_tor_ids[i + 1] and not found:
                    interface["bend%s" % (k + 1)] = np.array([max_point_tor_ids[i]])
                    k += 1
                    interface["bend%s" % (k + 1)] = np.array([max_point_tor_ids[i + 1]])
                    k += 1
                    start_id = i + 1
                    found = True
            found = False

        return interface

    # Compute and extract interface points
    interfaces = find_interface()
    landmarks = {}
    for k, v in interfaces.items():
        landmarks[k] = line.GetPoints().GetPoint(int(v))

    # Map landmarks to initial centerline
    landmarks = map_landmarks(landmarks, centerline, algorithm)

    # Save landmarks
    print("-- Case was successfully landmarked.")
    print("-- Number of landmarks (Segments): %s" % len(landmarks))
    try:
        os.remove(base_path + "_landmark_piccinelli_%s_r_%.2f_f_%.1f_i_%i.particles" % (
            approximation_method, resampling_step, smoothing_factor_torsion, iterations))
        os.remove(base_path + "_info.json")
    except:
        pass
    if landmarks is not None:
        write_parameters(landmarks, base_path)
        create_particles(base_path, algorithm, approximation_method, resampling_step, smoothing_factor_torsion,
                         iterations)

    return landmarks
