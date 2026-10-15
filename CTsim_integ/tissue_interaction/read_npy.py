from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "photon_output"

TOTAL_VIEWS = 600
EXPECTED_SHAPE = (64, 888)

VIEW_TO_INSPECT = 1

# Pixel coordinates use NumPy indexing: (row, channel).
SUSPICIOUS_PIXEL = (1, 830)

# Number of neighboring values to display on either side.
NEIGHBOR_RADIUS = 10


# ============================================================
# CHECK 1: LOAD AND INSPECT ONE DETECTOR IMAGE
# ============================================================

def inspect_single_view(view_id):
    file_path = OUTPUT_DIR / f"view_{view_id}_detector.npy"

    print("\n" + "=" * 65)
    print("CHECK 1: DETECTOR STATISTICS")
    print("=" * 65)

    if not file_path.is_file():
        print(f"ERROR: Detector file not found: {file_path}")
        return None

    try:
        image = np.load(file_path, allow_pickle=False)
    except Exception as exc:
        print(f"ERROR: Could not load detector image: {exc}")
        return None

    print(f"File: {file_path}")
    print(f"Shape: {image.shape}")
    print(f"Data type: {image.dtype}")

    if image.shape != EXPECTED_SHAPE:
        print(f"ERROR: Expected shape {EXPECTED_SHAPE}")
        return None

    if not np.issubdtype(image.dtype, np.number):
        print("ERROR: Detector image contains a non-numeric data type.")
        return None

    nan_count = int(np.count_nonzero(np.isnan(image)))
    inf_count = int(np.count_nonzero(np.isinf(image)))
    zero_count = int(np.count_nonzero(image == 0))
    negative_count = int(np.count_nonzero(image < 0))

    finite_mask = np.isfinite(image)

    print(f"Minimum: {np.nanmin(image)}")
    print(f"Maximum: {np.nanmax(image)}")
    print(f"Mean: {np.nanmean(image)}")
    print(f"Standard deviation: {np.nanstd(image)}")
    print(f"Zero-valued pixels: {zero_count}")
    print(f"Negative pixels: {negative_count}")
    print(f"NaN values: {nan_count}")
    print(f"Infinite values: {inf_count}")

    print("=" * 65)

    if not np.all(finite_mask):
        print("WARNING: Non-finite values were found.")

    # --------------------------------------------------------
    # Exact zero locations
    # --------------------------------------------------------

    zero_coordinates = np.argwhere(image == 0)

    print("\nZero-valued pixel locations:")

    if len(zero_coordinates) == 0:
        print("No zero-valued pixels found.")
    else:
        print(zero_coordinates)

    print(f"Total zero pixels: {len(zero_coordinates)}")

    # --------------------------------------------------------
    # Inspect suspicious pixel and neighboring values
    # --------------------------------------------------------

    row, channel = SUSPICIOUS_PIXEL

    print(
        f"\nValues around channel {channel}, row {row}:"
    )

    if 0 <= row < image.shape[0] and 0 <= channel < image.shape[1]:

        start = max(0, channel - NEIGHBOR_RADIUS)
        end = min(
            image.shape[1],
            channel + NEIGHBOR_RADIUS + 1
        )

        print(image[row, start:end])
        print(f"Target pixel value: {image[row, channel]}")

        if image[row, channel] == 0:
            print("WARNING: The target pixel has an exact zero value.")

    else:
        print("ERROR: The requested pixel is outside the detector.")

    # --------------------------------------------------------
    # Inspect central detector row
    # --------------------------------------------------------

    central_row = image[image.shape[0] // 2, :]

    print(
        f"\nValues across the central region, "
        f"row {image.shape[0] // 2}:"
    )

    print(central_row)

    return image


# ============================================================
# CHECK 2: INVESTIGATE ZERO PIXELS ACROSS ALL VIEWS
# ============================================================

def inspect_all_views():

    print("\n" + "=" * 65)
    print("CHECK 2: ZERO PIXELS ACROSS ALL VIEWS")
    print("=" * 65)

    if not OUTPUT_DIR.is_dir():
        print(f"ERROR: Output directory not found: {OUTPUT_DIR}")
        return None

    valid_views = []
    missing_views = []
    invalid_views = []

    zero_count_per_view = {}
    zero_coordinate_frequency = {}

    view_minimums = {}
    view_maximums = {}
    view_means = {}
    view_totals = {}

    for view_id in range(1, TOTAL_VIEWS + 1):

        file_path = OUTPUT_DIR / f"view_{view_id}_detector.npy"

        # ----------------------------------------------------
        # Check file existence
        # ----------------------------------------------------

        if not file_path.is_file():
            missing_views.append(view_id)
            continue

        # ----------------------------------------------------
        # Load file
        # ----------------------------------------------------

        try:
            image = np.load(file_path, allow_pickle=False)

        except Exception as exc:
            invalid_views.append(
                (view_id, f"Load error: {exc}")
            )
            continue

        # ----------------------------------------------------
        # Validate shape and type
        # ----------------------------------------------------

        if image.shape != EXPECTED_SHAPE:
            invalid_views.append(
                (
                    view_id,
                    f"Expected {EXPECTED_SHAPE}, got {image.shape}"
                )
            )
            continue

        if not np.issubdtype(image.dtype, np.number):
            invalid_views.append(
                (view_id, f"Non-numeric dtype: {image.dtype}")
            )
            continue

        # ----------------------------------------------------
        # Check numerical validity
        # ----------------------------------------------------

        if not np.all(np.isfinite(image)):
            invalid_views.append(
                (view_id, "Contains NaN or infinite values")
            )
            continue

        valid_views.append(view_id)

        # ----------------------------------------------------
        # Collect statistics
        # ----------------------------------------------------

        zero_coordinates = np.argwhere(image == 0)

        zero_count_per_view[view_id] = len(zero_coordinates)

        view_minimums[view_id] = float(np.min(image))
        view_maximums[view_id] = float(np.max(image))
        view_means[view_id] = float(np.mean(image))
        view_totals[view_id] = float(np.sum(image))

        # ----------------------------------------------------
        # Record every zero coordinate
        # ----------------------------------------------------

        for row, channel in zero_coordinates:

            coordinate = (int(row), int(channel))

            if coordinate not in zero_coordinate_frequency:
                zero_coordinate_frequency[coordinate] = []

            zero_coordinate_frequency[coordinate].append(view_id)

    # ========================================================
    # RESULTS
    # ========================================================

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Expected views: {TOTAL_VIEWS}")
    print(f"Valid views: {len(valid_views)}")
    print(f"Missing views: {len(missing_views)}")
    print(f"Invalid views: {len(invalid_views)}")

    if missing_views:
        print("\nMissing view numbers:")
        print(missing_views)

    if invalid_views:
        print("\nInvalid files:")

        for view_id, reason in invalid_views:
            print(f"View {view_id}: {reason}")

    # --------------------------------------------------------
    # Zero counts per view
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("ZERO-VALUED PIXELS PER VIEW")
    print("-" * 65)

    views_with_zeros = {
        view_id: count
        for view_id, count in zero_count_per_view.items()
        if count > 0
    }

    if not views_with_zeros:
        print("No exact-zero pixels found in any valid view.")

    else:
        print(f"Views containing zero pixels: {len(views_with_zeros)}")

        for view_id, count in views_with_zeros.items():
            print(f"View {view_id}: {count} zero pixels")

    total_zero_occurrences = sum(zero_count_per_view.values())

    print(f"\nTotal zero occurrences across all views: "
          f"{total_zero_occurrences}")

    # --------------------------------------------------------
    # Repeated zero coordinates
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("ZERO COORDINATES REPEATING ACROSS VIEWS")
    print("-" * 65)

    repeated_coordinates = {
        coordinate: views
        for coordinate, views in zero_coordinate_frequency.items()
        if len(views) > 1
    }

    if not repeated_coordinates:
        print("No detector coordinates repeat across views.")

    else:
        print(
            f"Unique coordinates containing zeros: "
            f"{len(zero_coordinate_frequency)}"
        )

        print(
            f"Coordinates repeating across views: "
            f"{len(repeated_coordinates)}"
        )

        for coordinate, views in sorted(
            repeated_coordinates.items(),
            key=lambda item: len(item[1]),
            reverse=True
        ):

            print(
                f"Pixel {coordinate}: zero in {len(views)} views; "
                f"first 20 views = {views[:20]}"
            )

    # --------------------------------------------------------
    # Specifically track suspicious pixel (1, 830)
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print(f"TRACKING PIXEL {SUSPICIOUS_PIXEL} ACROSS ALL VIEWS")
    print("-" * 65)

    suspicious_zero_views = []

    for view_id in valid_views:

        file_path = OUTPUT_DIR / f"view_{view_id}_detector.npy"
        image = np.load(file_path, allow_pickle=False)

        value = image[SUSPICIOUS_PIXEL]

        if value == 0:
            suspicious_zero_views.append(view_id)

        if view_id <= 10 or value == 0:
            print(
                f"View {view_id:3d}: "
                f"value = {value:.8f}"
            )

    print(
        f"\nPixel {SUSPICIOUS_PIXEL} is zero in "
        f"{len(suspicious_zero_views)} valid views."
    )

    if suspicious_zero_views:
        print("Affected views:", suspicious_zero_views)

    # --------------------------------------------------------
    # Signal statistics across views
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("TOTAL DETECTOR SIGNAL ACROSS VIEWS")
    print("-" * 65)

    if valid_views:

        totals = np.array(
            [view_totals[view_id] for view_id in valid_views]
        )

        print(f"Minimum view total: {totals.min():.8e}")
        print(f"Maximum view total: {totals.max():.8e}")
        print(f"Mean view total: {totals.mean():.8e}")
        print(f"Standard deviation: {totals.std():.8e}")

        print("\nFirst 10 valid views:")

        for view_id in valid_views[:10]:

            print(
                f"View {view_id:3d} | "
                f"min={view_minimums[view_id]:.6g} | "
                f"max={view_maximums[view_id]:.6g} | "
                f"mean={view_means[view_id]:.6g} | "
                f"total={view_totals[view_id]:.6g} | "
                f"zeros={zero_count_per_view[view_id]}"
            )

    print("=" * 65)

    return {
        "valid_views": valid_views,
        "missing_views": missing_views,
        "invalid_views": invalid_views,
        "zero_count_per_view": zero_count_per_view,
        "zero_coordinate_frequency": zero_coordinate_frequency,
        "repeated_coordinates": repeated_coordinates,
        "view_totals": view_totals,
    }


# ============================================================
# CHECK 3: DISPLAY DETECTOR IMAGE
# ============================================================

def plot_detector_image(image, view_id):

    if image is None:
        print("\nCannot plot detector image: loading or validation failed.")
        return

    print("\n" + "=" * 65)
    print("CHECK 3: DETECTOR IMAGE VISUALIZATION")
    print("=" * 65)

    plt.figure(figsize=(15, 6))

    plt.imshow(
        image,
        aspect="auto",
        origin="upper"
    )

    plt.colorbar(label="Detector signal")
    plt.xlabel("Detector channel")
    plt.ylabel("Detector row")
    plt.title(f"Detector Image — View {view_id}")

    plt.tight_layout()

    plot_path = OUTPUT_DIR / "detector_output.png"

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"Plot saved to: {plot_path}")

    plt.show()


# ============================================================
# CHECK 4: PLOT SIGNAL AND ZERO COUNTS ACROSS ALL VIEWS
# ============================================================

def plot_view_statistics(results):

    if results is None:
        return

    valid_views = results["valid_views"]
    view_totals = results["view_totals"]
    zero_count_per_view = results["zero_count_per_view"]

    if not valid_views:
        print("Cannot plot view statistics: no valid views.")
        return

    totals = [
        view_totals[view_id]
        for view_id in valid_views
    ]

    zero_counts = [
        zero_count_per_view[view_id]
        for view_id in valid_views
    ]

    # --------------------------------------------------------
    # Total detector signal
    # --------------------------------------------------------

    plt.figure(figsize=(14, 5))

    plt.plot(valid_views, totals)

    plt.xlabel("View number")
    plt.ylabel("Total detector signal")
    plt.title("Total Detector Signal Across Views")

    plt.grid(True)
    plt.tight_layout()

    total_plot_path = OUTPUT_DIR / "total_signal_across_views.png"

    plt.savefig(
        total_plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"Signal plot saved to: {total_plot_path}")

    plt.show()

    # --------------------------------------------------------
    # Zero pixels per view
    # --------------------------------------------------------

    plt.figure(figsize=(14, 5))

    plt.plot(valid_views, zero_counts)

    plt.xlabel("View number")
    plt.ylabel("Number of zero-valued pixels")
    plt.title("Zero-Valued Detector Pixels Across Views")

    plt.grid(True)
    plt.tight_layout()

    zero_plot_path = OUTPUT_DIR / "zero_pixels_across_views.png"

    plt.savefig(
        zero_plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"Zero-pixel plot saved to: {zero_plot_path}")

    plt.show()


# ============================================================
# MAIN DIAGNOSTIC EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\nCT SIMULATION — DETECTOR DIAGNOSTICS")
    print(f"Working directory: {Path.cwd()}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Detector output directory not found: {OUTPUT_DIR}"
        )

    # Check 1: Inspect one detector image.
    detector_image = inspect_single_view(VIEW_TO_INSPECT)

    # Check 2: Investigate zeros across all views.
    results = inspect_all_views()

    # Check 3: Display the detector image.
    plot_detector_image(
        detector_image,
        VIEW_TO_INSPECT
    )

    # Check 4: Plot view statistics.
    plot_view_statistics(results)

    print("\nAll detector diagnostics completed.")