import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import math
import re
import matplotlib.pyplot as plt
from io import BytesIO

if "calibration_runs" not in st.session_state:
    st.session_state.calibration_runs = []

# --- Title and Setup ---
st.set_page_config(layout="wide")
st.title("GAF Film Calibration GUI")

# --- Sidebar Controls ---
st.sidebar.header("Parameters")



folder_path = st.sidebar.text_input("Folder containing calibration GAF images:", value="calibration")
measure_folder = st.sidebar.text_input("Folder containing measurement GAF images:", value="measurement")
image_extension = st.sidebar.text_input("Image extension (e.g., .tif):", value=".tif")
name_irradiated = st.sidebar.text_input("Prefix for irradiated films:", value="irr")
name_velo = st.sidebar.text_input("Prefix for unirradiated films:", value="velo")

x_values_file = st.sidebar.text_input("MU values file (MU):", value="MU_values.dat")

dcc_file = st.sidebar.text_input("Dose conversion coefficient values file (cGy):", value="dcc.dat")


# Read dose conversion coefficient from file
try:
    with open(dcc_file, "r") as f:
        dose_conversion_coefficient = float(f.read().strip())
except Exception as e:
    st.sidebar.error("Error reading dcc.dat, is the file present in the same folder? It is necessary for calibration.")
    dose_conversion_coefficient = 100.0  # fallback

st.sidebar.markdown(f"**Dose Conversion Coefficient:** `{dose_conversion_coefficient:.3f} cGy`")


white_threshold_percent = st.sidebar.number_input(
    "White border segmentation threshold (%)",
    min_value=0.0, max_value=100.0, value=2.0, step=0.1, format="%.2f"
)

tolerance_percent = st.sidebar.number_input(
    "Green square segmentation treshold (%)",
    min_value=0.0, max_value=100.0, value=2.0, step=0.1, format="%.2f"
)

poly_order = st.sidebar.slider("Polynomial fit order", 1, 10, 4)

num_points = st.sidebar.number_input(
    "Number of fit points",
    min_value=10, max_value=10000, value=500, step=1, format="%i"
)

#num_points = st.sidebar.slider("Number of fit points", 100, 1000, 500)

show_final_seg = st.sidebar.checkbox("Show Irradiated Final Segmentation")
show_final_seg_velo = st.sidebar.checkbox("Show Unirradiated Final Segmentation")

if st.sidebar.button("Clear Saved Calibrations"):
    st.session_state.calibration_runs = []
    st.sidebar.success("Calibration history cleared.")



# --- Functions (from your script, slightly adapted) ---
def get_image_sequence(folder_path, image_extension, name_type):
    pattern = re.compile(rf"^{re.escape(name_type)}0*(\d+){re.escape(image_extension)}$")
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(image_extension):
            match = pattern.match(filename)
            if match:
                image_files.append((filename, int(match.group(1))))
    image_files.sort(key=lambda x: x[1])
    return [f[0] for f in image_files]

def load_image_and_segment_white(path, white_treshold_percent):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_white = np.max(gray)
    min_white = int(math.floor(max_white - (max_white * (white_treshold_percent / 100))))
    _, white_mask = cv2.threshold(gray, min_white, max_white, cv2.THRESH_BINARY)
    green_square = cv2.bitwise_not(white_mask)
    kernel = np.ones((3, 3), np.uint8)
    green_square = cv2.morphologyEx(green_square, cv2.MORPH_OPEN, kernel)
    green_square = cv2.morphologyEx(green_square, cv2.MORPH_CLOSE, kernel)
    return image, green_square

def final_segmentation(image, tolerance_percent, green_square):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    green_pixels = gray_image[green_square == 255]
    median_val = np.median(green_pixels)
    tolerance = median_val * (tolerance_percent / 100)
    lower_bound = median_val - tolerance
    upper_bound = median_val + tolerance
    within_range_mask = ((gray_image >= lower_bound) & (gray_image <= upper_bound)).astype(np.uint8) * 255
    final_mask = cv2.bitwise_and(within_range_mask, green_square)
    selected_pixels = gray_image[final_mask == 255]
    mean_val = np.mean(selected_pixels)
    std_dev_val = np.std(selected_pixels)
    return median_val, mean_val, std_dev_val, final_mask

def display_overlay(image, mask, title):
    overlay_result = image.copy()
    overlay_result[mask == 255] = [0, 0, 255]
    blended = cv2.addWeighted(image, 0.7, overlay_result, 0.3, 0)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")
    return fig

def write_dat_file(mean_values, stdev_values, filename):
    data = np.column_stack((mean_values, stdev_values))
    header = "Mean\tStdDev"
    np.savetxt(filename, data, delimiter='\t', header=header, comments='', fmt='%.6f')

def read_measurements(input_file):
    df = pd.read_csv(input_file, sep="\t", skipinitialspace=True)
    return df["Mean"].values, df["StdDev"].values

def read_x_values(x_values_file, coeff):
    x = np.loadtxt(x_values_file)
    return (coeff / 100) * x

def compute_optical_density(reading, reading_err, reference, reference_err):
    ln10 = np.log(10)
    OD = np.log10(reference / reading)
    OD_err = np.sqrt((reference_err / (reference * ln10))**2 + (reading_err / (reading * ln10))**2)
    return OD, OD_err

def perform_weighted_polyfit_inverted(x, y, x_err, order):
    weights = 1 / x_err
    coeffs = np.polyfit(x, y, order, w=weights)
    poly_fit = np.poly1d(coeffs)
    y_pred = poly_fit(x)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return poly_fit, coeffs, r2


def compare_fitted_vs_measured_df(OD, x_measured, poly_fit):
    fitted_dose = poly_fit(OD)
    abs_error = fitted_dose - x_measured
    percent_error = 100 * abs_error / x_measured

    df = pd.DataFrame({

        "OD": OD,
        "Measured Dose [cGy]": x_measured,
        "Fitted Dose [cGy]": fitted_dose,
        "Abs Error": abs_error,
        "% Error": percent_error
    }).rename_axis("Index").set_index(pd.RangeIndex(start=1, stop=len(OD)+1))
    return df


def format_sci_latex_signed(c):
    """Return LaTeX-formatted coefficient with correct sign and scientific notation."""
    if c == 0:
        return "+ 0"
    sign = "+" if c >= 0 else "-"
    abs_c = abs(c)

    # Clean exponent + base
    exponent = int(np.floor(np.log10(abs_c)))
    base = abs_c / (10 ** exponent)

    if exponent == 0:
        base_str = f"{base:.3g}"
        return f"{sign} {base_str}"
    else:
        base_str = f"{base:.3g}"
        return f"{sign} {base_str} \\times 10^{{{exponent}}}"


with st.expander("ℹ️ Show Info About This App"):
    st.markdown("""
    **This application:**
    - Performs GAF film calibration using automatic image segmentation and polynomial fitting.
    - Allows to quickly make measurements when already possessing a calibration.

    **Calibration Requirements**:
    - Place the numbered images (e.g. `.tif`) in the selected folder (e.g. `calibration`) and select the accurate prefixes (e.g. `irr`, `velo`)
    - `MU_values.dat` should be placed in the script folder and each line should contain the erogated MU for each film, in order.
    - The dose conversion coefficient can be placed in a file `dcc.dat`. This is how much cGy are actually received at the films geometry when using 100 MU.

    **Measurement Requirements**:
    - Place the numbered images (e.g. `.tif`) in the selected folder (e.g. `measurement`) and select the accurate prefixes (e.g. `irr`, `velo`)
    - Choose a calibration file or select a previously ran calibration from the list.
    
    **Workflow Summary**:
    - Automatically segment irradiated and unirradiated film scans
    - Compute mean value and standard deviation within the defined region
    - Compute Optical Density and fit a calibration curve to dose using a polynomial
    - Compare different calibrations
    - Make measurement using previous calibrations




    **Adjust parameters in the sidebar before running**
    """)

# --- Main logic ---

st.subheader("Make a New Calibration")


if st.button("Run Calibration"):
    st.session_state.run_calibration = True
if st.session_state.get("run_calibration", False):

    st.success("Running calibration pipeline...")
    irradiated_list = get_image_sequence(folder_path, image_extension, name_irradiated)
    velo_list = get_image_sequence(folder_path, image_extension, name_velo)

    if len(irradiated_list) != len(velo_list):
        st.error("Mismatch in number of irradiated and unirradiated films")
    else:
        mean_values_irr, stdev_values_irr = [], []
        mean_values_velo, stdev_values_velo = [], []

        for name_irr, name_vel in zip(irradiated_list, velo_list):
            path_irr = os.path.join(folder_path, name_irr)
            path_vel = os.path.join(folder_path, name_vel)

            img_irr, mask_irr = load_image_and_segment_white(path_irr, white_threshold_percent)
            _, mean_irr, std_irr, final_mask_irr = final_segmentation(img_irr, tolerance_percent, mask_irr)
            mean_values_irr.append(mean_irr)
            stdev_values_irr.append(std_irr)

            if show_final_seg:
                st.pyplot(display_overlay(img_irr, final_mask_irr, f"Final segmentation: {name_irr}"))
            
            

            img_vel, mask_vel = load_image_and_segment_white(path_vel, white_threshold_percent)
            _, mean_vel, std_vel, final_mask_vel = final_segmentation(img_vel, tolerance_percent, mask_vel)
            mean_values_velo.append(mean_vel)
            stdev_values_velo.append(std_vel)

            if show_final_seg_velo:
                st.pyplot(display_overlay(img_vel, final_mask_vel, f"Final segmentation: {name_vel}"))

        input_file = os.path.join(folder_path, 'readings_irradiated.dat')
        ref_file = os.path.join(folder_path, 'readings_velo.dat')

        write_dat_file(np.array(mean_values_irr), np.array(stdev_values_irr), input_file)
        write_dat_file(np.array(mean_values_velo), np.array(stdev_values_velo), ref_file)

        reading, reading_err = read_measurements(input_file)
        reference, reference_err = read_measurements(ref_file)
        x_measured = read_x_values(x_values_file, dose_conversion_coefficient)

        OD, OD_err = compute_optical_density(reading, reading_err, reference, reference_err)
        poly_fit, coeffs, r2 = perform_weighted_polyfit_inverted(OD, x_measured, OD_err, poly_order)


        # Build full LaTeX equation
        terms = []
        degree = len(coeffs) - 1
        
        for i, c in enumerate(coeffs):
            power = degree - i
            coeff_latex = format_sci_latex_signed(c)
        
            # Format OD term
            if power == 0:
                term = f"{coeff_latex}"
            elif power == 1:
                term = f"{coeff_latex} \\cdot \\text{{OD}}"
            else:
                term = f"{coeff_latex} \\cdot \\text{{OD}}^{power}"
        
            terms.append(term)
        
        # Join terms correctly
        latex_eq = " ".join(terms)
        
        # Strip leading '+' and clean whitespace
        if latex_eq.startswith("+"):
            latex_eq = latex_eq[2:].lstrip()
        
        latex_display = f"D(\\text{{OD}}) = {latex_eq}"
        
        st.subheader("Fitted Polynomial Equation")
        st.latex(latex_display)



        # Display fit results in one line
        coeff_text = "\t".join([f"a{len(coeffs)-1-i} = {c:.6g}" for i, c in enumerate(coeffs)])
        fit_summary = f"{coeff_text}\tR² = {r2:.4f}"
        
        st.subheader("Fit Results")
        st.text(fit_summary)


        fig, ax = plt.subplots(figsize=(10, 6), dpi = 300)

        # Plot fit and data
        OD_fit = np.linspace(np.min(OD), np.max(OD), num_points)
        dose_fit = poly_fit(OD_fit)
        ax.plot(OD_fit, dose_fit, 'r-', label='Polynomial Fit')
        ax.errorbar(OD, x_measured, xerr=OD_err, fmt='o', label='Measured Data', capsize=3)
        
        # Labels and grid
        ax.set_xlabel('Optical Density (OD)')
        ax.set_ylabel('Dose [cGy]')
        ax.set_title('Polynomial Fit: Dose vs OD')
        ax.grid(True)
        # Create the legend and store its reference
        legend = ax.legend(loc='upper left')
        
        # Format the fit results
        lines = [f"a{len(coeffs)-1-i} = {c:.5g}" for i, c in enumerate(coeffs)]
        lines.append(f"R² = {r2:.4f}")
        textstr = "\n".join(lines)
        
        # Get legend bounding box (in axes coordinates)
        renderer = fig.canvas.get_renderer()
        bbox = legend.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transAxes.inverted())  # convert to axes coords
        
        # Calculate position to the right of the legend box
        legend_right = bbox_data.x1
        legend_top = bbox_data.y1
        
        # Shift right with a small offset (e.g., 0.01)
        offset_x = 0.01
        box_x = legend_right + offset_x
        box_y = legend_top
        
        # Display the box
        props = dict(boxstyle='round', facecolor='white', alpha=0.85)
        ax.text(box_x, box_y, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)
        
        st.pyplot(fig)





        # Show comparison table
        st.subheader("Dose Comparison Table")
        comparison_df = compare_fitted_vs_measured_df(OD, x_measured, poly_fit)
        st.dataframe(comparison_df.style.format({
            "OD": "{:.4f}",
            "Measured Dose [cGy]": "{:.2f}",
            "Fitted Dose [cGy]": "{:.2f}",
            "Abs Error": "{:.2f}",
            "% Error": "{:.2f}"
        }))


        ## save the run?
        st.subheader("Save This Calibration Run")

        run_label = st.text_input("Label this run", value=f"Run {len(st.session_state.calibration_runs) + 1}")
        
        if st.button("Save Calibration Run"):
            st.session_state.calibration_runs.append({
                "label": run_label,
                "OD": OD,
                "OD_err": OD_err,
                "x_measured": x_measured,
                "fit": poly_fit,
                "coeffs": coeffs,
                "r2": r2
            })
            st.success(f"Saved calibration as '{run_label}'")


        if st.button("Write calibration.dat to script folder"):
            try:
                path = os.path.join(os.getcwd(), "calibration.dat")
                with open(path, "w") as f:
                    for c in coeffs[::-1]:  # write from lowest degree to highest
                        f.write(f"{c:.8e}\n")
                st.success(f"`calibration.dat` saved to: `{path}`")
            except Exception as e:
                st.error(f"Failed to write calibration.dat: {e}")



        

        # --- Comparison of saved calibrations ---
        if len(st.session_state.calibration_runs) > 1:
            st.subheader("Compare Saved Calibration Runs")
        
            selected_labels = st.multiselect(
                "Select runs to compare:",
                options=[run["label"] for run in st.session_state.calibration_runs],
                default=[run["label"] for run in st.session_state.calibration_runs]
            )
        
            fig, ax = plt.subplots(figsize=(10, 6))
        
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
            for i, run in enumerate(st.session_state.calibration_runs):
                if run["label"] not in selected_labels:
                    continue
        
                color = color_cycle[i % len(color_cycle)]
        
                # Fit curve
                OD_range = np.linspace(np.min(run["OD"]), np.max(run["OD"]), 500)
                dose_fit = run["fit"](OD_range)
                ax.plot(OD_range, dose_fit, label=f"{run['label']} (fit)", color=color)
        
                # Data points with error bars
                ax.errorbar(run["OD"], run["x_measured"], xerr=run["OD_err"],
                            fmt='o', markersize=4, capsize=3, linestyle='none',
                            label=f"{run['label']} (data)", color=color)
        
            ax.set_xlabel("Optical Density (OD)")
            ax.set_ylabel("Dose [cGy]")
            ax.set_title("Comparison of Calibration Fits")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
                                   





        st.subheader("Downloads")

        # Save plot button
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("Download Plot as PNG", data=buf.getvalue(), file_name="dose_fit_plot.png", mime="image/png")

        # Optional: download as CSV
        csv_bytes = comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Comparison Table (CSV)", data=csv_bytes, file_name="dose_comparison.csv", mime="text/csv")

        # Export calibration coefficients
        calibration_txt = "\n".join([f"{c:.8e}" for c in coeffs[::-1]])  # lowest order first
        st.download_button("Download Calibration Coefficients (calibration.dat)",
                           data=calibration_txt,
                           file_name="calibration.dat",
                           mime="text/plain")


        # Save final .dat files
        st.download_button("Download Irradiated Data (.dat)", data=open(input_file, "rb").read(), file_name="readings_irradiated.dat")
        st.download_button("Download Unirradiated Data (.dat)", data=open(ref_file, "rb").read(), file_name="readings_velo.dat")

        






# --- Calibration selection for measurement ---
st.subheader("Make a Measurement Using a Previous Calibration")

# Check if 'calibration.dat' exists in script folder
script_folder_calib_path = os.path.join(os.getcwd(), "calibration.dat")
file_in_script_folder = os.path.isfile(script_folder_calib_path)

calib_options = [
    "Upload calibration.dat",
    "Use saved calibration run",
    "Use calibration.dat from script folder"
]

default_selection = 2 if file_in_script_folder else 0  # index for default choice

calib_source = st.radio("Choose calibration source:", calib_options, index=default_selection)

selected_calib = None
uploaded_file = None
poly_fit = None

if calib_source == "Upload calibration.dat":
    uploaded_file = st.file_uploader("Upload calibration.dat", type=["dat", "txt"])
    if uploaded_file:
        try:
            coeffs = np.loadtxt(uploaded_file)
            poly_fit = np.poly1d(coeffs[::-1])
            st.success("Loaded uploaded calibration successfully.")
        except Exception as e:
            st.error(f"Failed to parse uploaded file: {e}")

elif calib_source == "Use saved calibration run":
    if not st.session_state.calibration_runs:
        st.warning("No saved calibrations available.")
    else:
        calib_labels = [run["label"] for run in st.session_state.calibration_runs]
        selected_label = st.selectbox("Select saved calibration:", calib_labels)
        selected_calib = next(run for run in st.session_state.calibration_runs if run["label"] == selected_label)
        poly_fit = selected_calib["fit"]

elif calib_source == "Use calibration.dat from script folder":
    if not file_in_script_folder:
        st.error("`calibration.dat` not found in script folder.")
    else:
        try:
            coeffs = np.loadtxt(script_folder_calib_path)
            poly_fit = np.poly1d(coeffs[::-1])  # reverse to highest-degree first
            st.success("Loaded calibration from script folder.")
        except Exception as e:
            st.error(f"Failed to read calibration.dat: {e}")




# Now make a measurement

if st.button("Make Measurement"):
    st.session_state.make_measurement = True

if st.session_state.get("make_measurement", False):
    st.success("Running measurement pipeline...")

    irr_list = get_image_sequence(measure_folder, image_extension, name_irradiated)
    velo_list = get_image_sequence(measure_folder, image_extension, name_velo)

    if len(irr_list) != len(velo_list):
        st.error("Mismatch in number of irradiated and unirradiated films")
    

        
    if poly_fit is None:
        st.error("No valid calibration loaded. Cannot proceed with measurement.")
        st.stop()


    mean_irr, std_irr, mean_velo, std_velo = [], [], [], []

    for f_irr, f_vel in zip(irr_list, velo_list):
        path_irr = os.path.join(measure_folder, f_irr)
        path_vel = os.path.join(measure_folder, f_vel)

        img_irr, mask_irr = load_image_and_segment_white(path_irr, white_threshold_percent)
        _, mean_i, std_i, _ = final_segmentation(img_irr, tolerance_percent, mask_irr)
        mean_irr.append(mean_i)
        std_irr.append(std_i)

        img_vel, mask_vel = load_image_and_segment_white(path_vel, white_threshold_percent)
        _, mean_v, std_v, _ = final_segmentation(img_vel, tolerance_percent, mask_vel)
        mean_velo.append(mean_v)
        std_velo.append(std_v)

    reading = np.array(mean_irr)
    reading_err = np.array(std_irr)
    reference = np.array(mean_velo)
    reference_err = np.array(std_velo)

    OD, OD_err = compute_optical_density(reading, reading_err, reference, reference_err)
    dose_estimates = poly_fit(OD)

    df = pd.DataFrame({
        "Film": irr_list,
        "OD": OD,
        "OD Error": OD_err,
        "Estimated Dose [cGy]": dose_estimates
    })
    
    df.index = np.arange(1, len(df) + 1)  # Set index to start from 1
    
    st.subheader("Measurement Results")
    st.dataframe(df.style.format({
        "OD": "{:.4f}",
        "OD Error": "{:.4f}",
        "Estimated Dose [cGy]": "{:.2f}"
    }))


    st.download_button("Download Measurement Results (CSV)",
                       data=df.to_csv(index=False).encode(),
                       file_name="measurement_results.csv",
                       mime="text/csv")











st.markdown(
    """
    <div style='text-align: right; font-size: 0.85em; color: gray; margin-top: 2em;'>
        Developed by A. M. Ferrara - alessandromichele.ferrara@gmail.com - &nbsp;|&nbsp; © 2025
    </div>
    """,
    unsafe_allow_html=True
)

