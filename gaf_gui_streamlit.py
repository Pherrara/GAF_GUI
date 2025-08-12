import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import math
import re
import matplotlib.pyplot as plt
from io import BytesIO

from pathlib import Path




if "calibration_runs" not in st.session_state:
    st.session_state.calibration_runs = []

# --- Title and Setup ---
st.set_page_config(layout="wide")
st.title("GAF Film Calibration GUI")

# --- Sidebar Controls ---
st.sidebar.header("Parameters")



# --- Channel selection (OpenCV loads as BGR) ---
channel_label = st.sidebar.radio(
    "Read channel", ["Red", "Green", "Blue"], index=0,
    help="The reading channel for GAF films depends on energy"
)
channel_code = {"Red": "R", "Green": "G", "Blue": "B"}[channel_label]



# --- Calibration Folder Selection (excluding hidden folders) ---
script_folder = os.getcwd()
all_folders = [
    f for f in os.listdir(script_folder)
    if os.path.isdir(os.path.join(script_folder, f)) and not f.startswith(".")
]
all_folders.sort()

folder_path = st.sidebar.selectbox("Select calibration folder:", [""] + all_folders)



measure_folder = st.sidebar.selectbox("Select measurement folder:", [""] + all_folders)




name_irradiated = st.sidebar.text_input("Prefix for irradiated films:", value="irr")
name_velo = st.sidebar.text_input("Prefix for unirradiated films:", value="velo")
image_extension = st.sidebar.text_input("Image extension (e.g., .tif):", value=".tif")

# x_values_file = st.sidebar.text_input("MU values file (MU):", value="MU_values.dat")
uploaded_mu_file = st.sidebar.file_uploader(
    "MU values file (MU):",
    type=["dat", "txt", "csv"],
    help="A file with the MU irradiated for each film, one value per line, in order. Default is MU_values.dat in the app folder. Upload another if needed."
)

if uploaded_mu_file is not None:
    app_dir = Path(__file__).resolve().parent
    dest_path = app_dir / uploaded_mu_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_mu_file.getbuffer())
    x_values_file = str(dest_path)  # <-- string path to the saved file
else:
    x_values_file = str(Path(__file__).resolve().parent / "MU_values.dat")


# Check and display info about MU values file
if os.path.isfile(x_values_file):
    try:
        with open(x_values_file, "r") as f:
            lines = [line for line in f if line.strip()]
            num_lines = len(lines)
        st.sidebar.markdown(f"✅ `{x_values_file}`found  with **{num_lines} values**.")
    except Exception as e:
        st.sidebar.error(f"⚠️ Failed to read `{x_values_file}`: {e}. It is necessary for calibration.")
else:
    st.sidebar.warning(f"❌ MU values file `{x_values_file}` not found. It is necessary for calibration.")



# Read dose conversion coefficient from file
# File uploader for DCC
uploaded_dcc_file = st.sidebar.file_uploader(
    "Dose Conversion Coefficient file (dcc.dat):",
    type=["dat", "txt", "csv"],
    help="This is how much cGy are actually received at the films geometry when using 100 MU. Default is dcc.dat in the app folder. Upload another if needed."
)

if uploaded_dcc_file is not None:
    app_dir = Path(__file__).resolve().parent
    dest_path = app_dir / uploaded_dcc_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_dcc_file.getbuffer())
    dcc_file = str(dest_path)  # string path to saved file
else:
    dcc_file = str(Path(__file__).resolve().parent / "dcc.dat")

# Read and validate the Dose Conversion Coefficient
if os.path.isfile(dcc_file):
    try:
        with open(dcc_file, "r") as f:
            dose_conversion_coefficient = float(f.read().strip())
        st.sidebar.markdown(
            f"✅ `{dcc_file}` found — **Dose Conversion Coefficient:** `{dose_conversion_coefficient:.3f} cGy`"
        )
    except Exception as e:
        st.sidebar.error(
            f"⚠️ Failed to read `{dcc_file}`: {e}. "
            "It is necessary for calibration. Defaulting to 100 cGy."
        )
        dose_conversion_coefficient = 100.0
else:
    st.sidebar.warning(
        f"❌ DCC file `{dcc_file}` not found. "
        "It is necessary for calibration. Defaulting to 100 cGy."
    )
    dose_conversion_coefficient = 100.0


# Thresholding
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

def _extract_channel(image: np.ndarray, channel: str) -> np.ndarray:
    """
    Return the selected channel from a BGR image.
    If image is already single-channel, return as-is.
    channel in {'R','G','B'}.
    """
    if image.ndim == 2:
        return image
    idx = {"B": 0, "G": 1, "R": 2}[channel]
    return image[:, :, idx]

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


def final_segmentation(image, tolerance_percent, green_square, channel):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    green_pixels = gray_image[green_square == 255]
    median_val = np.median(green_pixels)
    tolerance = median_val * (tolerance_percent / 100)
    lower_bound = median_val - tolerance
    upper_bound = median_val + tolerance
    within_range_mask = ((gray_image >= lower_bound) & (gray_image <= upper_bound)).astype(np.uint8) * 255
    final_mask = cv2.bitwise_and(within_range_mask, green_square)

    # Work on the same, user-selected channel
    ch = _extract_channel(image, channel)
    selected_pixels = ch[final_mask == 255]
    mean_val = np.mean(selected_pixels)
    std_dev_val = np.std(selected_pixels)
    return median_val, mean_val, std_dev_val, final_mask

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

def write_dat_file(mean_values, stdev_values):
    from io import StringIO
    data = np.column_stack((mean_values, stdev_values))
    buffer = StringIO()
    header = "Mean\tStdDev"
    np.savetxt(buffer, data, delimiter='\t', header=header, comments='', fmt='%.6f')
    return buffer.getvalue()

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

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Calibration')
    return output.getvalue()


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
    if folder_path == "": # check for calibration folder
        st.error("❌ You must select a calibration folder before running calibration.")
        st.session_state.run_calibration = False
    elif not os.path.isdir(folder_path):
        st.error(f"❌ The selected calibration folder '{folder_path}' does not exist.")
        st.session_state.run_calibration = False
    elif not os.path.isfile(x_values_file):
        st.error(f"❌ MU values file not found: `{x_values_file}`. Please upload or place it in the app folder.")
        st.session_state.run_calibration = False
    elif not os.path.isfile(dcc_file):
        st.error(f"⚠️ Dose Conversion Coefficient (DCC) file not found: `{dcc_file}`. Please upload or place it in the app folder. Defaulting to 100 cGy")
        st.session_state.run_calibration = True
    else: # or else start
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
            _, mean_irr, std_irr, final_mask_irr = final_segmentation(img_irr, tolerance_percent, mask_irr, channel_code)
            mean_values_irr.append(mean_irr)
            stdev_values_irr.append(std_irr)

            if show_final_seg:
                st.pyplot(display_overlay(img_irr, final_mask_irr, f"Final segmentation: {name_irr}"))
            
            

            img_vel, mask_vel = load_image_and_segment_white(path_vel, white_threshold_percent)
            _, mean_vel, std_vel, final_mask_vel = final_segmentation(img_vel, tolerance_percent, mask_vel, channel_code)
            mean_values_velo.append(mean_vel)
            stdev_values_velo.append(std_vel)

            if show_final_seg_velo:
                st.pyplot(display_overlay(img_vel, final_mask_vel, f"Final segmentation: {name_vel}"))

        input_file = os.path.join(folder_path, 'readings_irradiated.dat')
        ref_file = os.path.join(folder_path, 'readings_velo.dat')

        #write_dat_file(np.array(mean_values_irr), np.array(stdev_values_irr), input_file)
        #write_dat_file(np.array(mean_values_velo), np.array(stdev_values_velo), ref_file)

        reading, reading_err = np.array(mean_values_irr), np.array(stdev_values_irr)
        reference, reference_err = np.array(mean_values_velo), np.array(stdev_values_velo)
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
        ax.set_title(f'Polynomial Fit: Dose vs OD ({folder_path})')
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

        calib_filename = f"{os.path.basename(folder_path.strip(os.sep))}.dat"


        if st.button(f"Save {calib_filename} to script folder"):
            st.session_state.save_calib_clicked = True

        if st.session_state.get("save_calib_clicked", False):
            try:
                calib_filename = f"{os.path.basename(folder_path.strip(os.sep))}.dat"
                path = os.path.join(os.getcwd(), calib_filename)
                with open(path, "w") as f:
                    for c in coeffs[::-1]:
                        f.write(f"{c:.8e}\n")
                st.success(f"`{calib_filename}` saved to: `{path}`")
            except Exception as e:
                st.error(f"Failed to write {calib_filename}: {e}")
            finally:
                st.session_state.save_calib_clicked = False  # Reset after handling



        

        # --- Comparison of saved calibrations ---
        if len(st.session_state.calibration_runs) > 1:
            st.subheader("Compare Saved Calibration Runs")
        
            selected_labels = st.multiselect(
                "Select runs to compare:",
                options=[run["label"] for run in st.session_state.calibration_runs],
                default=[run["label"] for run in st.session_state.calibration_runs]
            )
        
            fig_compare, ax = plt.subplots(figsize=(10, 6), dpi = 300)

        
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
            st.pyplot(fig_compare)

            buf_compare = BytesIO()
            fig_compare.savefig(buf_compare, format="png")
            st.download_button("Download Comparison Plot", data=buf_compare.getvalue(), file_name="comparison_plot.png", mime="image/png")

                                   





        st.subheader("Downloads")

        # Save plot button
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("Download Plot as PNG", data=buf.getvalue(), file_name=f"dose_fit_plot_{calib_filename}.png", mime="image/png")

        # Optional: download as CSV
        csv_bytes = comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Comparison Table (CSV)", data=csv_bytes, file_name=f"dose_comparison_{calib_filename}.csv", mime="text/csv")

        excel_data = to_excel_bytes(comparison_df)
        st.download_button(
            label="Download Comparison Table (Excel .xlsx)",
            data=excel_data,
            file_name=f"dose_comparison_{calib_filename}_.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


        # Export calibration coefficients
        calibration_txt = "\n".join([f"{c:.8e}" for c in coeffs[::-1]])  # lowest order first
        st.download_button(f"Download Calibration Coefficients",
                   data=calibration_txt,
                   file_name=calib_filename,
                   mime="text/plain")



        # Download from memory, do not save to disk
        st.download_button(
            "Download Irradiated Data (.dat)",
            data=write_dat_file(mean_values_irr, stdev_values_irr),
            file_name=f"readings_irradiated_{calib_filename}.dat",
            mime="text/plain"
        )

        st.download_button(
            "Download Unirradiated Data (.dat)",
            data=write_dat_file(mean_values_velo, stdev_values_velo),
            file_name=f"readings_velo_{calib_filename}.dat",
            mime="text/plain"
        )







# --- Calibration selection for measurement ---
st.subheader("Make a Measurement Using a Previous Calibration")

# --- Check for available calibration files in the current folder ---
script_folder = os.getcwd()
calib_dat_files = [
    f for f in os.listdir(script_folder)
    if f.lower().endswith(".dat") and "calibration" in f.lower()
]
file_in_script_folder = len(calib_dat_files) > 0

calib_options = [
    "Upload calibration.dat",
    "Use saved calibration run",
    "Use calibration.dat from script folder"
]

default_selection = 2 if file_in_script_folder else 0  # prefer script folder if files found

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
        st.error("No calibration `.dat` files found in the script folder.")
    else:
        selected_file = st.selectbox("Select a calibration file:", calib_dat_files)
        selected_path = os.path.join(script_folder, selected_file)
        try:
            coeffs = np.loadtxt(selected_path)
            poly_fit = np.poly1d(coeffs[::-1])
            st.success(f"Loaded calibration from `{selected_file}`.")
        except Exception as e:
            st.error(f"Failed to load `{selected_file}`: {e}")



# Now make a measurement

if st.button("Make Measurement"):
    if measure_folder.strip() == "":
        st.error("❌ You must select a measurement folder before running measurement.")
        st.session_state.make_measurement = False
    elif not os.path.isdir(measure_folder):
        st.error(f"❌ The selected measurement folder '{measure_folder}' does not exist.")
        st.session_state.make_measurement = False
    else:
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
        _, mean_i, std_i, _ = final_segmentation(img_irr, tolerance_percent, mask_irr, channel_code)
        mean_irr.append(mean_i)
        std_irr.append(std_i)

        img_vel, mask_vel = load_image_and_segment_white(path_vel, white_threshold_percent)
        _, mean_v, std_v, _ = final_segmentation(img_vel, tolerance_percent, mask_vel, channel_code)
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


    meas_filename = f"{os.path.basename(measure_folder)}_results.csv"
    st.download_button("Download Measurement Results (.csv)",
                    data=df.to_csv(index=False).encode(),
                    file_name=meas_filename,
                    mime="text/csv")

    excel_data_meas = to_excel_bytes(df)
    st.download_button(
        label="Download Measurement Results (Excel .xlsx)",
        data=excel_data_meas,
        file_name=meas_filename.replace(".csv", ".xlsx"),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )













st.markdown(
    """
    <div style='text-align: right; font-size: 0.85em; color: gray; margin-top: 2em;'>
        Developed by A. M. Ferrara - alessandromichele.ferrara@gmail.com &nbsp;|&nbsp; v1.2.0 Aug 2025
    </div>
    """,
    unsafe_allow_html=True
)

