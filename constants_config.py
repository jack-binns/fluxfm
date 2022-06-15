"""
Constants:
"""
# Detector
EIGER_nx = 1062         # Pixel numbers for EIGER detector
EIGER_ny = 1028
MAX_PX_COUNT = 4.29E9   # Hot pixel value
MASK_MAX = 1e12

# Experiment geometry
cam_length = 0.1512     # Detector-to-sample distance in meters
wavelength = 0.67018E-10    # Incident X-ray wavelength in meters
pix_size = 75E-6            # Pixel size in meters

# Data path variables
experiment_id = 17635       # Experiment ID
maia_num = 86207            # Starting value for the MAIA detector
beamtime_data_path = f'/data/xfm/{experiment_id}/raw/eiger/'
beamtime_analysis_path = f'/data/xfm/{experiment_id}/analysis/eiger/SAXS/'