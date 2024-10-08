import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u


# --- Interpolation of a step-kind arrays --- #
def local_interpolation(arr, N=5):
    """
    
    """
    interpolated_arr = np.copy(arr)
    step_changes = np.where(np.diff(arr) != 0)[0]
    for step in step_changes:
        start = max(0, step - N // 2)
        end = min(len(arr) - 1, step + N // 2 + 1)
        x = np.arange(start, end + 1)
        interpolated_arr[start:end+1] = np.linspace(arr[start], arr[end], len(x))
    return interpolated_arr



# --- Reading magic schedule and returning a dictionary of schedule --- #
def read_magic_schedule(file_schedule, ref_day_str, t_night_ref, t_midnight):
    # --- Open the schedule txt file content --- #
    with open(file_schedule, "r") as file:
        file_raw_content = file.read()

    # Extracting the lines and selecting the ones that are relevant
    # We exclude some keywords such as T-points or interferometry
    lines_filter = []
    for line in file_raw_content.split("\n"):
        flag_filter = True
        for exc in str_filter_lines:
            if exc in line:
                flag_filter = False
        if flag_filter and line not in ["", "#"]:
            lines_filter.append(line)

    # Then we create a source dictionary with:
    # - Source names
    # - Coordinate
    source_dict = {}
    for line in lines_filter:
        if line[0] == "#":
            index_date = line.find(":") - 2
            key = line[2:index_date]
            for rep in str_filter_source_name:
                key = key.replace(rep, "")
            source_dict[key] = SkyCoord(line[index_date:], unit=(u.hourangle, u.deg))

    # Then we exclude other commented lines
    lines_filter = [l for l in lines_filter if l[0] != "#"]

    # Then we create a night schedule dictionary with the following information:
    # - Source name
    # - Start times []
    # - Stop times []
    # - Total obs time s

    dict_schedule_night = {} 
    flag_date_line = False
    for l in lines_filter:
        if "Date" in l and ref_day_str in l:
            flag_date_line = True
        elif "Date" in l and ref_day_str not in l:
            flag_date_line = False

        # Taking only the desired day lines on the schedule
        if flag_date_line and "Date" not in l:
            # print(l)

            # Array indexes for source name finding
            index_start_name = l.find(" - ") + 10
            index_stop_name = l.find("(") - 1     

            # Raw key or source name, with some extensions we want to extract
            key = l[index_start_name:index_stop_name]
            # Excluding 
            for rep in str_filter_source_name:
                key = key.replace(rep, "")

            # Finding index where hours of observations are located
            index_hours = l.find(" - ")
            str_tstart = l[index_hours - 5:index_hours]
            str_tstop = l[index_hours + 3:index_hours + 8]

            # Calculating the day-month-year of observation
            # For starting time
            if float(str_tstart[:2]) > 12.0: # If the hour is over 12 then the night reference
                day_start = f"{t_night_ref.year}-{t_night_ref.month}-{t_night_ref.day}"
            else: # If the hour is less than 12 means is the midnight reference
                day_start = f"{t_midnight.year}-{t_midnight.month}-{t_midnight.day}"
            # For stoping time
            if float(str_tstop[:2]) > 12.0:
                day_stop = f"{t_night_ref.year}-{t_night_ref.month}-{t_night_ref.day}"
            else:
                day_stop = f"{t_midnight.year}-{t_midnight.month}-{t_midnight.day}"

            # Then we compute the time of start and finish of observations
            tstart = Time(day_start + " " +  str_tstart + ":00", format="iso")        
            tstop  = Time(day_stop  + " " +  str_tstop  + ":00", format="iso")

            # We build the dictionary
            # If source not in dictionary
            if dict_schedule_night.get(key) == None: 
                dict_schedule_night[key] = {
                    "coord" : source_dict[key],
                    "tstart" : [tstart],
                    "tstop" : [tstop],
                    "tobs" : (tstop - tstart).to_value("second"),
                }
            # If source already in dictionary
            # means that source is observed twice in a night
            # so we only add the times
            else: 
                dict_schedule_night[key]["tstart"].append(tstart)
                dict_schedule_night[key]["tstop"].append(tstop)

    return dict_schedule_night
                

# --- Defining a star marker --- #
marker_x, marker_y = [], []
for i, ix, iy in zip([0, 1, 2, 3], [-1, -1, 1, 1], [-1, 1, 1, -1]):
    th = np.linspace((i + 1) * np.pi/2, i * np.pi/2)
    x, y = np.sin(th), np.cos(th)
    marker_x = [*marker_x, *(x + ix)]
    marker_y = [*marker_y, *(y + iy)]
marker_star = [(x, y) for x, y in zip(marker_x, marker_y)]


# List of strings to lines that contain them to be excluded 
str_filter_lines = [
    "Cat.01:", "Cat.02:", "Cat.03:", "Cat.04:", "Cat.05:", "Cat.06:",
    "PERIOD", "Categories", "Repositioning", "T-Points", "Interferometry"
]

# List of strings to exclude from source names
str_filter_source_name = [
    " ", "_Mon", "_zd60-70L3", "_zd0_35L3", "_zd35-50L3", "_zd50-60L3",
    "_Muons_Tech", "_Fix", "_updated", "Moon", "_ToO", "_MWL",
]