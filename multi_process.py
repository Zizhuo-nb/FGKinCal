import csv
import subprocess

csv_file = "fieldpheno4d_meta.csv"

"""
This script can be used to run the kinematic calibration method for the whole or parts of the "FieldPheno4D" datasets. 
It reads the metadata table "fieldpheno4d_meta.csv" of the dataset and filters not processed crop plots NOT containing a "x" at the end of the dates within the metadata files which should be processed.
Finally it sequential runs the "main.py" script with the input datasets to be processed with the plot_id and date intput argument.
"""

with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        # skip row starting with "#"
        if row[0][0] == "#":
            continue

        # get plot id
        plot_id = row[0].strip()

        # get dates
        if len(row) == 2 and "," in row[1]:
            dates = [d.strip() for d in row[1].split(",") if d.strip()]
        else:
            dates = [col.strip() for col in row[1:] if col.strip()]

        # exclude any date containing the letter "x" (case-insensitive)
        dates = [d for d in dates if 'x' not in d.lower()]

        # Execute main script
        for j in range(len(dates)):

            cmd = ["python3", "main.py", "--plot_id", plot_id, "--date", dates[j]]
            try:
                subprocess.run(cmd, check=True)
                print(f"Executed: python3 main.py --plot_id {plot_id} --date {dates[j]}")
            except subprocess.CalledProcessError as e:
                print(f"Error executing: python3 main.py --plot_id {plot_id} --date {dates[j]}")
                print(e)

                