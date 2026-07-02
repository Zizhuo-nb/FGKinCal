import json
from rich.console import Console
from rich.table import Table
from colorama import init, Fore, Style
import numpy as np
import time

class sICPconfig:

    window_size: float
    step_size: float
    max_iterations: int
    voxelization_use: bool
    voxel_size: float
    max_dist: float
    normals_radius = float
    normals_minpoints = int
    normals_maxpoints = int
    normals_maxroughness = float
    normal_angle_use = bool
    normal_angle_max = float
    mad_use = bool
    txyz: np.ndarray

    def __init__(self):

        # IntervalConfig (default parameters)
        self.window_size = 0.20
        self.step_size = 0.02
        
        # ICPConfig (default parameters)
        self.max_iterations = 50
        self.convergence_threshold = 0.001
        self.voxelization_use = True
        self.voxel_size = 0.06
        self.max_dist = 0.06
        self.normals_radius = 0.01
        self.normals_minpoints = 10
        self.normals_maxpoints = 1000
        self.max_roughness = 0.005
        self.roughness_filter_use = True
        self.normal_angle_use = True
        self.normal_angle_max = 40
        self.mad_use = True

        # Georeferencing Config
        self.txyz = np.array([0, 0, 0])

    def readfromjson(self, fname):

        t = time.strftime("%H:%M:%S")
        print(( f"| {Style.BRIGHT}{Fore.GREEN}{t + ' Read config file from folder'}{Style.RESET_ALL}" ))
        print("| - path: ", fname)

        with open(fname, 'r') as file:
            config = json.load(file)

        self.window_size = config["IntervalConfig"].get("windowsize")
        self.step_size = config["IntervalConfig"].get("stepsize")
        
        # ICPConfig
        self.max_iterations = config["ICPConfig"].get("maxiterations")  
        self.convergence_threshold = config["ICPConfig"].get("convergence_threshold")
        self.voxelization_use = config["ICPConfig"]["Voxelization"].get("use")
        self.voxel_size = config["ICPConfig"]["Voxelization"].get("voxelsize")
        self.max_dist = config["ICPConfig"]["Matching"].get("maxdist")
        self.normals_radius = config["ICPConfig"]["Normals"].get("radius")
        self.normals_minpoints = config["ICPConfig"]["Normals"].get("minpoints")
        self.normals_maxpoints = config["ICPConfig"]["Normals"].get("maxpoints")
        self.roughness_filter_use = config["ICPConfig"]["Rejection"]["roughness"].get("use")
        self.max_roughness = config["ICPConfig"]["Rejection"]["roughness"].get("max_roughness")
        self.normal_angle_use = config["ICPConfig"]["Rejection"]["normalangle"].get("use")
        self.normal_angle_max = config["ICPConfig"]["Rejection"]["normalangle"].get("maxangle")
        self.mad_use = config["ICPConfig"]["Rejection"]["MAD"].get("use")

        self.txyz = np.array( [config["GeorefConfig"]["globaloffset"].get("x"),
                               config["GeorefConfig"]["globaloffset"].get("y"),
                               config["GeorefConfig"]["globaloffset"].get("z")] )

        self.display_config()

    def writeToJson(self, fname):
        # Construct the config dictionary with current attribute values
        config = {
            "IntervalConfig": {
                "windowsize": self.window_size,
                "stepsize": self.step_size
            },
            "ICPConfig": {
                "maxiterations": self.max_iterations,
                "convergence_threshold": self.convergence_threshold,
                "Voxelization": {
                    "use": self.voxelization_use,
                    "voxelsize": self.voxel_size
                },
                "Matching": {
                    "maxdist": self.max_dist
                },
                "Normals": {
                    "radius": self.normals_radius,
                    "minpoints": self.normals_minpoints,
                    "maxpoints": self.normals_maxpoints
                },
                "Rejection": {
                    "roughness": {
                        "use": self.roughness_filter_use,
                        "max_roughness": self.max_roughness
                    },
                    "normalangle": {
                        "use": self.normal_angle_use,
                        "maxangle": self.normal_angle_max
                    },
                    "MAD": {
                        "use": self.mad_use
                    }
                }
            },
            "GeorefConfig": {
                "globaloffset": {
                    "x": float(self.txyz[0]),
                    "y": float(self.txyz[1]),
                    "z": float(self.txyz[2])
                }
            }
        }

        # Write the dictionary to JSON file
        with open(fname, 'w') as file:
            json.dump(config, file, indent=4)

        # Optional: print a confirmation
        print(f"Configuration saved to {fname}")


    def display_config(self):

        console = Console()
        table1 = Table()

        # Adding columns to the table
        table1.add_column("Interval Parameter       ", style="cyan")
        table1.add_column("Value       ", style="magenta")
        table1.add_row("Window Size", str(self.window_size))
        table1.add_row("Step Size", str(self.step_size))

        table2 = Table()
        table2.add_column("ICP Parameter            ", style="cyan")
        table2.add_column("Value       ", style="magenta")
        table2.add_row("Maximum Iterations", str(self.max_iterations))
        table2.add_row("Voxelization Use", str(self.voxelization_use))
        table2.add_row("Voxel Size", str(self.voxel_size))
        table2.add_row("Max Distance", str(self.max_dist))
        table2.add_row("Normals Radius", str(self.normals_radius))
        table2.add_row("Normals min points", str(self.normals_minpoints))
        table2.add_row("Normals max points", str(self.normals_maxpoints))
        table2.add_row("Roughness Filter Use", str(self.roughness_filter_use))
        table2.add_row("Max Roughness", str(self.max_roughness))
        table2.add_row("Normal Angle Use", str(self.normal_angle_use))
        table2.add_row("Normal Angle Max", str(self.normal_angle_max))
        table2.add_row("MAD Use", str(self.mad_use))

        table3 = Table()
        table3.add_column("Georeferencing Parameter ", style="cyan")
        table3.add_column("Value       ", style="magenta")
        table3.add_row("Global Offset X", str(self.txyz[0]))
        table3.add_row("Global Offset Y", str(self.txyz[1]))
        table3.add_row("Global Offset Z", str(self.txyz[2]))


        # Display the table
        console.print(table1)
        console.print(table2)
        console.print(table3)