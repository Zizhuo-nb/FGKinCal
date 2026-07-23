from factor_graph.src.FactorGraphOptimizerCubicSpline import FactorGraphOptimizerCubicSpline
from factor_graph.src.FactorGraphOptimizer import FactorGraph
import click

@click.command()
@click.option("--parent_dir", "-pa", default="/mnt/syn180/241111_FieldPheno4D_multi_crop_multi_modal/01_cropplotdata/New_structure", type=str, help="Path to the dataset directory")
@click.option("--output_dir", "-pb", default="output/", type=str, help="Path to the output data directory")
@click.option("--calibration_dir", "-pc", default="input/calibration/", type=str, help="Path to the static calibration of the laser scanners")
@click.option("--configfile", "-pd", default="config/kin_calibration_config.json", type=str, help="Config file of the kinematic calibration")
@click.option("--plot_id", "-pe", default="P144", type=str, help="Plot id to process")
@click.option("--date", "-pf", default="230516", type=str, help="Plot id to process")

def main(parent_dir,
         output_dir,
         calibration_dir,
         configfile,
         plot_id,
         date):
    

    factorgraph = FactorGraphOptimizerCubicSpline(parent_dir,
                              output_dir,
                              calibration_dir,
                              configfile,
                              plot_id,
                              date)
    factorgraph.run()

    
if __name__ == "__main__":
    main()




