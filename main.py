from src.core.KinematicCalibration import KinematicCalibration

from colorama import init, Fore, Style
import click
import shutil
import os
import glob

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
    
    # Directory of the current dataset
    dataset_dir = os.path.join(parent_dir, plot_id, date)
    
    #####################################################################################
    # 2) Initialization and data preparation

    # Initialize 
    '''
    第一步，对程序进行初始化。
    输入参数包括：
    - parent_dir：数据集的父目录路径。
    - output_dir：输出数据的目录路径。
    - calibration_dir：激光扫描仪的静态校准文件路径。
    - configfile：运动校准的配置文件路径。
这些参数将用于后续的数据处理和校准步骤。
    这个对象后面会保存：
    -配置文件
    -左右扫描仪轨迹
    -左右扫描仪原始激光数据
    -左右静态外参
    -ICP 结果
    -动态外参结果
    '''
    kin_cal = KinematicCalibration( parent_dir,
                                    output_dir,
                                    calibration_dir,
                                    configfile )
    
    # Copy data from dataset path to local repo
    '''
    第二步，复制准备数据。
    输出会出现当前plot对应的数据文件，包括左右scanner的轨迹文件、激光数据文件和静态外参文件。
    '''
    kin_cal.copy_data( plot_id, date )
    
    # Print dataset info
    kin_cal.print_info()

    # Load kinematic calibration config file
    '''
    第三步，加载配置文件。
    具体见json文件。
    icp的窗口切分，点云处理都依赖这个配置文件。
    '''
    kin_cal.loadconfig()

    # Load static calibration from path
    '''
    第四步，加载静态校准参数。
    就是那些XML文件，描述的是左右scanner相当于激光扫描仪的坐标系的外参关系。
    '''
    kin_cal.loadcalibration()

    # Load data from path
    '''
    第五步，读取轨迹和激光数据。
    输入就是左右的 .trj 以及左右的 .bin文件。
    程序内部的：
    -self.TL        左 scanner 对应的轨迹
    -self.TR        右 scanner 对应的轨迹
    -self.lmidataL  左 scanner 原始 profile 数据
    -self.lmidataR  右 scanner 原始 profile 数据

    到这里位置和原始数据都准备好了，后续就可以进行校准了。
    '''
    kin_cal.loaddata()
    
# #     Create initial point cloud with static calibration parameter (optional)
#     pcl, pcr = kin_cal.create_pointcloud( calibration = "static" )
#     pc_s = pcl.concatenate(pcr)

#     pc_s.write_to_file( path = kin_cal.output_dir, filename = "pc_static_calibration", offset = kin_cal.config.txyz )

    pcl_static, pcr_static = kin_cal.create_pointcloud(calibration="static")

    pcl_static.write_to_file(
    path=kin_cal.output_dir,
    filename="pc_left_static_calibration",
    offset=kin_cal.config.txyz
)

    pcr_static.write_to_file(
    path=kin_cal.output_dir,
    filename="pc_right_static_calibration",
    offset=kin_cal.config.txyz
)
    #####################################################################################
    # 2) Kinematic calibration

    # Run alignment
    '''
    动态校准：
    1.首先是get_alignment_window()，根据配置文件中icp的窗口切分参数(window_size 与 step_size)，对左右scanner的轨迹进行切分，得到每个窗口对应的轨迹段。
    也就是比如窗口 1：第 0 m 到第 2 m，窗口 2：第 1 m 到第 3 m，窗口 3：第 2 m 到第 4 m... 每个窗口里面有一段左右scanner的数据。
    输出就是左右数据对应的窗口索引idxL与idxR，简单说就是第 k 个窗口应该取左 scanner 的哪一段数据/第 k 个窗口应该取右 scanner 的哪一段数据

    2.在run()里面，会调用direct georeferencing，输入包括每边的激光数据，轨迹数据（当前窗口）以及对应的静态外参，输出就是第k个窗口的两块局部点云。

    3.然后就是SymPlanePlaneICP.runICP()，输入就是：
    -当前窗口左 scanner 点云 pc_l_k
    -当前窗口右 scanner 点云 pc_r_k
    -ICP 初始值 x0
    -ICP 配置参数
    程序会尝试找到一个小的6DoF变换，让左右点云更好重合。输入包括xi_k（修正量）以及success_k （ICP是否成功）

    4.保存每个窗口的ICP结果，也在run()函数中，输入就是每个窗口的xi_k以及每个窗口对应的中心时间。输出是一个离散的修正序列，描述了每个窗口的修正量xi_1以及对应的时间time_1。
    到此为止只是得到了很多离散窗口的局部ICP修正量

    5.最后是compute_kinematic_calibration_parameter()，输入就是每个窗口的修正量xi_k以及对应的中心时间time_k还有左右scanner的静态外参calL, calR.
    程序把每个窗口的左右点云的对其误差，转换成左右scanner的外参修正（分配到左右scanner上面）
    输出就是离散的动态外参修正：
    -time_1 → left calibration update
    -time_1 → right calibration update

    -time_2 → left calibration update
    -time_2 → right calibration update

    -time_3 → left calibration update
    -time_3 → right calibration update...
    也就是每个窗口中心时刻会有一个 新的左右scanner外参

    6.对离散外参进行插值，得到连续的动态外参，也是在compute_kinematic_calibration_parameter()里面，输入就是离散的动态外参修正以及对应的时间，输出就是连续的动态外参修正函数，描述了每个时刻的左右scanner外参
    然后输出得到time_dependent calibration

    7.利用动态外参重新生成最终点云create_pointcloud( calibration = "kinematic" )，输入就是每个时刻的动态外参，输出就是每个时刻的点云，最后把所有时刻的点云合并成一个完整的点云


    说白了，就是：
    KinScanCal 先利用轨迹数据和静态外参，在每个局部窗口内生成左右扫描仪点云；然后对每个窗口的左右点云做 ICP，得到该窗口的相对 6DoF 对齐修正；
    接着把这些修正量转换为左右扫描仪在窗口中心时刻的外参修正；最后对这些离散时间上的外参修正进行插值，得到连续动态外参，并用它重新生成最终点云。
    '''
#     kin_cal.run()

#     # Compute kinematic calibration parameter
#     kin_cal.compute_kinematic_calibration_parameter()

#     #####################################################################################
#     # 3) Create final point clouds with the time-dependent calibration parameter

#     pcl, pcr = kin_cal.create_pointcloud( calibration = "kinematic" )

#     # Merge point clouds to one
#     pc = pcl.concatenate(pcr)

#     # Write point clouds to file
#     pc.write_to_file( path = kin_cal.output_dir, filename = "pc_kinematic_calibration", offset = kin_cal.config.txyz )

    #####################################################################################
    # 4) Copy files back to dataset folder structure (optional)
    
#     # 4.1 Calibration files
#     cal_files = glob.glob(os.path.join("output/", "*.txt"))

#     # Copy each file
#     for file_path in cal_files:
#         filename = os.path.basename(file_path)
#         destination_path = os.path.join(os.path.join(dataset_dir,"03_calibration"), filename)

#         try:
#             shutil.copy(file_path, destination_path)
#         except Exception as e:
#             continue

#     # 4.2 Point clouds
#     pc_files = glob.glob(os.path.join("output/", "*.las"))

#     # Copy each file
#     for file_path in pc_files:
#         filename = os.path.basename(file_path)
#         destination_path = os.path.join(os.path.join(dataset_dir,"04_pointcloud"), filename)
#         try:
#             shutil.copy(file_path, destination_path)
#         except Exception as e:
#             continue

if __name__ == "__main__":
    main()



'''
KinScanCal 动态校准整体流程总结

这个库的核心目标是：

利用已有轨迹、左右扫描仪原始 laser profile、左右静态外参，先在局部窗口内估计左右点云的相对错位，再把这些窗口级修正转换成随时间变化的动态外参，最后用动态外参重新生成点云。

原始轨迹 + 原始 laser profile + 静态外参
        ↓
按轨迹切分窗口
        ↓
每个窗口生成左右局部点云
        ↓
把局部点云转到窗口中心 body frame
        ↓
左右局部点云做 symmetric plane-to-plane ICP
        ↓
得到每个窗口中心时间的 ICP 修正 xi
        ↓
写入 Px.txt
        ↓
读取 Px.txt，把 xi 转成左右 scanner 动态外参
        ↓
首尾补边界
        ↓
Hermite/Huber 样条插值到完整 trajectory timestamps
        ↓
写出动态外参文件
        ↓
用动态外参重新 direct georeferencing 生成最终点云


'''


# ''''
# python .\static_cloud_generate.py  --parent_dir "F:\UNIVERSITY_BONN\master_thesis\working_space\KinScanCal_data\" --output_dir "./output/" --calibration_dir "F:\UNIVERSITY_BONN\master_thesis\working_space\KinScanCal_data\P151\230821\03_calibration\" --configfile ".\config\kin_calibration_config.json" --plot_id P151 --date 230821
# '''