#!/bin/bash
#PBS -N transports
#PBS -P e14
#PBS -q normalbw
#PBS -l walltime=4:00:00
#PBS -l mem=128GB
#PBS -l software=netcdf
#PBS -l ncpus=6
#PBS -l storage=gdata/e14+gdata/v45+gdata/hh5+gdata/cj50+gdata/ik11+scratch/e14
#PBS -j oe
#PBS -v month,year

## Note, run this with:
## qsub -v month=1,year=2150 submit_transport_job.sh

# Also, note that ncpu = 6 is best. It can nearly run with ncpu = 4, but 1 in 10 runs or so will fail then.

## I/O filenames
# this reads the name of the current run directory to use for output etc:
#script_dir=/home/157/akm157/data_processing/cross_slope_transports/
script_dir=/home/561/hd4873/project3/cross_slope_transport/
cd $script_dir
#output_dir=/g/data/v45/akm157/model_data/access-om2/01deg_jra55v140_iaf_cycle3/vhrho_binned/

# load conda
module use /g/data3/hh5/public/modules
module load conda/analysis3-23.04

# run
python3 save_transports_along_contour.py $month $year &>> output_${month}.out

