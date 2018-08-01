Package HDclust for clustering high dimensional data with Hidden Markov model on Variable Blocks (HMM-VB) in C.
For reference see Lin Lin and Jia Li, "Clustering with hidden Markov model on variable blocks," 
Journal of Machine Learning Research, 18(110):1-49, 2017.

The following functionality is provided in the package:

1.) trainmaster_mpi - trains an HMM-VB with defined variable block structure and searches for variable block structure.
The code is parallelized using OpenMPI. Thus installed MPI is required to run it on multiple cores and/or nodes.
Example: mpirun -np 2 ./trainmaster_mpi

To see the manual for using trainmaster_mpi, type trainmaster_mpi without any argument.

Here we describe the most important arguments. 

-i [input data filename]
The input data should be stored in a text file. The data should be in the typical ascii format: each row contains variables for one data point.
The path to the file should be specified after argument -i.
Example: mpirun -np 2 ./trainmaster_mpi -i data.txt

-b [input variable block structure filename]
If provided, a variable block structure should be stored in a text file. The file has the following format:
1st line: "dimensionality of the data" "number of variable blocks"
2nd line: "array with dimensionality of each variable block"
3rd line: "array with the number of mixture components in each variable block" 
subsequent lines: "variable ordering in each variable block"

Example: Below we show an example for the data with dimension 40. Variable block structure
contains 2 variable blocks with 10 and 30 variables correspondingly. First block has 3 mixture components,
second block has 5 mixture components. Variables have natural ordering:
40 2
10 30
3 5
0 1 2 3 4 5 6 7 8 9
10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39

The path to the file with variable block structure should be specified after argument -b.
Example: mpirun -np 2 ./trainmaster_mpi -i data.txt -b blockpar.txt

-v
This is a flag. If not flagged, non-diagonal covariance is assumed. If flagged, diagonal covariance.
Example: mpirun -np 2 ./trainmaster_mpi -i data.txt -b blockpar.txt -v

-m [output model filename]
Parameters of the trained HMM-VB will be stored in the binary file.
Example: mpirun -np 2 ./trainmaster_mpi -i data.txt -b blockpar.txt -m model.bin

-d
If a variable block structure is not provided, a greedy search algorithm trying to minimize BIC will be performed to 
find the variable block structure. In this case data dimensionality should be provided after argument -d:
Example: mpirun -np 2 ./trainmaster_mpi -i data.txt -d 10 -m model.bin

2.) testsync_mpi - performs clustering of data using a trained HMM-VB
The code is parallelized using OpenMPI. Thus installed MPI is required to run it on multiple cores and/or nodes.
Example: mpirun -np 2 ./testsync_mpi

To see the manual for using testsync_mpi, type testsync_mpi without any argument.

Here we describe the most important arguments. 

-i [input data filename]
The input data should be stored in a text file. The data should be in the typical ascii format: each row contains variables for one data point.
The path to the file should be specified after argument -i.
Example: mpirun -np 2 ./testsync_mpi -i data.txt

-m [input model filename] 
The input model should be stored in a binary file. It is obtained after running trainmaster_mpi
Example: mpirun -np 2 ./testsync_mpi -i data.txt -m model.bin

-o [output cluster result filename]
Text file containing results of the clustering. Each row contains the cluster label of one data point.
Example: mpirun -np 2 ./testsync_mpi -i data.txt -m model.bin -o clustering.txt

-v
This is a flag. If not flagged, non-diagonal covariance is assumed. If flagged, diagonal covariance.
The flag should be the same as for the trained HMM-VB.
Example: mpirun -np 2 ./testsync_mpi -i data.txt -m model.bin -o clustering.txt -v

A user can control the clustering results with the following arguments. following control  cluster

-l [minimum cluster size]
Clusters with sizes lower than the value are merged with the closest big clusters according to modes. Default value is 1.
Example: mpirun -np 2 ./testsync_mpi -i data.txt -m model.bin -o clustering.txt -l 10

-t [identical mode threshold]
The larger the threshold, the less stringent to declare two modes being identical.
Example: mpirun -np 2 ./testsync_mpi -i data.txt -m model.bin -o clustering.txt -t 0.05


3.) ridgeline_md - compute separability between clusters based on ridgelines.

To see the manual for using ridgeline_md, type ridgeline_md without any argument.
