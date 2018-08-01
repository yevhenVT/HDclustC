/*
Copyright (C) [Dec 2015]-[April 2017] Jia Li, Department of Statistics, 
The Pennsylvania State University, USA (<jiali@stat.psu.edu>)- All Rights Reserved  

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#include "hmm_mpi.h"
#include <mpi.h>
#include <string.h>

void printmanual()
{
  int i;

  fprintf(stdout, "---------------- Usage manual for trainmaster_mpi ---------------------\n");
  fprintf(stdout, "To see the usage manual, use trainmaster without any argument.\n\n");
  fprintf(stdout, "Syntax: trainmaster_mpi -i [...] -m [...] -b [...] ......\n\n");
  fprintf(stdout, "-i [input data filename]\n");
  fprintf(stdout, "   data file is in a typical ascii matrix format: each row contains variables for one data point,\n");
  fprintf(stdout, "   each column in a row is the value of one variable. \n\n");
  fprintf(stdout, "-m [output model filename], the output model is a binary file\n\n");
  fprintf(stdout, "-b [input variable block structure filename], this is a parameter ascii file that follows strict format\n");
  fprintf(stdout, "   This file specifies variable block structure including partion and ordering. When it is given, the block\n");
  fprintf(stdout, "   structure will not be searched.\n");
  fprintf(stdout, "   If this file is given, the data dimension will be read from the file and does not need command-line\n");
  fprintf(stdout, "   input by the \"-d\" option.\n");
  fprintf(stdout, "   If this file is given, the \"-p\" and \"-o\" options will be ignored.\n\n");
  fprintf(stdout, "-d [data dimension], this has to be specified if \"-b\" option is not given.\n\n");
  fprintf(stdout, "-p [input parameter filename], this is a parameter ascii file that specifies the number of components given\n");
  fprintf(stdout, "   the dimension of a variable block. The same block dimension is assigned with one value of #components.\n");
  fprintf(stdout, "   If not given, the program will choose #components by a fixed scheme.\n\n");
  fprintf(stdout, "-o [input parameter filename], this is an ascii file specifying the order of the variables.\n");
  fprintf(stdout, "   Each ordering of variables is in one line. There can be multiple lines giving multiple ways of ordering.\n");
  fprintf(stdout, "   For data with D dimensions, the variable identification numbers should be 0, 1, ..., D-1. \n");
  fprintf(stdout, "   This file is optional. If not given, the program will generate random permutation of variables.\n\n");
  fprintf(stdout, "-e [floating number], this determines the precision of convergence. It can be skipped and a default value will be used.\n\n");
  fprintf(stdout, "-n [number of permutations], number of permutations to be used when searching for variable block structure.\n");
  fprintf(stdout, "   The default value is 1 or equals the number of permutations given in the \"-o\" option file.\n\n");
  fprintf(stdout, "-x [maximum variable block dimension], the default is 10.\n\n");
  fprintf(stdout, "-y This is a flag. If not flagged, the minimum block dimension is 1 if variable block search is performed. \n");
  fprintf(stdout, "   If flagged, the minimum block dimension is 2.\n\n");
  fprintf(stdout, "-0 This is flag (note: zero, not capital o). If not flagged, k-means initialization will always be performed.\n");
  fprintf(stdout, "   If flagged, k-means initialization will be skipped. \n\n");
  fprintf(stdout, "-1 [number of times for scheme 1 initialization], default is 0.\n\n");
  fprintf(stdout, "-2 [number of times for scheme 2 initialization], default is 0.\n\n");
  fprintf(stdout, "-r [integer as random seed for scheme 1 and 2 initialization], default is 0.\n\n");
  fprintf(stdout, "-s This is a flag. If not flagged, when performing search for block variable structure, more striction is imposed on \n");
  fprintf(stdout, "   the formation of the blocks. If flagged, the search range is expanded, and the search algorithm is that described \n");
  fprintf(stdout, "   in the paper.\n\n");
  fprintf(stdout, "-v This is a flag. If not flagged, non-diagonal covariance is assumed. If flagged, diagonal covariance.\n");
  fprintf(stdout, "-t This is a flag. If flagged, the final trained model will be printed on stdout.\n\n");
  fprintf(stdout, "----------------------------------------------------------------------\n");
}

int main(argc, argv)
     int argc;
     char *argv[];
{
  char infilename[300]; // input file with data
  char mdfilename[300]; // output file 
  char bparfilename[300]; // file with variable block structure
  char parfilename[300];
  char permfilename[300];
  FILE *infile, *mdfile, *bparfile, *parfile, *permfile;
  int i,j,k,m,n;
  int dim=2; // data dimensionality
  double *dat; // 1d representation of the data
  double **u; // 2d representation of the data. Each row - new datapoint
  double **u_local; // 2d representation of the local data on process. Each row - new datapoint
  
  double *wt=NULL; // array with weights for sample points
 
  int nseq; // num datapoints
  
  int nb; // num variable blocks
  int *bdim; // dimensions of variable blocks
  int **var; // variable id in variable blocks
  int *numst; // number of states in blocks
  CondChain *md=NULL;
  double *loglikehd, lhsum;
  float epsilon=EPSILON;
  float tp1, tp2;
  int ninit0=1, ninit1=0, ninit2=0;
  int nperm=1; // number of permutations for searching variable block structure inputted with -n parameter or 1 if non were supplied
  int nperm0; // number of permutations for searching variable block structure in file with permutations
  int *Numst0, bparexist=0, parexist=0, permexist=0;
  int **vlist0=NULL; // array that contains permutations used for searching variable block structure
  int maxdim=10, mindim=1, relaxsearch=0, randomseed=0;
  int No_nperm=1, printmodel=0;
	
  int rank; // MPI process rank
  int numproc; // total number of MPI processes
  MPI_Status status; // status of MPI Recv operation
  
  int openfileerror = 0; // indicator of an error while opening files
  int errorinvar = 0; // indicator of an error in variables inside file with block structure
  int errorindata = 0; // indicator of an error in data file
  int errorinpar = 0; // indicator of an error in parameter file
  int errorinperm = 0; // indicator of an error in permutation file
 
  double starttime, endtime; // used for timing performance

  DIAGCOV=0;
  /*----------------------------------------------------------------*/
  /*---------------- Read in parameters from command line-----------*/
  /*----------------------------------------------------------------*/
  
  MPI_Init(&argc, &argv); // initialize MPI
  
  // obtain process rank and total number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);  


  if (argc<=1) {
	if ( rank == 0 ) printmanual();
	MPI_Finalize();
    exit(0);
  }

  i = 1;
  while (i <argc)
    {
      if (*(argv[i]) != '-')
        {
		  if (rank == 0){
			printf("**ERROR** bad arguments\n");
			printf("To see manual, type trainmaster without arguments\n");
          }
          MPI_Finalize();
          exit(1);
        }
      else
        {
          switch(*(argv[i++] + 1))
            {
            case 'i':
              strcpy(infilename,argv[i]);
              break;
            case 'm':
              strcpy(mdfilename,argv[i]);
              break;
	    case 'p':
              strcpy(parfilename,argv[i]);
	      parexist=1;
	      break;
	    case 'b':
              strcpy(bparfilename,argv[i]);
	      bparexist=1;
	      break;
	    case 'o':
              strcpy(permfilename,argv[i]);
	      permexist=1;
	      break;
	    case 'e':
	      sscanf(argv[i],"%f",&epsilon);
	      break;
	    case 'd':
	      sscanf(argv[i],"%d",&dim);
	      break;
	    case 'n':
	      sscanf(argv[i],"%d",&nperm);
	      if (nperm<1) {
		     if (rank == 0) fprintf(stderr, "Wrong number of variable permutations: %d\n",nperm);
		     MPI_Finalize();
		     exit(1);
	      }
	      No_nperm=0;
	      break;
	    case 'x':
	      sscanf(argv[i],"%d",&maxdim);
	      break;
	    case '0':
	      ninit0=0;
	      i--;
	      break;
	    case '1':
	      sscanf(argv[i],"%d",&ninit1);
	      break;
	    case '2':
	      sscanf(argv[i],"%d",&ninit2);
	      break;
	    case 'r':
	      sscanf(argv[i],"%d",&randomseed);
	      break;
	    case 's':
	      relaxsearch=1; //if flagged, more flexible with the formulation of variable blocks
	      i--;
	      break;
	    case 'y':
	      mindim=2; //if flagged, more flexible with the formulation of variable blocks
	      i--;
	      break;
	    case 't':
	      printmodel=1; //if flagged, more flexible with the formulation of variable blocks
	      i--;
	      break;
	    case 'v':
	      DIAGCOV=1; //if flagged, more flexible with the formulation of variable blocks
	      i--;
	      break;
            default:
              {
				  if (rank == 0){
				    printf("**ERROR** bad arguments\n");
					printf("To see manual, type trainmaster without arguments\n");
				  }
				  MPI_Finalize();
                  exit(1);
              }
            }
          i++;
        }
    }

  /*----------------------------------------------------------------*/
  /*--------------------- open files -------------------------------*/
  /*----------------------------------------------------------------*/
  
  
  
  if (rank == 0){
	  infile = fopen(infilename, "r");
	  if (infile == NULL)
		{
		  printf("Couldn't open input data file \n");
		  openfileerror = 1;
	   }

	  if (bparexist) {
		bparfile = fopen(bparfilename, "r");
		if (bparfile == NULL)
		  {
		    printf("Couldn't open input data file \n");
		    openfileerror = 1;
		  }
	  }

	  if (parexist) {
		parfile = fopen(parfilename, "r");
		if (parfile == NULL)
		  {
			openfileerror = 1;
			printf("Couldn't open input data file \n");
		  }
	  }

	  if (permexist) {
		permfile = fopen(permfilename, "r");
		if (permfile == NULL)
		  {
			openfileerror = 1;
			printf("Couldn't open input data file \n");
		  }
	  } 
	  
	  mdfile = fopen(mdfilename, "w");
	  if (mdfile == NULL)
		{
		  openfileerror = 1;
		  printf("Couldn't open output model file \n");
		}
  }
  
  // check if any error while opening files occured and exit in this case	
  MPI_Bcast(&openfileerror, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (openfileerror){
	  MPI_Finalize();
	  exit(1);
  }
  
  /*----------------------------------------------------------------*/
  /*----------------- Read in data ---------------------------------*/
  /*----------------------------------------------------------------*/

  if (bparexist) {
	// read data on master process
	if (rank == 0){
		fscanf(bparfile, "%d %d\n", &dim, &nb);
		bdim=(int *)calloc(nb,sizeof(int));
		numst=(int *)calloc(nb,sizeof(int));
		
		printf("Num dimensions in bparfile: %d\n", dim);
		printf("Num blocks in bparfile: %d\n", nb);
		
		for (i=0;i<nb;i++){
		  fscanf(bparfile, "%d", bdim+i);
		  printf("Num dimensions in block %d: %d\n", i, bdim[i]);
		
		}
		
		for (i=0;i<nb;i++){
		  fscanf(bparfile, "%d", numst+i);
		  printf("Num states in block %d: %d\n", i, numst[i]);
		}
		
		var=(int **)calloc(nb,sizeof(int *));
		for (i=0;i<nb;i++) {
		  var[i]=(int *)calloc(bdim[i],sizeof(int));
		  for (j=0;j<bdim[i];j++) {
			fscanf(bparfile, "%d", var[i]+j);
			if (var[i][j]<0 || var[i][j]>=dim) {
				fprintf(stderr, "Error in variables: %d, dim=%d\n",var[i][j],dim);
				errorinvar = 1;
				
			}
		  }
	   }
    }
    
    // check if there was error in variables
    MPI_Bcast(&errorinvar, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (errorinvar){
		MPI_Finalize();
		exit(1);
	}
	
	// send parameters from master to slaves
	MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
	if (rank != 0){
		bdim=(int *)calloc(nb,sizeof(int));
		numst=(int *)calloc(nb,sizeof(int));
		var=(int **)calloc(nb,sizeof(int *));	
	}
	
	MPI_Bcast(bdim, nb, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(numst, nb, MPI_INT, 0, MPI_COMM_WORLD);
	
	if (rank != 0){
		for (i=0;i<nb;i++)
			var[i]=(int *)calloc(bdim[i],sizeof(int));
	}
	
	for (i=0;i<nb;i++)
		MPI_Bcast(var[i], bdim[i], MPI_INT, 0, MPI_COMM_WORLD);
  }

  nseq=0;
  
  // read data on master process
  if (rank == 0)
  {
	while (!feof(infile)) {
		for (j=0;j<dim;j++) {
			m=fscanf(infile, "%e",&tp1);
		}
		nseq++;
		fscanf(infile, "\n");
	}
	rewind(infile);

	fprintf(stderr, "Load in data: dim=%d, data size=%d\n",dim,nseq);

	dat=(double *)calloc(nseq*dim,sizeof(double));
	u=(double **)calloc(nseq,sizeof(double *));
	
	for (i=0;i<nseq;i++) { u[i]=dat+i*dim;  }

	for (m=0;m<nseq;m++) {
		if (feof(infile)) {
			fprintf(stderr, "Error: not enough data in input file\n");
			errorindata = 1;
			break;
		}
		for (j=0;j<dim;j++) {
			fscanf(infile, "%e",&tp1);
			dat[m*dim+j]=(double)tp1;
		}
		fscanf(infile, "\n");
	}
	
  }
  
  // check for errors in data file
  MPI_Bcast(&errorindata, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (errorindata){
	  MPI_Finalize();
	  exit(1);
  }
  
  // distribute work equally to all processes
  MPI_Bcast(&nseq, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  int *nseq_proc; // array with number of samples assigned to each process
  int nseq_local; // number of samples assigned to a process
  
  distribute_work(nseq, &nseq_local, &nseq_proc);
  scatter_doubleVectors(u, &u_local, nseq, nseq_local, dim, nseq_proc);
  
  // free global array with data
  if (rank == 0){
	  free(dat);
	  free(u);
  }
  

  if (parexist && bparexist==0) {//specify the number of components given the number of variables in a block
    Numst0=(int *)calloc(dim,sizeof(int));
    
    if (rank == 0){
		while (!feof(parfile)) {
		  fscanf(parfile, "%d %d\n",&m,&n);
		  if (m<=0 || m>dim) {
			fprintf(stderr, "Error in parameter file, block size exceeding dimension: %d %d\n",m,n);
			errorinpar = 1;
			break;
		  }
		  Numst0[m-1]=n;
		}
		
		for (i=0;i<dim;i++) {
		  if (Numst0[i]==0) {
			Numst0[i]=5+(i+1); //block size dimension plus 5
		  }
		}
	}
	
	// check if there is an error in parameter file
	MPI_Bcast(&errorinpar, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (errorinpar){
		MPI_Finalize();
		exit(1);
	}
	
	MPI_Bcast(Numst0, dim, MPI_INT, 0, MPI_COMM_WORLD);
		
  } else {Numst0=NULL;}
  
  
  if (permexist && bparexist==0){
	if (rank == 0){
		nperm0=0;
		while (!feof(permfile)) {
		  for (j=0;j<dim;j++) {
			fscanf(permfile, "%d",&m);
		  }
		  fscanf(permfile, "\n");
		  nperm0++;
		}
		rewind(permfile);
		vlist0=(int **)calloc(nperm0,sizeof(int *));
		for (i=0;i<nperm0;i++) vlist0[i]=(int *)calloc(dim,sizeof(int));

		for (i=0;i<nperm0;i++){
		  for (j=0;j<dim;j++) {
			fscanf(permfile, "%d",vlist0[i]+j);
			if (vlist0[i][j]<0 || vlist0[i][j]>=dim) {
				fprintf(stderr, "Error variable dimension in permutation file: dim=%d, variable: %d\n",
			  dim, vlist0[i][j]);
			  errorinperm = 1;
			  break;
			}
		  }
		  fscanf(permfile, "\n");
		}
	}
	
	// check if there is an error in permutation file
	MPI_Bcast(&errorinperm, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (errorinperm){
		MPI_Finalize();
		exit(1);
	}
	
	// send permutation data to slaves
	MPI_Bcast(&nperm0, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank != 0){
		vlist0=(int **)calloc(nperm0,sizeof(int *));
		for (i=0;i<nperm0;i++) vlist0[i]=(int *)calloc(dim,sizeof(int));
	}
	
	for (i = 0; i < nperm0; i++)
		MPI_Bcast(vlist0[i], dim, MPI_INT, 0, MPI_COMM_WORLD);
  } else {nperm0=0;}

  //Added 5/24/2017, properly set nperm and nperm0
  //If nperm is not specified in command line and nperm0>0, then use nperm0 for nperm
  if (No_nperm){
    if (nperm0>0) nperm=nperm0;
  }

  /*----------------------------------------------------------------*/
  /*----------------- Estimate HMM  ---------------------------------*/
  /*----------------------------------------------------------------*/
  
  loglikehd=(double *)calloc(nseq,sizeof(double));

  MPI_Barrier(MPI_COMM_WORLD);
  starttime = MPI_Wtime();

  if (bparexist) {//variable block structure specified
    hmmfit_minit(u_local, nseq_local, nb, bdim, var, numst, &md, loglikehd, &lhsum, (double)epsilon, wt,
		 ninit0, ninit1, ninit2, randomseed); //lhsum is loglikelihood
    lhsum-=(double)(computenp(nb, bdim,numst))*log((double)nseq)*0.5; //BIC           
  }
  else {//variable block structure not given and will be searched
    hmmfit_vb(u_local, nseq_local, dim, &nb, &bdim, &var, nperm, nperm0, vlist0,
	      &md, loglikehd, &lhsum, (double)epsilon, wt, ninit0, ninit1,ninit2, randomseed,
	      Numst0, maxdim, mindim, relaxsearch); //output lhsum is BIC not loglikelihood
  }

  MPI_Barrier(MPI_COMM_WORLD);
  endtime = MPI_Wtime();
  
  if (rank == 0)
	fprintf(stdout, "Exec time: %e\n", endtime - starttime);

  //Output the BIC value
  if (rank == 0)
	fprintf(stdout, "BIC of the model: %e\n", lhsum);

  //Output loglikehd from hmmfit() is not written out

  // Binary file for the output model
  if (rank == 0)
	write_ccm(md, mdfile);

  //~ //Ascii file for the model
  //~ if (printmodel) print_ccm(md,stdout); //debug

  for (i = 0; i < nseq_local; i++)
		free(u_local[i]);
  free(u_local);

  MPI_Finalize();
}
