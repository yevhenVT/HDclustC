/*==========================================================================================*/
/*                                                                                          */
/* Copyright (C) [Dec 2015]-[April 2017] Jia Li, Department of Statistics,                  */
/* The Pennsylvania State University, USA - All Rights Reserved                             */
/*                                                                                          */
/* Unauthorized copying of this file, via any medium is strictly prohibited                 */
/*                                                                                          */
/* Proprietary and CONFIDENTIAL                                                             */
/*                                                                                          */
/* NOTICE: All information contained herein is, and remains the property of The             */
/* Pennsylvania State University. The intellectual and technical concepts                   */
/* contained herein are proprietary to The Pennsylvania State University and may            */
/* be covered by U.S. and Foreign Patents, patents in process, and are protected            */
/* by trade secret or copyright law. Dissemination of this information or                   */
/* reproduction of this material is strictly forbidden unless prior written                 */
/* permission is obtained from Jia Li at The Pennsylvania State University. If              */
/* you obtained this code from other sources, please write to Jia Li.                       */
/*                                                                                          */
/*                                                                                          */
/* The software is a part of the package for                                                */
/* Clustering with Hidden Markov Models on Variable Blocks                                  */
/*                                                                                          */
/* Written by Jia Li <jiali@stat.psu.edu>, April 7, 2017                                    */ 
/*                                                                                          */
/*==========================================================================================*/

#include "hmm_mpi.h"
#include <string.h>
#include <mpi.h>

int DIAGCOV=0;

void printmanual()
{
  fprintf(stdout, "---------------- Usage manual for testsync_mpi ---------------------\n");
  fprintf(stdout, "To see the usage manual, use testsync without any argument.\n\n");
  fprintf(stdout, "Syntax: testsync_mpi -i [...] -m [...] ......\n\n");
  fprintf(stdout, "-i [input data file]\n");
  fprintf(stdout, "   data file is in a typical ascii matrix format: each row contains varia\
bles for one data point,\n");
  fprintf(stdout, "   each column in a row is the value of one variable. \n\n");
  fprintf(stdout, "-m [input model filename], the input model is a binary file\n\n");
  fprintf(stdout, "-o [output cluster result filename], each row contains the cluster label of one data point.\n\n");
  fprintf(stdout, "-r [input reference cluster filename], when this file is supplied, clusters are aligned.\n\n");
  fprintf(stdout, "-a [output cluster info filename], when this file is given, existing clustering results are \n");
  fprintf(stdout, "   stored in this file for future alignment. \n\n");
  fprintf(stdout, "-l [minimum cluster size], clusters with sizes lower than the value are merged with\n");
  fprintf(stdout, "   closest big clusters according to modes. Default value is 1. \n\n");
  fprintf(stdout, "-t [identical mode threshold], The larger the threshold, the less\n");
  fprintf(stdout, "   stringent to declare two modes being identical. Default: 0.01. \n\n");
  fprintf(stdout, "-u A flag. When set, the normalized L1 norm is used to compute distance between modes.\n");
  fprintf(stdout, "-v A flag. If flagged, diagonal covariance is assumed. If not, non-diagonal covariance is assumed.\n");
  fprintf(stdout, "   If not set, maximum L1 distance in a certain dimension is used, more stringent.\n");
  fprintf(stdout, "------------------------------------------------------------------\n");
}

// write file with existing cluster information, If previous cluster structure exists,
// it is augmented by new clusters
void augmentrefcls(FILE *clsfile, int rncls, int rnseq, double **refmode, double *sigmadat,
		   int **refpath, int *refcls, int dim, int nb, int numcls, int *clsid, int *id3,
		   double **mode, int newnseq, int **newoptst, int *clsid2, int *newid, int nseq, 
		   int *id2, double ***combmode_pt)
{
  int i,j,k,m,n;
  double **newrefmode=NULL, **newrefmode2, **combmode;
  int *ct; // number of points in new clusters
  int rncls2, rnseq2, **newrefpath, *newrefcls, *newrefcls2;

  if (numcls>0 && id3!=NULL && mode!=NULL) {
    ct=(int *)calloc(numcls,sizeof(int));
    for (i=0;i<numcls;i++) ct[i]=0;
    newrefmode=(double **)calloc(numcls,sizeof(double *));
    for (i=0;i<numcls;i++) {
      newrefmode[i]=(double *)calloc(dim,sizeof(double));
      for (j=0;j<dim;j++) newrefmode[i][j]=0.0;
    }
    newrefmode2=(double **)calloc(numcls,sizeof(double *));
    for (i=0;i<numcls;i++) {
      newrefmode2[i]=(double *)calloc(dim,sizeof(double));
      for (j=0;j<dim;j++) newrefmode2[i][j]=0.0;
    }

    //Computes the average first, may argue using average directly
    for (i=0;i<nseq;i++) {
      if (clsid[i]>=rncls) {
		k=id3[newid[id2[i]]];
		ct[clsid[i]-rncls]++; //number of times new cluster appears
		for (j=0;j<dim;j++) newrefmode[clsid[i]-rncls][j]+=mode[k][j];
      }
    }
    
    for (i=0;i<numcls;i++) {
      if (ct[i]>0) {
		for (j=0;j<dim;j++) {
			newrefmode[i][j]/=(double)ct[i];
			newrefmode2[i][j]=newrefmode[i][j];
		}
      }
    }
    //Find minimum and maximum per dimension and per cluster
    for (i=0;i<nseq;i++) {
      if (clsid[i]>=rncls) {
		k=id3[newid[id2[i]]];
		for (j=0;j<dim;j++) {
			if (mode[k][j]> newrefmode[clsid[i]-rncls][j])
				newrefmode[clsid[i]-rncls][j]=mode[k][j]; //store maximum
			if (mode[k][j]< newrefmode2[clsid[i]-rncls][j])
				newrefmode2[clsid[i]-rncls][j]=mode[k][j]; //store minimum
		}
      }
    }
    
    for (i=0;i<numcls;i++) {
      if (ct[i]>0) //use the midpoint between the maximum and the minimum
		for (j=0;j<dim;j++) newrefmode[i][j]=0.5*(newrefmode[i][j]+newrefmode2[i][j]);
    }
    
    free(ct);
    for (i=0;i<numcls;i++) free(newrefmode2[i]);
    free(newrefmode2);

  } else {//No new modes, only new paths
    numcls=0;
  }
  
  rncls2=rncls+numcls;
  rnseq2=rnseq+newnseq;

  newrefpath=(int **)calloc(rnseq2,sizeof(int *));
  for (i=0;i<rnseq;i++) newrefpath[i]=refpath[i];
  for (i=rnseq;i<rnseq+newnseq;i++) newrefpath[i]=newoptst[i-rnseq];

  newrefcls=(int *)calloc(rnseq2,sizeof(int));
  newrefcls2=(int *)calloc(rnseq2,sizeof(int)); //for use as sorted newrefcls[]
  for (i=0;i<rnseq;i++) newrefcls[i]=refcls[i];
  for (i=rnseq;i<rnseq+newnseq;i++) newrefcls[i]=clsid2[i-rnseq];

  if (numcls>0) {
    combmode=(double **)calloc(numcls+rncls,sizeof(double *));
    for (i=0;i<rncls;i++) combmode[i]=refmode[i];
    for (i=0;i<numcls;i++) combmode[rncls+i]=newrefmode[i];
  }
  else {
    combmode=refmode;
  }
  
  int **bufoptst, *invid;
  bufoptst=(int **)calloc(rnseq2,sizeof(int *));
  invid=(int *)calloc(rnseq2,sizeof(int));
  
  SortLexigraphicInt(newrefpath, bufoptst,invid,nb,rnseq2);
  for (i=0;i<rnseq+newnseq;i++) newrefcls2[i]=newrefcls[invid[i]];

  int *ct2;
  ct2=(int *)calloc(rncls2,sizeof(int));
  for (i=0;i<rncls2;i++) ct2[i]=0;
  for (i=0;i<nseq;i++) ct2[clsid[i]]++; //#occurrences for each cluster in this dataset

  if (clsfile!=NULL)
    printrefcls(clsfile, rncls2, rnseq2, combmode, sigmadat, bufoptst, newrefcls2, dim, nb, nseq,ct2);

  free(ct2);
  free(newrefpath);
  free(newrefcls);
  free(newrefcls2);
  free(bufoptst);
  free(invid);
  //the memory of newrefmode[i] should NOT be released, taken by combmode
  if (newrefmode!=NULL) free(newrefmode); 
  
  *combmode_pt=combmode;
}

//Switch cluster label of very small clusters to the big one closest to the data point
void AdjustCluster(int *clsid, int nseq, int rncls, double **refmode, double **u, int dim, int mincls)
{
  int i,j,k,m,n;
  int *ct;
  double db1,db2;

  ct=(int *)calloc(rncls,sizeof(int));
  for (i=0;i<rncls;i++) ct[i]=0;
  for (i=0;i<nseq;i++) ct[clsid[i]]++;
  for (i=0,k=-1,m=0,n=0;i<rncls;i++) {
    if (ct[i]>=mincls) {
      m++;
      n+=ct[i];
      if (k<0) k=i; //get the first big cluster id
    }
  }
  if (k<0) {
    fprintf(stderr, "Minimum cluster size is too big: No cluster has size >=%d\n",mincls);
    exit(1);
  }
  else {
    fprintf(stderr, "Data size: %d, number of large clusters: %d, #points in large clusters: %d\n",
	    nseq,m,n);
    db1=(double)n/(double)nseq;
    if (db1<0.8)
      fprintf(stderr, "Warning: percentage of points in large clusters is small: %.1f < 80 percent\n",db1*100);
  }

  for (i=0;i<nseq;i++) {
    if (ct[clsid[i]]<mincls) {
      m=k;
      db1=l2sq(refmode[k],u[i],dim);
      for (j=k+1;j<rncls;j++) {
		if (ct[j]>=mincls) {
			db2=l2sq(refmode[j],u[i],dim);
			if (db2<db1) {
				db1=db2;
				m=j;
			}
		}
      }
      clsid[i]=m;
    }
  }

  free(ct);
}

int main(argc, argv)
     int argc;
     char *argv[];
{
  char infilename[300], mdfilename[300], outfilename[300], reffilename[300], clsfilename[300];
  FILE *infile, *mdfile, *outfile, *reffile, *clsfile;
  int i,j,k,m,n,noinfile=1;
  
  int dim=2;
  double *dat;
  double **u;
  double **u_local; // samples assigned to a process 
  double *wt=NULL,*merit,*meritbuf;
  int nseq=0, nb, *bdim, **var,  *numst, **optst, *intbuf, *cd, *cd2;
  CondChain *md=NULL;
  float epsilon=EPSILON, modethreshold=0.01;
  float tp, tp1, tp2;
  int refexist=0, clsexist=0;
  int mincls=1, usemeandist=0;
  double **combmode;

  // MPI support
  int rank; // rank of MPI process
  int numproc; // number of MPI processes
  
  // error indicators
  int errorreffile = 0; // indicator of an error while opening file with reference clusters
  int errorclsfile = 0; // indicator of an error while opening output file with cluster info
  int errorinputfile = 0; // indicator of an error while opening file with input data
  int errorindata = 0; // indicator of an error in input data file
  int errormodelfile = 0; // indicator of an error while opening input file with Hmm model
  int erroroutfile = 0; // indicator of an error while opening output file with results

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);  

  /*----------------------------------------------------------------*/
  /*---------------- Read in parameters from command line-----------*/
  /*----------------------------------------------------------------*/
  DIAGCOV=0;
  if (argc<=1) {
	if (rank == 0)
		printmanual();
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
			printf("To see manual, type testsync without arguments\n");
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
			  noinfile=0;
              break;
            case 'm':
              strcpy(mdfilename,argv[i]);
              break;
            case 'o':
              strcpy(outfilename,argv[i]);
              break;
            case 'r':
              strcpy(reffilename,argv[i]);
	      refexist=1;
              break;
            case 'a':
              strcpy(clsfilename,argv[i]);
	      clsexist=1;
              break;
            case 'l':
              sscanf(argv[i],"%d",&mincls);
              break;	     
	    case 't':
              sscanf(argv[i],"%f",&modethreshold);
              break;
	    case 'u':
              usemeandist=1;
              i--;
	      break;
	    case 'v':
              DIAGCOV=1;
              i--;
              break;
            default:
              {
				if (rank == 0){
					printf("**ERROR** bad arguments\n");
					printf("To see manual, type testsync without arguments\n");
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
	  mdfile = fopen(mdfilename, "r");
	  if (mdfile == NULL)
		{
		  printf("Couldn't open input model file \n");
		  errormodelfile = 1;
		}
  }
  MPI_Bcast(&errormodelfile, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (errormodelfile){
	MPI_Finalize();
	exit(1);
  }

  
  if (noinfile) {
	if (rank == 0){
		printf("No file with input data was provided!\n");
		
		md=(CondChain *)calloc(1,sizeof(CondChain));
		read_ccm(md, mdfile);
		print_ccm(md,stdout);
	}
	MPI_Finalize();
	exit(0);
  }
  
  if (rank == 0){
	  outfile = fopen(outfilename, "w");
	  if (outfile == NULL)
	  {
		  printf("Couldn't open output outfile\n");
		  erroroutfile = 1;
	  }
  }

  MPI_Bcast(&erroroutfile, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (erroroutfile){
	MPI_Finalize();
	exit(1);
  }

  if (rank == 0){
	  if (refexist) {
		reffile = fopen(reffilename, "r");
		if (reffile == NULL){
			printf("Couldn't open output reffile\n");
			errorreffile = 1;
		}
	  }
  }
  
  MPI_Bcast(&errorreffile, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (errorreffile){
	MPI_Finalize();
	exit(1);
  }
  
  if (rank == 0){
	if (clsexist) {//clsfile stores reference cluster info for future testsync 
		clsfile = fopen(clsfilename, "w");
		if (clsfile == NULL)
		{
			printf("Couldn't open output clsfile\n");
			errorclsfile = 1;
		}
	}
  }
  
  MPI_Bcast(&errorclsfile, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (errorclsfile){
	MPI_Finalize();
	exit(1);
  }
  
  
  if (rank == 0){
	infile = fopen(infilename, "r");
	if (infile == NULL)
	{
		printf("Couldn't open input data file \n");
		errorinputfile = 1;
	}
  }
  
  MPI_Bcast(&errorinputfile, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (errorinputfile){
	MPI_Finalize();
	exit(1);
  }

  /*----------------------------------------------------------------*/
  /*----------------- Read in data ---------------------------------*/
  /*----------------------------------------------------------------*/
  md=(CondChain *)calloc(1,sizeof(CondChain));

  // Binary file for the model
  if (rank == 0)
	read_ccm(md, mdfile);
	
  send_new_ccm(md);
	
  for (i=0,dim=0;i<md->nb;i++) dim+=md->bdim[i];

  nseq=0;
  
  if (rank == 0){
	while (!feof(infile)) {
		for (j=0;j<dim;j++) {
		  m=fscanf(infile, "%e",&tp1);
		}
		nseq++;
		fscanf(infile, "\n");
	  }
	  rewind(infile);

	  dat=(double *)calloc(nseq*dim,sizeof(double));
	  u=(double **)calloc(nseq,sizeof(double *));
	  for (i=0;i<nseq;i++) { u[i]=dat+i*dim;  }

	  // For testing purpose only
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
   
  // distribute work equally among all processes
  MPI_Bcast(&nseq, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  int *nseq_proc; // array with number of samples assigned to each process
  int nseq_local; // number of samples assigned to a process
  
  distribute_work(nseq, &nseq_local, &nseq_proc);
  scatter_doubleVectors(u, &u_local, nseq, nseq_local, dim, nseq_proc);
  
  //fprintf(stdout, "Rank %d nseq_local = %d\n",rank,nseq_local);
	
  
  // calculate standard deviation for entire data set and
  // free global arrays with data
  double *sigmadat;
  
  if (rank == 0){
	  sigmadat=(double *)calloc(dim,sizeof(double));

	  //The overall deviation
	  DataSigma(u,sigmadat,dim,nseq);

	  free(dat);
	  free(u);
  }
  

  /*----------------------------------------------------------------*/
  /*--------------- Read in existing cluster information -----------*/
  /*----------------------------------------------------------------*/

  double **refmode; // coordinates of modes
  double *refsigma=NULL; // common standard deviation
  int rncls=0; // number of clusters
  int rnseq=0; // number of distinct sequences
  int *refcls; // cluster ids
  int rsz; // total number of sequences
  int **refpath; // optimal state sequences
  int *rct=NULL; // number of points in clusters
  
  if (refexist) {
	if (rank == 0)
		readrefcls(reffile, &rncls, &rnseq, &refmode, &refsigma, &refpath, &refcls, dim, md->nb, &rsz, &rct);
  }

  /*----------------------------------------------------------------*/
  /*----------------- Estimate HMM  ---------------------------------*/
  /*----------------------------------------------------------------*/
  ordervar(u_local,nseq_local,md->nb,md->bdim,md->var);

  int **optst_local;
  double *merit_local;

  optst_local=(int **)calloc(nseq_local,sizeof(int *));
  for (i=0;i<nseq_local;i++) optst_local[i]=(int *)calloc(md->nb,sizeof(int));
  
  
  merit_local=(double *)calloc(nseq_local,sizeof(double));
  meritbuf=(double *)calloc(md->maxnumst,sizeof(double));

  for (i=0;i<nseq_local;i++) {
    viterbi(md,u_local[i],optst_local[i],NULL,meritbuf);
    merit_local[i]=meritbuf[optst_local[i][md->nb-1]];
  }
  
  // gather viterbi results on master node  
  gather_intVectors(&optst, optst_local, nseq, nseq_local, md->nb, nseq_proc);
  
  //optst[nseq][md->nb] stores the viterbi optimal state sequence for each instance
  //Output the path information
  /****
  fprintf(stdout, "Optimal path of states for every instance:\n");
  for (i=0;i<nseq;i+=100) {
    for (j=0;j<md->nb;j++)
      fprintf(stdout, "%d ", optst[i][j]);
    fprintf(stdout, "\n");
  }
  ****/
  if (rank == 0)
	fprintf(stderr, "Viterbi paths found\n");
  
  //---------------------------------------------------------//
  //----------- Check with reference cluster ----------------//
  //---------------------------------------------------------//
  
  int *clsid, noref;
  int allpathsfound = 0; // indicator that found sequences match with the reference 

  if (rank == 0){
	  clsid=(int *)calloc(nseq,sizeof(int));
	  for (i=0;i<nseq;i++) clsid[i]=-1;
	  if (refexist) {
		noref=0;
		for (i=0;i<nseq;i++) {
		  clsid[i]=FindEntry(refpath, optst[i],md->nb,rnseq);
		  if (clsid[i]>=0) {
			clsid[i]=refcls[clsid[i]];
		  }
		  else {
			noref++;	
		  }
		}
	  } else {noref=nseq;}

	  if (noref==0) { //all done
		if (clsexist) {
		  if (rct!=NULL) free(rct);
		  rct=(int *)calloc(rncls,sizeof(int));
		  for (i=0;i<rncls;i++) rct[i]=0;
		  for (i=0;i<nseq;i++) rct[clsid[i]]++;
		  printrefcls(clsfile, rncls, rnseq, refmode, refsigma, refpath, refcls, dim, md->nb, nseq, rct);
		}

		if (mincls>1) {
		  AdjustCluster(clsid, nseq,rncls,refmode,u, dim, mincls);
		}
		printclsinfo(outfile,clsid,nseq,rncls);
		fprintf(stdout, "All paths found\n");
		allpathsfound = 1;
	  }
  }
  
  MPI_Bcast(&allpathsfound, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (allpathsfound)
  {
	  MPI_Finalize();
	  exit(0);
  }
  
  //---------------------------------------------------------//
  // Samples that can't be treated by reference clusters     //
  //---------------------------------------------------------//

  int **optst2, **newoptst, newnseq, *newid, *id2;
  
  if (rank == 0){
	  id2=(int *)calloc(nseq,sizeof(int));

	  if (noref<nseq) {
		optst2=(int **)calloc(noref,sizeof(int *));
		for (i=0,k=0;i<nseq;i++){
		  if (clsid[i]<0){//path doesn't exist in reference
			optst2[k]=optst[i];
			id2[i]=k;
			k++;
		  } else {id2[i]=-1;}
		}
	  }
	  else {
		optst2=optst;
		for (i=0;i<nseq;i++) id2[i]=i;
	  }

	  newid=(int *)calloc(noref,sizeof(int));
	  // find distinct best sequences 
	  FindDifSeq(optst2, noref, md->nb, &newoptst, &newnseq, newid);

	  fprintf(stderr, "After FindDifSeq, noref=%d, newnseq=%d\n",noref,newnseq);
  }
  
  // distribute new distinct sequences among all processes
  MPI_Bcast(&newnseq, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  int *newnseq_proc; // array with number of distinct sequences assigned to each process
  int newnseq_local; // number of sequences assigned to a process
  int **newoptst_local;
  
  distribute_work(newnseq, &newnseq_local, &newnseq_proc);
  scatter_intVectors(newoptst, &newoptst_local, newnseq, newnseq_local, md->nb, newnseq_proc);
  
  
  //=============================//
  // Compute modes               //
  //=============================//
  CompMode *cpm, *cpm_local;
  
  if (rank == 0){
	cpm=(CompMode *)calloc(newnseq,sizeof(CompMode));
	
	for (i=0;i<newnseq;i++) SetCompMode(cpm+i,newoptst[i], md);
  }
  
  cpm_local=(CompMode *)calloc(newnseq_local,sizeof(CompMode));
  
  for (i=0;i<newnseq_local;i++) {
    SetCompMode(cpm_local+i,newoptst_local[i], md);
    cpm_local[i].logpdf=bwmem(md, cpm_local[i].mu, cpm_local[i].mode);
  }

  send_comp_mode(cpm_local, newnseq_local, cpm, newnseq, dim, newnseq_proc);

  int *cls, numcls=0;
  double **mode;
  
  //--------------------------------------------------------------------//
  //Second round alignment with reference based on newly computed modes //
  //--------------------------------------------------------------------//
  int norefmode, *clsid2;

  int alldone = 0; // indicator that all new modes were accosiated with old ones
  
  if (rank == 0){
	  clsid2=(int *)calloc(newnseq,sizeof(int));
	  norefmode=0;
	  for (i=0;i<newnseq;i++){//See whether new modes coincide with existing modes
		clsid2[i]=FindCluster(cpm[i].mode, dim, rncls, refmode, sigmadat, modethreshold, usemeandist);
		if (clsid2[i]<0) {
		  norefmode++; //still not assigned to existing cluster,can't compute this way
		}     
	  }

	  for (i=0;i<nseq;i++) {
		if (clsid[i]<0) {
		  clsid[i]=clsid2[newid[id2[i]]];
		} 
	  }

	  if (norefmode==0) { //all done
		if (clsexist)
		  augmentrefcls(clsfile, rncls, rnseq, refmode, sigmadat, refpath, refcls, dim, md->nb, numcls,clsid,
				NULL, NULL, newnseq, newoptst, clsid2, newid, nseq, id2, &combmode);
		fprintf(stdout, "All modes associated, nseq=%d, noref=%d, newnseq=%d\n", nseq, noref, newnseq);
		if (mincls>1) {
		  AdjustCluster(clsid, nseq,rncls,refmode,u, dim, mincls);
		}
		printclsinfo(outfile,clsid,nseq,rncls);
		alldone = 1;
	  }
  }
  
  MPI_Bcast(&alldone, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (alldone){
	MPI_Finalize();
	exit(0);
  }
  
  //--------------------------------------------------------------------//
  //  Samples left that have to form their own new clusters             //
  //--------------------------------------------------------------------//
  int *id3;
  
  if (rank == 0){
	  id3=(int *)calloc(newnseq,sizeof(int));
	  cls=(int *)calloc(norefmode,sizeof(int));
	  mode=(double **)calloc(norefmode,sizeof(double *));
	  for (i=0,k=0;i<newnseq;i++) {
		if (clsid2[i]<0) {
		  mode[k]=cpm[i].mode;
		  id3[i]=k;
		  k++;
		} else {id3[i]=-1;}
	  }

	  fprintf(stderr, "nseq=%d, newnseq=%d, norefmode=%d\n",nseq,newnseq, norefmode);
	  
	  // group modes that are close together
	  groupmode(mode, dim, norefmode, cls, &numcls, sigmadat, modethreshold, usemeandist);

	  //There're numcls new clusters and rncls old clusters, offset the cls[] labels
	  //by rncls
	  for (i=0;i<nseq;i++) {
		if (clsid[i]<0) {
		  k=id3[newid[id2[i]]];
		  clsid[i]=cls[k]+rncls;
		  clsid2[newid[id2[i]]]=cls[k]+rncls;
		}
	  }

	  //--------------------------------------------------------------------//
	  //  Print out information                                             //
	  //--------------------------------------------------------------------//
	  
	  fprintf(stdout, "nseq=%d, noref=%d, newnseq=%d, norefmode=%d, numcls=%d, rncls=%d, #clusters=%d\n",
		  nseq, noref, newnseq, norefmode, numcls, rncls, numcls+rncls);

	  //If "-a" specified, output the proper cluster info file
	  if (clsexist) {
		augmentrefcls(clsfile, rncls, rnseq, refmode, sigmadat, refpath, refcls, dim, md->nb, numcls,clsid,
			  id3, mode, newnseq, newoptst, clsid2, newid, nseq, id2, &combmode);
	  }

	  //Output cluster labels
	  if (mincls>1) {
		if (clsexist==0) {
		  augmentrefcls(NULL, rncls, rnseq, refmode, sigmadat, refpath, refcls, dim, md->nb, numcls,clsid,
				id3, mode, newnseq, newoptst, clsid2, newid, nseq, id2, &combmode);
		}
		AdjustCluster(clsid, nseq,rncls+numcls,combmode,u, dim, mincls);
	  }
	  printclsinfo(outfile,clsid,nseq,rncls+numcls);
	}
	
	MPI_Finalize();
}



