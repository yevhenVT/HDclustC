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
#include "agglomerate.h"
#include <string.h>

void printmanual()
{
  fprintf(stdout, "---------------- Usage manual for ridgeline_md ---------------------\n");
  fprintf(stdout, "Compute separability between clusters based on ridgelines.\n");
  fprintf(stdout, "Each cluster contains multiple state sequence paths and is approximated by \n");
  fprintf(stdout, "a single Gaussian distribution when computing ridgelines.\n");
  fprintf(stdout, "The cluster information is from the \"-r\" option, which is output by testsync \"-a\"\n");
  fprintf(stdout, "option. No data file required. The Gaussian is fused from the paths.\n\n");
  fprintf(stdout, "To see the usage manual, use ridgeline_md without any argument.\n\n");
  fprintf(stdout, "Syntax: ridgeline_md -r [...] -m [...] ......\n\n");
  fprintf(stdout, "-m [input model filename], the output model is a binary file\n\n");
  fprintf(stdout, "-o [output ridgeline result filename], each row contains ridgeline density values and \n");
  fprintf(stdout, "   separability value in the last field.\n\n");
  fprintf(stdout, "-r [input cluster filename], existing cluster info based on mode association\n\n");
  fprintf(stdout, "-a A flag. When flagged, agglomerative clustering will be applied.\n\n");
  fprintf(stdout, "-v This is a flag. If not flagged, non-diagonal covariance is assumed. If flagged, diagonal covariance.\n");
  fprintf(stdout, "-t [floating number], this is the stepsize when computing ridgelines. Default is 0.02.\n\n");
  fprintf(stdout, "-s [minimum cluster size], for being qualified to participate in agglomerative clustering.\n\n");
  fprintf(stdout, "------------------------------------------------------------------\n");
}

double separability(double *pdfv, int na)
{
  int i,j,k,m,n;
  double v;

  v=pdfv[0];
  for (i=1;i<na;i++){
    if (pdfv[i]<v) v=pdfv[i];
  }

  if (pdfv[0]<pdfv[na-1]){
    return(1.0-v/pdfv[0]);
  }
  else {
    return(1.0-v/pdfv[na-1]);
  }
}

void ridgeline(GaussModel *md1, GaussModel *md2, double pi1, 
	       double *alpha, int na, double *pdfv)
{
  int i,j,k,m,n,mm;
  double *ft,*mu,**A,**A_inv,v1, *buf1, *buf2;
  int dim;

  dim=md1->dim;
  ft=(double *)calloc(dim,sizeof(double));
  mu=(double *)calloc(dim,sizeof(double));
  buf1=(double *)calloc(dim,sizeof(double));
  buf2=(double *)calloc(dim,sizeof(double));
  matrix_2d_double(&A,dim,dim);
  matrix_2d_double(&A_inv,dim,dim);

  for (m=0;m<na;m++) {
    for (i=0;i<dim;i++)
      for (j=0;j<dim;j++) {
	A[i][j]=(1.0-alpha[m])*md1->sigma_inv[i][j]
	  +alpha[m]*md2->sigma_inv[i][j];
      }

    for (i=0;i<dim;i++) {
      buf1[i]=0.0; buf2[i]=0.0;
      for (j=0;j<dim;j++) {
	buf1[i]+=md1->sigma_inv[i][j]*md1->mean[j];
	buf2[i]+=md2->sigma_inv[i][j]*md2->mean[j];
      }
      mu[i]=(1.0-alpha[m])*buf1[i]+alpha[m]*buf2[i];
    }

    mm=mat_det_inv_diag_double(A, A_inv, &v1, dim, DIAGCOV);

    if (mm==2) {
      fprintf(stderr, "Warning: Singular matrix A in ridgeline\n");
    }

    for (i=0;i<dim;i++) {
      ft[i]=0.0;
      for (j=0;j<dim;j++) {
	ft[i]+=A_inv[i][j]*mu[j];
      }
    }
    pdfv[m]=pi1*gauss_pdf(ft,md1)+(1.0-pi1)*gauss_pdf(ft,md2);
  }
  
  free(ft);
  free(mu);
  free(buf1);
  free(buf2);
  for (i=0;i<dim;i++) {
    free(A[i]);
    free(A_inv[i]);
  }
  free(A);
  free(A_inv);
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
  double *wt=NULL,*merit,*meritbuf;
  int nseq=0, nb, *bdim, **var,  *numst, **optst, *intbuf, *cd, *cd2;
  CondChain *md=NULL;
  double *loglikehd, lhsum;
  float epsilon=EPSILON, modethreshold=0.01;
  float tp, tp1, tp2;
  int refexist=0, clsexist=0;
  int *ct, *cls, *nonsing, na;
  GaussModel *mds;
  double lambda=0.01,v1,v2;
  double *alpha, *pdfv;
  double stepsz=0.02;
  int linkage=0;
  int nbigcls, *orgid;
  float **dist;
  int mincls=1;
  double *prior, *prior2;
  DIAGCOV=0;
  /*----------------------------------------------------------------*/
  /*---------------- Read in parameters from command line-----------*/
  /*----------------------------------------------------------------*/

  if (argc<=1) {
    printmanual();
    exit(0);
  }

  i = 1;
  while (i <argc)
    {
      if (*(argv[i]) != '-')
        {
          printf("**ERROR** bad arguments\n");
	  printf("To see manual, type ridgeline_md without arguments\n");
          exit(1);
        }
      else
        {
          switch(*(argv[i++] + 1))
            {
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
	    case 't':
	      sscanf(argv[i],"%f",&tp1);
	      stepsz=tp1;
	      break;
	    case 's':
	      sscanf(argv[i],"%d",&mincls);
	      break;
	    case 'a':
	      linkage=1;
	      i--;
	      break;
            case 'v':
              DIAGCOV=1; 
	      i--;
	      break;
            default:
              {
                printf("**ERROR** bad arguments\n");
		printf("To see manual, type ridgeline_md without arguments\n");
                exit(1);
              }
            }
          i++;
        }
    }

  /*----------------------------------------------------------------*/
  /*--------------------- open files -------------------------------*/
  /*----------------------------------------------------------------*/
  
  mdfile = fopen(mdfilename, "r");
  if (mdfile == NULL)
    {
      printf("Couldn't open input model file \n");
      exit(1);
    }

  outfile = fopen(outfilename, "w");
  if (outfile == NULL)
    {
      printf("Couldn't open output outfile\n");
      exit(1);
    }

  reffile = fopen(reffilename, "r");
  if (reffile == NULL)
    {
      printf("Couldn't open output reffile\n");
      exit(1);
    }
  
  /*----------------------------------------------------------------*/
  /*----------------- Read in data ---------------------------------*/
  /*----------------------------------------------------------------*/
  md=(CondChain *)calloc(1,sizeof(CondChain));

  // Binary file for the model
  read_ccm(md, mdfile);
  for (i=0,dim=0;i<md->nb;i++) dim+=md->bdim[i];

  /*----------------------------------------------------------------*/
  /*--------------- Read in existing cluster information -----------*/
  /*----------------------------------------------------------------*/

  double **refmode, *refsigma=NULL;
  int rncls=0,rnseq=0, **refpath, *refcls, rsz, *rct;
 
  readrefcls(reffile, &rncls, &rnseq, &refmode, &refsigma, &refpath, &refcls, dim, md->nb, &rsz, &rct);

  /*----------------------------------------------------------------*/
  /*----------------- Estimate Gaussian for each cluster -----------*/
  /*----------------------------------------------------------------*/

  // Estimate the mean and the covariance matrices with shrinkage
  // Use subroutines to do this.
  mds=(GaussModel *)calloc(rncls,sizeof(GaussModel));
  nonsing=(int *)calloc(rncls,sizeof(int));
  
  int **mypath;
  mypath=(int **)calloc(rnseq,sizeof(int *));

  for (m=0;m<rncls;m++) {
    newgauss(mds+m,dim,1);
    for (i=0,k=0;i<rnseq;i++){
      if (refcls[i]==m) {
	mypath[k]=refpath[i];
	k++;
      }
    }
    nonsing[m]=FuseGauss(mds+m, mypath, k, md);
  }

  fprintf(stderr, "rncls=%d, Gaussian models fused\n",rncls);

  //-----------------------------------------------//
  // Compute ridgeline                              //  
  //-----------------------------------------------//

  na=1;
  v1=na*stepsz;
  while (v1<1.0) {
    na++;
    v1=na*stepsz;
  }
  na++; //alpha[0]=0.0 has to be counted
  alpha=(double *)calloc(na,sizeof(double));
  pdfv=(double *)calloc(na,sizeof(double));
  for (i=0;i<na;i++) {
    alpha[i]=i*stepsz;
  }
  if (alpha[na-1]>1.0) alpha[na-1]=1.0;

  fprintf(stderr, "na=%d\n",na);
  
  //For each pair of clusters, if qualified, compute the ridgeline
  //Compute the separability of the each ridgeline
  for (m=0,nbigcls=0;m<rncls;m++) {
    if (nonsing[m] && rct[m]>=mincls) {
      nbigcls++;
    }
  }

  if (nbigcls==0) {
    fprintf(stderr, "no big and non-singular cluster\n");
    exit(0);
  }

  orgid=(int *)calloc(nbigcls,sizeof(int));
  for (m=0,n=0;m<rncls;m++) {
    if (nonsing[m] && rct[m]>=mincls) {
      orgid[n]=m;
      n++;
    }
  }

  dist=(float **)calloc(nbigcls,sizeof(float *));
  for (i=0;i<nbigcls;i++) {
    dist[i]=(float *)calloc(nbigcls,sizeof(float));
    dist[i][i]=0.0;
  }

  for (m=0;m<nbigcls;m++) {
    for (n=m+1;n<nbigcls;n++) {
      v1=((double)rct[orgid[m]])/((double)rct[orgid[m]]+rct[orgid[n]]);//compute relative freq of first model
      ridgeline(mds+orgid[m],mds+orgid[n],v1,alpha, na, pdfv);
      v2=separability(pdfv,na);
      dist[m][n]=v2;
      dist[n][m]=v2;

      //Output the result
      fprintf(outfile, "%d %d ", orgid[m],orgid[n]);
      for (i=0;i<na;i++)
	fprintf(outfile, "%e ", pdfv[i]);
      fprintf(outfile, "%e\n",v2);
    }
  }

  if (!linkage) exit(0); //no need to go further

  /*-----------------------------------------------------*/
  /*------------ Linkage clustering start here ----------*/
  /*-----------------------------------------------------*/
  float disthred;
  int numnd,*newcls;
  TreeNode **ndlist;

  newcls=(int *)calloc(nbigcls,sizeof(int));
  //merge until a single node
  aggcluster(dist,nbigcls,HUGE,0,1,newcls, &ndlist, &numnd); //single linkage flag=0 used here

  fprintf(stderr, "numnd=%d\n",numnd);

  //Tree cutting and other info
  int *ids, nd, nstep;
  float *mdist, *dlist, dstepsz=0.05;

  fprintf(stdout, "Original number of clusters: %d, Qualified big clusters: %d\n",rncls,nbigcls);
  fprintf(stdout, "Original cluster sizes (ID, Size): ");
  for (i=0;i<rncls;i++) {
    fprintf(stdout, "(%d, %d) ",i, rct[i]);
  }
  fprintf(stdout, "\n");

  find_distances(ndlist[0],&nd, &mdist, &ids);
  fprintf(stdout, "#nodes separability_cutoff\n");
  for (i=0;i<nd;i++){
    n=2*nbigcls-ids[i]-1;//number of clusters formed
    fprintf(stdout, "%d %f\n",n,mdist[i]);
  }

  nstep=(int)(1.0/dstepsz)+1;
  dlist=(float *)calloc(nstep,sizeof(float));
  for (i=0;i<nstep;i++)
    dlist[i]=((float)i)*dstepsz;

  for (i=0;i<nstep;i++) {
    if (dlist[i]<mdist[0] || (i>0 && dlist[i-1]>mdist[nd-1])) continue;
    cuttree_dist(ndlist[0], dlist[i], newcls);
    fprintf(stdout, "===At distance threshold: %f ===\n",dlist[i]);
    print_cluster(newcls,nbigcls,orgid,stdout);
  }

  for (i=0;i<numnd;i++)
    freetreenode(ndlist+i);
  free(ndlist);

}



