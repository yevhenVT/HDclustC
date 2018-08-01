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

int DIAGCOV;

unsigned char mat_logdet_inv_diag_double(double **mt, double **y, double *det_log, 
					 int dim, int diagonal)
{
  double d, *col;
  int i,j,m,n,mm;
  double **a;
  double *ptr1, *ptr2,v1;

  if (diagonal!=1) {
    mm=mat_det_inv_double(mt,y,&v1,dim);
    if (v1>0.0) *det_log=log(v1);
    else {mm=2;} //return bad code 2 when the not positive definite
    return(mm);
  }

  /** initialize matrix determinant **/
  *det_log=0.0;
  mm=1;
  for (i=0;i<dim;i++) {
    (*det_log) += log(mt[i][i]);
    if (mt[i][i]<=0.0) mm=2;
  }
  for (i=0;i<dim;i++)
    for (j=0;j<dim;j++) y[i][j]=0.0;
  for (i=0;i<dim;i++)
    y[i][i]=1.0/mt[i][i];

  return(mm);
}

void forward(double *u, double *thetalog, CondChain *md, double *loglikehd)
// thetalog[nb*numst?] is the log of the forward probabilities
// md is the input CondChain
// *loglikehd is the log likelihood for the entire sequence
{
  int i,j,k,l,m,n,ii,jj,mm,nn;
  int precls, curcls;
  int *numst, *cnumst, maxnumst, nb, *cbdim;
  double *dbpt,v1,v2,v3,maxv,*buf, *a, **a2;

  nb=md->nb;
  numst=md->numst; //number of states per block
  cnumst=md->cnumst; //cumulated starting position of each block
  maxnumst=md->maxnumst;
  cbdim=md->cbdim;

  buf=(double *)calloc(maxnumst,sizeof(double));

  /* treat the first position */
  a=md->mds[0]->a00;

  for (l=0; l<numst[0]; l++) {
    v1=gauss_pdf_log(u, md->mds[0]->stpdf[l]);
    if (a[l]>0.0)
      thetalog[l]=log(a[l])+v1;
    else {
      thetalog[l]=-HUGE;
      if (v1<0.0) thetalog[l]+=v1; //push thetalog even smaller
      //fprintf(stderr, "Warning: prior prob for a state is zero, -HUGE=%e used, thetalog=%e, dif=%e\n",
	    //  -HUGE, thetalog[l], v1);
    }
  }

  /* treat the rest columns */
  for (jj=1; jj<nb; jj++) {
    for (l=0;l<numst[jj-1];l++) {
      buf[l]=thetalog[cnumst[jj-1]+l];
    }
    maxv=buf[0];
    for (l=0; l<numst[jj-1]; l++)
      if (buf[l]>maxv) maxv=buf[l];
    
    a2=md->mds[jj]->a;
    
    for (m=0, mm=cnumst[jj]; m<numst[jj]; m++,mm++) {
	v3=gauss_pdf_log(u+cbdim[jj], md->mds[jj]->stpdf[m]);
	
	for (l=0,v1=0.0;l<numst[jj-1];l++) {
	  v1+=exp(buf[l]-maxv)*a2[l][m];
	}
	if (v1>0.0)
	  thetalog[mm]=maxv+log(v1)+v3;	  
	else {
	  thetalog[mm]=-HUGE;
	  if (maxv+v3<0) thetalog[mm]+=(maxv+v3); //push thetalog even smaller
	  //fprintf(stderr, "Warning: -HUGE=%e used for thetalog=%e, dif=%e\n",-HUGE, thetalog[mm], maxv+v3);
	}
    }
  }
  

  /** compute loglikelihood for the image **/
  v3=0.0;
  dbpt=thetalog+cnumst[nb-1];
  v3=dbpt[0];
  for (m=0; m<numst[nb-1]; m++) {
    if (dbpt[m]>v3) v3=dbpt[m];
  }
  for (m=0,v1=0.0; m<numst[nb-1]; m++) {
    v1+=exp(dbpt[m]-v3);
  }
  v3+=log(v1);
  
  *loglikehd=v3;

  free(buf); 
}


void backward(double *u, double *betalog, CondChain *md)
// betalog[nb*numst?] is the log of the forward probabilities
// md is the input HMM
{
  int i,j,k,l,m,n,ii,jj,mm,nn;
  int nextcls, curcls;
  int *numst, *cnumst, maxnumst, nb, *cbdim;
  double *dbpt,v1,v2,v3,maxv,*buf, *a, **a2;

  nb=md->nb;
  numst=md->numst; //number of states per block
  cnumst=md->cnumst; //cumulated starting position of each block
  maxnumst=md->maxnumst;
  cbdim=md->cbdim;

  buf=(double *)calloc(maxnumst,sizeof(double));

  /* treat the last block */
  for (l=0; l<numst[nb-1]; l++) {
    betalog[cnumst[nb-1]+l]=0.0;
  }

  /* treat the rest blocks */  
  for (jj=nb-2; jj>=0; jj--) {
    for (l=0; l<numst[jj+1]; l++) {
      buf[l]=betalog[cnumst[jj+1]+l]+
	gauss_pdf_log(u+cbdim[jj+1],md->mds[jj+1]->stpdf[l]);
    }
    
    maxv=buf[0];
    for (l=0; l<numst[jj+1]; l++)
      if (buf[l]>maxv) maxv=buf[l];
    
    a2=md->mds[jj+1]->a;
    
    for (m=0, mm=cnumst[jj]; m<numst[jj]; m++,mm++) {
      for (l=0,v1=0.0;l<numst[jj+1];l++) {
	v1+=exp(buf[l]-maxv)*a2[m][l];
      }
      if (v1>0.0)
	betalog[mm]=maxv+log(v1);
      else {
	betalog[mm]=-HUGE;
	if (maxv<0) betalog[mm]+=maxv; //push betalog even smaller
	//fprintf(stderr, "Warning: -HUGE=%e used for betalog=%e, dif=%e\n",-HUGE, betalog[mm], maxv);
      }
    }
  }

  free(buf);
}


void CompLm(double *thetalog, double *betalog, double **Lm, CondChain *md)
     /* Lm=double[nb][numst], space allocated */
{
  int t,i,j,k,l,m,n;
  double *curLm,v1,v2;
  int curcls;
  int *numst, *cnumst, nb;

  nb=md->nb;
  numst=md->numst; //number of states per block
  cnumst=md->cnumst; //cumulated starting position of each block

  for (j=0; j<nb; j++) {
    curLm=Lm[j];
    for (m=0; m<numst[j];m++)
      curLm[m]=thetalog[cnumst[j]+m]+betalog[cnumst[j]+m];
    
    // Robust normalization
    v1=curLm[0];
    for (m=0; m<numst[j]; m++)
      if (curLm[m]>v1) v1=curLm[m];
    
    for (m=0,v2=0.0;m<numst[j];m++){
      curLm[m]=exp(curLm[m]-v1);
      v2+=curLm[m];
    }

    for (m=0;m<numst[j];m++) 
      curLm[m]/=v2;
  }
}


void CompHml(double *u, double *thetalog, double *betalog, double ***Hml, CondChain *md)
     /* Hml=double[nb][prenumst][numst], space allocated */
{
  int t,i,j,k,l,m,n,ll;
  double *curHml,v1,v2;
  double loglikehd, *dbpt;
  int *numst, *cnumst, nb, *cbdim;

  nb=md->nb;
  numst=md->numst; //number of states per block
  cnumst=md->cnumst; //cumulated starting position of each block
  cbdim=md->cbdim;

  /* Hml, m is fixed state of the previous neighbor */

  /* compute the log likelihood for the whole sequence */
  loglikehd=0.0;
  dbpt=thetalog+cnumst[nb-1];

  // find the largest log-likelihood for the last block;
  // since block is the last, beta = 1 and log beta = 0
  loglikehd=dbpt[0];
  for (m=0; m<numst[nb-1]; m++) {
      if (dbpt[m]>loglikehd) loglikehd=dbpt[m];
  }

  for (m=0,v1=0.0; m<numst[nb-1]; m++) {
      v1+=exp(dbpt[m]-loglikehd);
  }
  loglikehd+=log(v1);
  
  //special treat of the first block
  for (l=0; l<numst[0];l++) Hml[0][0][l]=1.0/(double)numst[0]; 

  for (j=1; j<nb; j++) {
    for (k=0;k<numst[j-1];k++) {
      for (l=0;l<numst[j];l++) {
		Hml[j][k][l]=-loglikehd+thetalog[cnumst[j-1]+k]+betalog[cnumst[j]+l]+
					gauss_pdf_log(u+cbdim[j],md->mds[j]->stpdf[l]);
		Hml[j][k][l]=exp(Hml[j][k][l])*md->mds[j]->a[k][l];
      }	
    }
  }
}


void updatepar_adder(double *u, double *thetalog, double *betalog, CondChain *md, 
		     double **musum, double ***mom2sum, double ***Hml, double **Lm)
//musum[nb][numst*dim],mom2sum[nb][numst*dim][numst*dim],
//Hml[nb][prenumst][numst], Lm[nb][numst] allocated
{
  int i,j,k,l,m,n,ii,jj,mm,nn,ll,t,k1,k2;
  double v1,v2,tpdb,epsilon=1.0e-2;
  int *numst, nb, *cbdim, *bdim;

  nb=md->nb;
  numst=md->numst; //number of states per block
  cbdim=md->cbdim;
  bdim=md->bdim;

  CompLm(thetalog, betalog, Lm, md);
  CompHml(u, thetalog, betalog, Hml, md);

  for (i=0; i<nb; i++) {
    for (m=0; m<numst[i]; m++) {
      for (jj=0; jj<bdim[i]; jj++)
		musum[i][m*bdim[i]+jj]= Lm[i][m]*u[cbdim[i]+jj];
      
      /* mom2sum is the second order moment */
      /* covariance is second moment minus the square of the mean */
      // For HMM-VB, mom2sum doesn't sum up, but store the result for each variable block
      if (DIAGCOV==1) {
	for (ii=0; ii<bdim[i]; ii++) 
	  mom2sum[i][m*bdim[i]+ii][ii]=Lm[i][m]*u[cbdim[i]+ii]*u[cbdim[i]+ii];
      }
      else {
	for (ii=0; ii<bdim[i]; ii++) 
	  for (jj=0; jj<bdim[i]; jj++)
	    mom2sum[i][m*bdim[i]+ii][jj]=Lm[i][m]*u[cbdim[i]+ii]*u[cbdim[i]+jj];
      }
    }
  }
  
}

/*-------------------------------*/
/* Viterbi for a single sequence */
/*-------------------------------*/
void viterbi(CondChain *md, double *u, int *optst, double *inita, double *lastmerit)
// optst stores the optimal sequence of states with maximum posterior
{
  int i,j,k,l,m,n;
  double *merit, **a, *astart;
  int *prest;
  double v1,v2,v3,db1,db2,db3;
  int *numst, nb, *cbdim, *bdim,maxnumst;

  nb=md->nb;
  numst=md->numst; //number of states per block
  cbdim=md->cbdim;
  bdim=md->bdim;
  maxnumst=md->maxnumst;

  prest=(int *)calloc(nb*maxnumst,sizeof(int)); // best state in the previous block
  merit=(double *)calloc(nb*maxnumst,sizeof(double));

  if (inita==NULL)
    astart=md->mds[0]->a00;
  else
    astart=inita;
  
  /* treat the first location */
  for (l=0; l<numst[0]; l++) {
    v1=gauss_pdf_log(u, md->mds[0]->stpdf[l]);
    if (astart[l]>0.0) {
      merit[l]=log(astart[l])+v1;
    }
    else {
      merit[l]=-HUGE;
      if (v1<0.0) merit[l]+=v1; //push even smaller
      fprintf(stderr, "Warning: prior prob for a state is zero, -HUGE=%e used, dif=%e\n",-HUGE,v1);
    }
  }

  /* treat the rest locations */
  for (j=1; j<nb; j++) {
    a=md->mds[j]->a;

    for (l=0; l<numst[j]; l++) {
      v1=gauss_pdf_log(u+cbdim[j], md->mds[j]->stpdf[l]);
      db1=merit[(j-1)*maxnumst];
      if (a[0][l]>0.0) {
		v2=db1+log(a[0][l]);
      }
      else {
		v2=-HUGE;
		if (db1<0.0) v2+=db1;
      }
      //if (a[0][l]==0.0) 
      //	fprintf(stderr, "Warning: Transition probability is zero, used -HUGE=%e\n",-HUGE);
      
      prest[j*maxnumst+l]=0;
      for (m=1; m<numst[j-1]; m++) {
		db3=merit[(j-1)*maxnumst+m];
		if (a[m][l]>0.0) {
			v3=db3+log(a[m][l]);
		}
		else {
			v3=-HUGE;
			if (db3<0.0) v3+=db3;
		}
	//if (a[m][l]==0.0) 
	// fprintf(stderr, "Warning: Transition probability is zero, used -HUGE=%e\n",-HUGE);
	
		if (v2<v3) {
			v2=v3;
			prest[j*maxnumst+l]=m;
		}
      }
      merit[j*maxnumst+l]=v2+v1;
    }
  }

  m=0;
  v1=merit[(nb-1)*maxnumst];
  for (l=1;l<numst[nb-1];l++) {
    if (merit[(nb-1)*maxnumst+l]>v1) {
      v1=merit[(nb-1)*maxnumst+l];
      m=l;
    }
  }
      
  if (lastmerit!=NULL) {
    for (l=0;l<numst[nb-1];l++) lastmerit[l]=merit[(nb-1)*maxnumst+l];
  }

  optst[nb-1]=m;
  for (j=nb-2; j>=0; j--) {
    optst[j]=prest[(j+1)*maxnumst+optst[j+1]];
  }
  
  free(prest);
  free(merit);
}


/*--------------------------------*/
/* Viterbi for multiple sequences */
/*--------------------------------*/
void viterbi_mulseq(CondChain *md, double **u, int nseq, int **st)
     /* st=int[nseq][len[?]], space allocated */
{
  int i,j,k,l,m,n;

  /* treat the rest rows */
  for (i=0; i<nseq; i++) {
    viterbi(md, u[i], st[i], NULL, NULL);
  }
}


/*-------------------------------*/
/** initialization of variable block using kmeans **/
/*-------------------------------*/
void initialize(double *u, int nseq, int dim, HmmModel *md, int ranflag)
{
  int i,j,k,l,m,n,n_local,nseq_total, ii,jj, mm,nn, mm2, k1, k2;
  int numst, prenumst;
  int rank; // rank of MPI process
  double *cdbk; // centroid coordinates
  int *code;
  double tpdb, epsilon=1.0e-2, lambda=0.5;
  double **sigma, **sigma_local, **sigma_inv, *mu, *mu_local, **sigcom, **sigcom_local, mysigma_det;
 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /** use kmeans to decide the initial states **/
  numst=md->numst;
  prenumst=md->prenumst;
  
  code=(int *)calloc(nseq,sizeof(int));
  cdbk=(double *)calloc(numst*dim,sizeof(double)); //May 9, 2016
 
  matrix_2d_double(&sigcom, dim, dim);
  matrix_2d_double(&sigcom_local, dim, dim);
  matrix_2d_double(&sigma_local, dim, dim);
  mu_local=(double *)malloc(dim*sizeof(double));
  
  if (!ranflag) {
    lloyd(cdbk,dim,numst,u,nseq,1.0e-4);
    encode(cdbk,dim,numst,u,code,nseq);  }
  else {
    srand48(0);
    for (i=0;i<nseq;i++) {
      code[i]=(int)(drand48()*numst);
      if (code[i]>=numst) code[i]=numst-1;
    }
  }

 
  
  /* compute gaussian mean */
  for (m=0; m<numst; m++) {
    mu = md->stpdf[m]->mean;
    
    for (jj=0; jj<dim; jj++) mu_local[jj]=0.0; // initizalize mean to with zero
    for (i=0,n_local=0; i<nseq; i++) {
      if (code[i]==m) {
		n_local++;
		for (jj=0; jj<dim; jj++) mu_local[jj]+=u[i*dim+jj];
      }
    }
    
    // gather on all processes
    MPI_Allreduce(&n_local, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(mu_local, mu, dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (n==0) { //don't divide for empty cell
      for (jj=0; jj<dim; jj++) mu[jj]=0.0; // redundant line, since it was initialized to 0 before
    }
    else {
      for (jj=0; jj<dim; jj++) mu[jj]/=(double)n;
    }
  }


  // Compute common covariance matrix
  for (k1=0; k1<dim; k1++)
    for (k2=0; k2<dim; k2++)
      sigcom_local[k1][k2]=0.0; // initialize with zeros
  
  for (i=0; i<nseq; i++) { //forced diagonal
    for (k1=0; k1<dim; k1++)
      sigcom_local[k1][k1]+=(u[i*dim+k1]-md->stpdf[code[i]]->mean[k1])*
			(u[i*dim+k1]-md->stpdf[code[i]]->mean[k1]);
  }
  
  // send common covariance matrix to master process
  for (k1=0; k1<dim; k1++)
	MPI_Allreduce(sigcom_local[k1], sigcom[k1], dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  MPI_Allreduce(&nseq, &nseq_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  for (k1=0; k1<dim; k1++) {
	sigcom[k1][k1]/=(double)nseq_total; //forced diagonal
  }

  /* compute gaussian mean and covariance matrix */
  for (m=0; m<numst; m++) { 
	mu=md->stpdf[m]->mean;
	sigma=md->stpdf[m]->sigma;
	sigma_inv=md->stpdf[m]->sigma_inv;
    
    for (i=0,n_local=0; i<nseq; i++) {
      if (code[i]==m) 	n_local++;
    }

	MPI_Allreduce(&n_local, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (n==0) { //don't divide for empty cell
	  for (k1=0; k1<dim; k1++) 
		for (k2=0; k2<dim; k2++)
			sigma[k1][k2]=sigcom[k1][k2];  // use common covariance
    }
    else {
      for (k1=0; k1<dim; k1++)
		for (k2=0; k2<dim; k2++)
		  sigma_local[k1][k2]=0.0; //initialize

	  //Add up
	  for (i=0; i<nseq; i++) {
		if (code[i]==m) {
		  if (DIAGCOV==1) {
			for (k1=0; k1<dim; k1++)
			  sigma_local[k1][k1]+=(u[i*dim+k1]-mu[k1])*(u[i*dim+k1]-mu[k1]);
		  }
		  else {
			for (k1=0; k1<dim; k1++)
			  for (k2=k1; k2<dim; k2++) {
				sigma_local[k1][k2]+=(u[i*dim+k1]-mu[k1])*(u[i*dim+k2]-mu[k2]);
			  }
		  }
		}
	  }

	  // send covariance matrix to master process
	  for (k1=0; k1<dim; k1++)
		MPI_Allreduce(sigma_local[k1], sigma[k1], dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
	  
	  //Normalize
	  if (DIAGCOV==1) {
		for (k1=0; k1<dim; k1++)
		  sigma[k1][k1]/=(double)n;
	  }
	  else {
		for (k1=0; k1<dim; k1++)
		  for (k2=k1; k2<dim; k2++) {
			sigma[k1][k2]/=(double)n;
			if (k2!=k1) sigma[k2][k1]=sigma[k1][k2];
		  }
	  }

	  for (k1=0; k1<dim; k1++)
		for (k2=0; k2<dim; k2++) {
		  if (!ranflag) //dilate slightly
			sigma[k1][k2]=sigma[k1][k2]*lambda+(1.1-lambda)*sigcom[k1][k2]; 
		  else
			sigma[k1][k2]=sigma[k1][k2]*lambda+(1.0-lambda)*sigcom[k1][k2];

		  }
    }

	
	mm=mat_logdet_inv_diag_double(sigma,sigma_inv, &(md->stpdf[m]->sigma_det_log), 
				  dim, DIAGCOV);

	mm2 = 0;

	if (mm==2) {
		if (rank == 0) fprintf(stderr, "Covariance matrix: not positive definite\n");
	}
	
	
	if (mm==2) { /* singular matrix or ill-conditioned*/
		for (k1=0, tpdb=0.0; k1<dim; k1++) tpdb+=sigma[k1][k1];
		tpdb=(tpdb>0.0)?(tpdb/(double)dim*epsilon):epsilon;
		/* modify sigma by adding a scalar matrix */
		for (k1=0; k1<dim; k1++) sigma[k1][k1]+=tpdb;
			mm2=mat_logdet_inv_diag_double(sigma, sigma_inv, &(md->stpdf[m]->sigma_det_log),dim, DIAGCOV);
	}
		
	if (mm2==2) {
		if (rank == 0) fprintf(stderr, "Non-positive definite covariance matrix after adding scalar matrix, abort\n");
		MPI_Finalize();
		exit(1);	
	}
    
   
  }
  
  /** Set the transition probabilities to uniform **/
  
  tpdb=1.0/(double)numst;
  for (i=0;i<numst;i++) md->a00[i]=tpdb;
  for (i=0;i<prenumst;i++) 
	for (j=0; j<numst; j++)
	  md->a[i][j]=tpdb;
  
  free(cdbk);
  free(code);
  free(mu_local);
  free_matrix_2d_double(&sigcom, dim);
  free_matrix_2d_double(&sigcom_local, dim);
  free_matrix_2d_double(&sigma_local, dim);
}

//Another scheme of initialization, set codebook by the given input
void initialize2(double *u, int nseq, int dim, HmmModel *md, double *cdbk)
{
  int i,j,k,l,m,n,n_local,ii,jj, mm,nn, mm2, k1, k2;
  int nseq_total; // total number of samples in dataset
  int numst, prenumst;
  int rank; // rank of MPI process
  int *code;
  double tpdb, epsilon=1.0e-2, lambda=0.5;
  double **sigma, **sigma_local, **sigma_inv, *mu, *mu_local, **sigcom, **sigcom_local;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  /** use kmeans to decide the initial states **/
  numst=md->numst;
  prenumst=md->prenumst;

  code=(int *)calloc(nseq,sizeof(int));
  mu_local=(double *)malloc(dim*sizeof(double));

  matrix_2d_double(&sigcom_local, dim, dim);
  matrix_2d_double(&sigma_local, dim, dim);
  matrix_2d_double(&sigcom, dim, dim);

  encode(cdbk,dim,numst,u,code,nseq);    

  /* compute gaussian mean */
  for (m=0; m<numst; m++) {
    mu=md->stpdf[m]->mean;
    for (jj=0; jj<dim; jj++) mu_local[jj]=0.0;
    for (i=0,n_local=0; i<nseq; i++) {
      if (code[i]==m) {
		n_local++;
		for (jj=0; jj<dim; jj++) mu_local[jj]+=u[i*dim+jj];
      }
    }
    
    // gather on all processes
    MPI_Allreduce(&n_local, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(mu_local, mu, dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (n==0) { //don't divide for empty cell
      for (jj=0; jj<dim; jj++) mu[jj]=0.0;
    }
    else {
      for (jj=0; jj<dim; jj++) mu[jj]/=(double)n;
    }
  }


  // Compute common covariance matrix
  for (k1=0; k1<dim; k1++)
    for (k2=0; k2<dim; k2++)
      sigcom_local[k1][k2]=0.0;
  for (i=0; i<nseq; i++) { //forced diagonal
    for (k1=0; k1<dim; k1++)
      sigcom_local[k1][k1]+=(u[i*dim+k1]-md->stpdf[code[i]]->mean[k1])*
			(u[i*dim+k1]-md->stpdf[code[i]]->mean[k1]);
  }
  
 // send common covariance matrix to master process
  for (k1=0; k1<dim; k1++)
	MPI_Allreduce(sigcom_local[k1], sigcom[k1], dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  MPI_Allreduce(&nseq, &nseq_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  for (k1=0; k1<dim; k1++) {
    sigcom[k1][k1]/=(double)nseq_total; //forced diagonal
  }
  
  /* compute gaussian mean and covariance matrix */
  for (m=0; m<numst; m++) {
    mu=md->stpdf[m]->mean;
    sigma=md->stpdf[m]->sigma;
    sigma_inv=md->stpdf[m]->sigma_inv;
    
    for (i=0,n_local=0; i<nseq; i++) {
      if (code[i]==m) 	n_local++;
    }
	
	MPI_Allreduce(&n_local, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (n==0) { //don't divide for empty cell
      for (k1=0; k1<dim; k1++) 
		for (k2=0; k2<dim; k2++)
			sigma[k1][k2]=sigcom[k1][k2];  // use common covariance
    }
    else {
      for (k1=0; k1<dim; k1++)
		for (k2=0; k2<dim; k2++)
			sigma_local[k1][k2]=0.0; //initialize

      //Add-up
      for (i=0; i<nseq; i++) {
		if (code[i]==m) {
			if (DIAGCOV==1) {
				for (k1=0; k1<dim; k1++)
					sigma_local[k1][k1]+=(u[i*dim+k1]-mu[k1])*(u[i*dim+k1]-mu[k1]);
			}
			else {
				for (k1=0; k1<dim; k1++)
					for (k2=k1; k2<dim; k2++)
						sigma_local[k1][k2]+=(u[i*dim+k1]-mu[k1])*(u[i*dim+k2]-mu[k2]);
			}
		}
      }
	  
	  // send covariance matrix to master process
	  for (k1=0; k1<dim; k1++)
		MPI_Allreduce(sigma_local[k1], sigma[k1], dim, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
      //Normalize
      if (DIAGCOV==1) {
		for (k1=0; k1<dim; k1++)
			sigma[k1][k1]/=(double)n;
      }
      else {
		for (k1=0; k1<dim; k1++)
			for (k2=k1; k2<dim; k2++) {
				sigma[k1][k2]/=(double)n;
				if (k2!=k1) sigma[k2][k1]=sigma[k1][k2];
			}
      }

      for (k1=0; k1<dim; k1++)
		for (k2=0; k2<dim; k2++) {
			//dilate slightly
			sigma[k1][k2]=sigma[k1][k2]*lambda+(1.1-lambda)*sigcom[k1][k2]; 
		}
    }

    mm=mat_logdet_inv_diag_double(sigma,sigma_inv, &(md->stpdf[m]->sigma_det_log), dim, DIAGCOV);
    
    if (mm==2) {
      if (rank == 0) fprintf(stderr, "Covariance matrix: not positive definite\n");
    }
    
    if (mm==2) { /* singular matrix */
      for (k1=0, tpdb=0.0; k1<dim; k1++) tpdb+=sigma[k1][k1];
      tpdb=(tpdb>0.0)?(tpdb/(double)dim*epsilon):epsilon;
      /* modify sigma by adding a scalar matrix */
      for (k1=0; k1<dim; k1++) sigma[k1][k1]+=tpdb;
      mm2=mat_logdet_inv_diag_double(sigma, sigma_inv, &(md->stpdf[m]->sigma_det_log),dim, DIAGCOV);

      if (mm2==2) {
		if (rank == 0) fprintf(stderr, "Non-positive definite covariance matrix after adding scalar matrix, abort\n");
		MPI_Finalize();
		exit(1);
      }
    }
  }
  
  /** Set the transition probabilities to uniform **/
  tpdb=1.0/(double)numst;
  for (i=0;i<numst;i++) md->a00[i]=tpdb;
  for (i=0;i<prenumst;i++) 
    for (j=0; j<numst; j++)
      md->a[i][j]=tpdb;
  
  free(code);
  free_matrix_2d_double(&sigcom, dim);
  free_matrix_2d_double(&sigcom_local, dim);
  free_matrix_2d_double(&sigma_local, dim);
  free(mu_local);
}

// initialize mixture models in variable blocks with lloyd algorithm
void initial_ccm(double **u,int nseq, CondChain *md)
{
  int i,j,k,l,m,n,ii;
  double *ublock;
  int nb, *bdim, *cbdim, **var, *numst;

  nb=md->nb; bdim=md->bdim; numst=md->numst; cbdim=md->cbdim;
  
  for (i=0,m=0;i<nb;i++) {if (bdim[i]>m) m=bdim[i];} // find largest dimension in blocks
	ublock=(double *)calloc(nseq*m,sizeof(double));
  
  for (ii=0;ii<nb;ii++) {
    //fprintf(stdout, "inside initial_ccm, ii=%d, bdim[ii]=%d, cbdim[ii]=%d\n",ii,bdim[ii], md->cbdim[ii]);
    //take the current block of variables
    for (i=0;i<nseq;i++) 
      for (j=0;j<bdim[ii];j++)
		ublock[i*bdim[ii]+j]=u[i][cbdim[ii]+j]; 
    initialize(ublock, nseq, bdim[ii], md->mds[ii],0);//use clustering to initialize
  }
  
  free(ublock);
}

void initial_ccm1(double **u,int nseq, CondChain *md, int sd)
{
  int i,j,k,l,m,n,ii;
  double *ublock;
  int nb, *bdim, *cbdim, **var, *numst;
  int nseq2, factor=5, *id;
  double **u2, *buf,*buf2;
 
  // send variable block info to slave processes
  nb=md->nb; bdim=md->bdim; cbdim=md->cbdim; numst=md->numst;
 
  // select smaller number of samples
  nseq2=(nseq/factor>100)? (nseq/factor):100;
  if (nseq2>nseq) nseq2=nseq;

  // select randomly nseq2 samples
  for (i=0,m=0;i<nb;i++) {if (bdim[i]>m) m=bdim[i];} // find largest dimension
  ublock=(double *)calloc(nseq2*m,sizeof(double));
  u2=(double **)calloc(nseq2,sizeof(double *));

  buf=(double *)calloc(nseq,sizeof(double));
  buf2=(double *)calloc(nseq,sizeof(double));
  id=(int *)calloc(nseq,sizeof(int));
  srand48(sd);
  for (i=0;i<nseq;i++) buf[i]=drand48();
  SortDouble(buf,buf2,id,nseq);
  for (i=0;i<nseq2;i++) u2[i]=u[id[i]];
  free(buf);
  free(buf2);
  free(id);

  for (ii=0;ii<nb;ii++) {
    //fprintf(stdout, "inside initial_ccm1, ii=%d, bdim[ii]=%d, cbdim[ii]=%d\n",ii,bdim[ii], md->cbdim[ii]);
    //take the current block of variables
    for (i=0;i<nseq2;i++) 
      for (j=0;j<bdim[ii];j++)
		ublock[i*bdim[ii]+j]=u2[i][cbdim[ii]+j]; 
    
    initialize(ublock, nseq2, bdim[ii], md->mds[ii],0);//use clustering to initialize
  }

  free(ublock);
  free(u2);
}

void initial_ccm2(double **u,int nseq, CondChain *md, int sd)
{
  int i,j,k,l,m,n,ii;
  double *ublock;
  int nb, *bdim, *cbdim, **var, *numst;
  double *cdbk;

  
  // send variable block info to slave processes
  nb=md->nb; bdim=md->bdim; cbdim=md->cbdim; numst=md->numst;
 
  // find largest number of dimensions and mixture states in blocks
  for (i=0,m=0,n=0;i<nb;i++) {
    if (bdim[i]>m) m=bdim[i];
    if (numst[i]>n) n=numst[i];
  }
  
  ublock=(double *)calloc(nseq*m,sizeof(double));
  cdbk=(double *)calloc(n*m,sizeof(double));

  double *buf, *buf2;
  int *id;
  buf=(double *)calloc(nseq,sizeof(double));
  buf2=(double *)calloc(nseq,sizeof(double));
  id=(int *)calloc(nseq,sizeof(int));
  srand48(sd);
  for (i=0;i<nseq;i++) buf[i]=drand48();
  SortDouble(buf,buf2,id,nseq);
  free(buf);
  free(buf2);
    
  for (ii=0;ii<nb;ii++) {
    //fprintf(stdout, "inside initial_ccm2, ii=%d, bdim[ii]=%d, cbdim[ii]=%d\n",ii,bdim[ii], md->cbdim[ii]);
    //take the current block of variables
    for (i=0;i<nseq;i++) 
      for (j=0;j<bdim[ii];j++)
		ublock[i*bdim[ii]+j]=u[i][cbdim[ii]+j]; 

    for (i=0;i<numst[ii];i++) {
      for (j=0;j<bdim[ii];j++) cdbk[i*bdim[ii]+j]=u[id[i]][cbdim[ii]+j]; 
    }

    initialize2(ublock, nseq, bdim[ii], md->mds[ii],cdbk);//use clustering to initialize
  }
  
  free(ublock);
  free(cdbk);
  free(id);
}

/*-----------------------------------------------------------*/
/** compute the log likelihood of sequences under HMM        */
/*-----------------------------------------------------------*/
double comploglike(CondChain *md, double **u, int nseq, double *wt, double *logl)
// weight wt can be turned off by setting it to NULL
// output logl[nseq] has space allocated
{
  int i,j,k,m,n;
  double *thetalog,v1, loglikehd;;
  int *numst, nb;

  nb=md->nb;
  numst=md->numst; //number of states per block

  for (i=0,m=0;i<nb;i++) m+=numst[i];
  thetalog=(double *)calloc(m, sizeof(double));

  for (i=0, loglikehd=0.0; i<nseq; i++) {
    forward(u[i],thetalog,md,logl+i);
    if (wt==NULL) loglikehd+=logl[i];
    else loglikehd += wt[i]*logl[i];
  }

  free(thetalog);
  return(loglikehd);
}
	

/*-----------------------------------------------------------*/
/** compute the probability of each position existing under  */
/** a particular state given feature vectors for the entire  */
/** sequence and under a given HMM.                          */
/** This subroutine also returns the log likelihood as the   */
/** function comploglike.                                    */
/*-----------------------------------------------------------*/
double classlikehd(CondChain *md, double **u, int nseq, double ***cprob, double *wt)
     /* cprob[nseq][nb][numst] has been allocated with space */
{
  double loglikehd,v1;
  double *thetalog,*betalog;
  double dbtp;
  int i,j,k,m,n,ii,mm;
  int *numst, nb;

  nb=md->nb;
  numst=md->numst; //number of states per block

  for (i=0,m=0;i<nb;i++) m+=numst[i];
  thetalog=(double *)calloc(m, sizeof(double));
  betalog=(double *)calloc(m, sizeof(double));

  loglikehd=0.0;
  for (ii=0;ii<nseq;ii++) {
    forward(u[ii], thetalog, md, &v1);
    backward(u[ii], betalog, md);
    CompLm(thetalog, betalog, cprob[ii], md);

    if (wt==NULL) loglikehd+= v1;
    else loglikehd+= wt[ii]*v1;

    /* normalization */
    for (j=0; j<nb; j++) {
      for (m=0,dbtp=0.0; m<numst[j]; m++) 
	dbtp+=cprob[ii][j][m];
      if (dbtp>0.0) {
	for (m=0; m<numst[j]; m++) cprob[ii][j][m]/=dbtp;
      }
      else {
	for (m=0; m<numst[j]; m++) cprob[ii][j][m]=1.0/(double)numst[j];
      }
    }
  }

  free(thetalog);
  free(betalog);
  return(loglikehd);
}


void transprob(double ***asum, CondChain *md)
//asum[nb][prenumst][numst] allocated
{
  int i,j,k,m,n,l;
  double v1,v2,v3;
  int *numst, nb, prenumst;

  nb=md->nb;
  numst=md->numst; //number of states per block

  // Row wise normalization for asum[][][]
  for (m=0;m<nb;m++) {
    if (m>0) prenumst=numst[m-1]; else prenumst=1;
    for (k=0; k<prenumst; k++) {
      for (l=0,v1=0.0; l<numst[m]; l++)
		v1+=asum[m][k][l];
      if (v1>0.0) {
		for (l=0; l<numst[m]; l++)
			asum[m][k][l]/=v1;
      }
      else {
		for (l=0; l<numst[m]; l++)
			asum[m][k][l]=1.0/(double)numst[m];
      }
    }
  }
}


/*---------------------------------------------------------------*/
/** EM estimation assuming that the initial model is set up     **/
/*---------------------------------------------------------------*/
int baumwelch(double **u, int nseq, CondChain *md, double *loglikehd, 
	      double *lhsumpt, double epsilon, double *wt)
/* The only outputs are loglikehd, lhsumpt, and updated md */
// Input wt[nseq] gives a weight to each sequence, normally it contains all 1
{
  int i,j,k,l,m,n,mm,k1,m1,t,ite,minite=3, ii,mm2;
  int *nseq_proc; // array with number of samples on each process
  int *displs; // array with displacement to gather loglikelyhood
  int numproc; // number of MPI processes 
  double ratio=10.0, epsilon2=5.0e-2;
  double oldlhsum, lhsum, lhsum_local;
  double *thetalog, *betalog, *loglikehd_local;
  double **musum, **musum_local, **mu, ***mom2sum, ***mom2sum_local,***mom2, ***sigma;
  double ***asum, ***asum_local, ***a, **lsum, **lsum_local, **l1img;
  double ***sigcom, lambda=LAMBDA;
  int step;
  GaussModel *curg;
  HmmModel *oldmd;
  double v1,tpdb;
  int twomdflag=0;
  int res=0;
  int *numst, *cnumst, maxnumst, nb, *cbdim, *bdim, *prenumst;
  int errornonposcov = 0; // error indicating non positive covariance matrix

  nb=md->nb;
  numst=md->numst; //number of states per block
  cnumst=md->cnumst; //cumulated starting position of each block
  maxnumst=md->maxnumst;
  cbdim=md->cbdim;
  bdim=md->bdim;
  prenumst=(int *)calloc(nb,sizeof(int));
  for (i=0;i<nb;i++) prenumst[i]=md->mds[i]->prenumst;

  if (nseq==0) return(res);

  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  nseq_proc = (int *)malloc(numproc*sizeof(int));
  displs = (int *)calloc(numproc, sizeof(int));
  
  MPI_Allgather(&nseq, 1, MPI_INT, nseq_proc, 1, MPI_INT, MPI_COMM_WORLD);
  
  for (i=1; i<numproc; i++) displs[i] = displs[i-1] + nseq_proc[i-1];
  
  loglikehd_local=(double *)calloc(nseq, sizeof(double));
  
  musum_local=(double **)calloc(nb,sizeof(double *));
  musum=(double **)calloc(nb,sizeof(double *));
  mu=(double **)calloc(nb,sizeof(double *));
  mom2sum_local=(double ***)calloc(nb,sizeof(double **));
  mom2sum=(double ***)calloc(nb,sizeof(double **));
  mom2=(double ***)calloc(nb,sizeof(double **));
  sigma=(double ***)calloc(nb,sizeof(double **));
  sigcom=(double ***)calloc(nb,sizeof(double **));
  asum_local=(double ***)calloc(nb,sizeof(double **));
  asum=(double ***)calloc(nb,sizeof(double **));
  a=(double ***)calloc(nb,sizeof(double **));
  lsum_local=(double **)calloc(nb,sizeof(double *));
  lsum=(double **)calloc(nb,sizeof(double *));
  
  l1img=(double **)calloc(nb,sizeof(double *));

  for (i=0;i<nb;i++) {
    vector_double(musum_local+i, numst[i]*bdim[i]);
    vector_double(musum+i, numst[i]*bdim[i]);
    vector_double(mu+i, numst[i]*bdim[i]);
    matrix_2d_double(mom2sum+i, numst[i]*bdim[i], bdim[i]);
    matrix_2d_double(mom2sum_local+i, numst[i]*bdim[i], bdim[i]);
    matrix_2d_double(mom2+i, numst[i]*bdim[i], bdim[i]);
    matrix_2d_double(sigma+i, numst[i]*bdim[i], bdim[i]);
    matrix_2d_double(sigcom+i, bdim[i], bdim[i]);
    matrix_2d_double(asum+i, prenumst[i], numst[i]);
    matrix_2d_double(asum_local+i, prenumst[i], numst[i]);
    matrix_2d_double(a+i, prenumst[i], numst[i]);
    vector_double(lsum_local+i, numst[i]);
    vector_double(lsum+i, numst[i]);
    vector_double(l1img+i, numst[i]);
  }

  for (i=0,m=0;i<nb;i++) m+=numst[i]; // find total number of states
  thetalog=(double *)calloc(m, sizeof(double));
  betalog=(double *)calloc(m, sizeof(double));


  ite=0;
  twomdflag=0;
  oldlhsum=HUGE;

  while (ite<minite || twomdflag==0 || ratio>epsilon) {
    /* Initialization */
    for (ii=0;ii<nb;ii++) {
      for (i=0; i<numst[ii]; i++) {
		lsum_local[ii][i]=0.0;
		for (j=0; j<bdim[ii]; j++)
			musum_local[ii][i*bdim[ii]+j]=0.0;
		for (j=0; j<bdim[ii]; j++)
			for (k=0; k<bdim[ii]; k++) {
				mom2sum_local[ii][i*bdim[ii]+j][k]=0.0;
				mom2[ii][i*bdim[ii]+j][k]=0.0;
			}
      }
      
      for (j=0; j<prenumst[ii]; j++)
		for (k=0; k<numst[ii]; k++)
			asum_local[ii][j][k]=0.0;
    } // end loop over blocks

    for (t=0; t<nseq; t++) {
      forward(u[t],thetalog,md,loglikehd_local+t);
      backward(u[t],betalog, md);
      updatepar_adder(u[t],thetalog, betalog, md, mu, mom2, a, l1img);

      // Add over sequences
      for (ii=0;ii<nb;ii++) {
		for (i=0; i<numst[ii]; i++) {
			lsum_local[ii][i]+=wt[t]*l1img[ii][i];
			for (j=0; j<bdim[ii]; j++)
				musum_local[ii][i*bdim[ii]+j]+=wt[t]*mu[ii][i*bdim[ii]+j];
			if (DIAGCOV==1) {
				for (j=0; j<bdim[ii]; j++)
					mom2sum_local[ii][i*bdim[ii]+j][j]+=wt[t]*mom2[ii][i*bdim[ii]+j][j];
			}
			else {
				for (j=0; j<bdim[ii]; j++)
					for (k=0; k<bdim[ii]; k++)
						mom2sum_local[ii][i*bdim[ii]+j][k]+=wt[t]*mom2[ii][i*bdim[ii]+j][k];
			}
		}
		
		for (j=0; j<prenumst[ii]; j++)
			for (l=0; l<numst[ii]; l++)
				asum_local[ii][j][l]+=wt[t]*a[ii][j][l];
      }
    } // for (t=0; ...)
    
    // collect results from all processes
    for (ii=0;ii<nb;ii++) {
		MPI_Allreduce(lsum_local[ii], lsum[ii], numst[ii], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(musum_local[ii], musum[ii], numst[ii]*bdim[ii], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
		for (i=0; i<numst[ii]; i++)
			for (j=0; j< bdim[ii]; j++)
				MPI_Allreduce(mom2sum_local[ii][i*bdim[ii]+j], mom2sum[ii][i*bdim[ii]+j], bdim[ii], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		
		for (j=0; j<prenumst[ii]; j++)
			MPI_Allreduce(asum_local[ii][j], asum[ii][j], numst[ii], MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	}	
	
    /* Normalization */
    for (ii=0;ii<nb;ii++) {
      for (i=0; i<numst[ii]; i++) {
		for (j=0; j<bdim[ii]; j++)
			musum[ii][i*bdim[ii]+j]/=lsum[ii][i];
		if (DIAGCOV==1) {
			for (j=0; j<bdim[ii]; j++)
				mom2sum[ii][i*bdim[ii]+j][j]/=lsum[ii][i];
		}
		else {
			for (j=0; j<bdim[ii]; j++)
				for (k=0; k<bdim[ii]; k++)
					mom2sum[ii][i*bdim[ii]+j][k]/=lsum[ii][i];
		}
      }
      
      for (i=0; i<numst[ii]; i++) {
		if (DIAGCOV==1){
			for (j=0; j<bdim[ii]; j++)
				sigma[ii][i*bdim[ii]+j][j]=mom2sum[ii][i*bdim[ii]+j][j]-
									musum[ii][i*bdim[ii]+j]*musum[ii][i*bdim[ii]+j];
		}
		else {
			for (j=0; j<bdim[ii]; j++)
				for (k=0; k<bdim[ii]; k++) {
					sigma[ii][i*bdim[ii]+j][k]=mom2sum[ii][i*bdim[ii]+j][k]-
								musum[ii][i*bdim[ii]+j]*musum[ii][i*bdim[ii]+k];
				}
		}
      }
    }
    // asum adjustment
    transprob(asum, md);

    for (t=0,lhsum_local=0.0;t<nseq;t++) lhsum_local+=wt[t]*loglikehd_local[t];

	MPI_Allreduce(&lhsum_local, &lhsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
    // Judge whether to quit iteration loop
    if (twomdflag>0) {
      ratio=(lhsum-oldlhsum)/fabs(lhsum);
    }
    else {
      ratio=10.0;
    }

    oldlhsum=lhsum;
    ite++;
    //fprintf(stdout, "ite=%d, lhsum=%e\n",ite, lhsum);

    if (ratio <= epsilon && ite>=minite) {
      break;
    }

    // exit if empty state appears
    for (ii=0,k=0;ii<nb;ii++) {
      for (i=0;i<numst[ii];i++) {
		if (lsum[ii][i]==0.0) 
			k++;
      }
    }
    if (k) {
      res=1;
      break;
    }

    twomdflag=1;

    /*---------------------------*/
    /** update model parameters **/
    /*---------------------------*/
    for (ii=0;ii<nb;ii++) {
      for (i=0,v1=0.0; i<numst[ii]; i++)
		v1+=lsum[ii][i];
      if (v1>0.0) {
		for (i=0; i<numst[ii]; i++)
			md->mds[ii]->a00[i]=lsum[ii][i]/v1; // prior for the states in this block
      }
      else {
		for (i=0; i<numst[ii]; i++)
			md->mds[ii]->a00[i]=1.0/(double)numst[ii];
      }
      
      for (j=0; j<prenumst[ii]; j++)
		for (k=0; k<numst[ii]; k++)
			md->mds[ii]->a[j][k]=asum[ii][j][k];
      
      for (j=0;j<bdim[ii];j++)
		for (k=0;k<bdim[ii];k++) sigcom[ii][j][k]=0.0;
      
      for (i=0;i<numst[ii];i++) {
		if (DIAGCOV==1) {
			for (j=0;j<bdim[ii];j++)
				sigcom[ii][j][j]+=md->mds[ii]->a00[i]*sigma[ii][i*bdim[ii]+j][j];
		}
		else {
			for (j=0;j<bdim[ii];j++)
				for (k=0;k<bdim[ii];k++) 
					sigcom[ii][j][k]+=md->mds[ii]->a00[i]*sigma[ii][i*bdim[ii]+j][k];
		}
      }
      
      for (i=0; i<numst[ii]; i++) {
		curg=md->mds[ii]->stpdf[i];
		for (j=0; j<bdim[ii]; j++)
			curg->mean[j]=musum[ii][i*bdim[ii]+j];
		for (j=0; j<bdim[ii]; j++)
			for (k=0; k<bdim[ii]; k++)
				curg->sigma[j][k]=sigma[ii][i*bdim[ii]+j][k]*lambda+(1.0-lambda)*sigcom[ii][j][k];
	
		/* compute the inverse sigma and the determinant of sigma */
		mm=mat_logdet_inv_diag_double(curg->sigma, curg->sigma_inv, 
				   &(curg->sigma_det_log),bdim[ii], DIAGCOV);

		if (mm==2) {
			fprintf(stderr, "Covariance matrix: not positive definite\n");
		}

		if (mm==2) { /* singular matrix */
			for (k1=0, tpdb=0.0; k1<bdim[ii]; k1++) tpdb+=curg->sigma[k1][k1];
			tpdb=(tpdb>0.0)?(tpdb/(double)bdim[ii]*epsilon2):epsilon2;
	  
			/* modify sigma by adding a scalar matrix */
			for (k1=0; k1<bdim[ii]; k1++) curg->sigma[k1][k1]+=tpdb;
			mm2=mat_logdet_inv_diag_double(curg->sigma, curg->sigma_inv, 
					 &(curg->sigma_det_log),bdim[ii], DIAGCOV);

			if (mm2==2) {
				fprintf(stderr, "Non-positive definite covariance matrix after adding scalar matrix, abort\n");
				MPI_Finalize();
				exit(1);
			}
		}
      }
    }
  } // while (ite<minite ...)

  MPI_Allgatherv(loglikehd_local, nseq, MPI_DOUBLE, loglikehd, nseq_proc, displs, MPI_DOUBLE, MPI_COMM_WORLD);

  *lhsumpt=lhsum;

  for (i=0;i<nb;i++) {
    free(musum[i]);
    free(musum_local[i]);
    free(mu[i]);
    free_matrix_2d_double(mom2sum+i, numst[i]*bdim[i]);
    free_matrix_2d_double(mom2sum_local+i, numst[i]*bdim[i]);
    free_matrix_2d_double(mom2+i, numst[i]*bdim[i]);
    free_matrix_2d_double(sigma+i, numst[i]*bdim[i]);
    free_matrix_2d_double(sigcom+i,bdim[i]);
    free_matrix_2d_double(asum+i, prenumst[i]);
    free_matrix_2d_double(asum_local+i, prenumst[i]);
    free_matrix_2d_double(a+i, prenumst[i]);
    free(lsum[i]);
    free(lsum_local[i]);
    free(l1img[i]);
  }
  free(musum);free(musum_local);free(mu);free(mom2sum);free(mom2sum_local);free(mom2);free(sigma);free(sigcom);
  free(asum);free(asum_local);free(a);free(lsum);free(lsum_local);free(l1img);
  free(thetalog);
  free(betalog);
  free(prenumst);
  free(loglikehd_local);
  free(nseq_proc);
  free(displs);

  return(res);
}

// Put the variables in the order of the conditional chain
// Input u[][] will be changed.
void ordervar(double **u,int nseq, int nb, int *bdim, int **var)
{
  int i,j,k,m,n,dim;
  double *buf;

  for (i=0,dim=0;i<nb;i++) dim+=bdim[i];
  buf=(double *)calloc(dim,sizeof(double));

  for (i=0;i<nseq;i++) {
    for (j=0,m=0;j<nb;j++) 
      for (k=0;k<bdim[j];k++) {
	buf[m]=u[i][var[j][k]];
	m++;
      }
    for (j=0;j<dim;j++) u[i][j]=buf[j];
  }

  free(buf);
}

//Intended to use for selection of variable blocks, the original input u[][]
//is not overwritten, output is in *u2_pt[][]
void ordervar2(double **u, double ***u2_pt, int nseq, int nb, int *bdim, int **var)
{
  int i,j,k,m,n,dim;
  double **u2;

  for (i=0,dim=0;i<nb;i++) dim+=bdim[i];
  if (dim==0) {
	  
    fprintf(stderr, "Dimension is zero in ordervar2\n");
    MPI_Finalize();
    exit(1);
  }
  
  u2=(double **)calloc(nseq,sizeof(double *));
  for (i=0;i<nseq;i++) u2[i]=(double *)calloc(dim,sizeof(double));
  
  for (i=0;i<nseq;i++) {
    for (j=0,m=0;j<nb;j++) 
      for (k=0;k<bdim[j];k++) {
		u2[i][m]=u[i][var[j][k]];
		m++;
      }
  }

  *u2_pt=u2;
}


void hmmfit(double **u, int nseq, int nb, int *bdim, int **var, int *numst, CondChain *md, 
	    double *loglikehd, double *lhsumpt, double epsilon, double *wt)
{
  int t,i,j,k,l,m,n;
  double *thiswt;
 
  // Order the variables according to the info in bdim and var
  ordervar(u,nseq,nb,bdim,var);
  //fprintf(stdout, "variables ordered\n");

  /*** Set up the model and allocate space its members ***/
  newccm(md, nb,bdim,var,numst);

  /*** initialize parameters using the states given by kmeans **/
  initial_ccm(u,nseq,md);
  //fprintf(stdout, "ccm initialized\n");

  /** start em iterative estimation **/
  if (wt==NULL) {
    thiswt=(double *)calloc(nseq, sizeof(double));
    for (i=0;i<nseq;i++) thiswt[i]=1.0;
  } else {thiswt=wt;}

  baumwelch(u, nseq, md, loglikehd, lhsumpt, epsilon, thiswt);

  if (wt==NULL) free(thiswt);

}

/*------------------------------------------------------------------------*/
/* Revisd from hmmfit() to allow multiple initializations.                */
/* ninit1 is the number of different initializations by scheme 1 and      */
/* ninit2 is the number of different initializations by scheme 2. The     */
/* original initialization by k-means is always kept. If both ninit1      */
/* and ninit2 are zero, then hmmfit_minit() is equivalent to hmmfit().    */ 
/* In the first scheme of initialization, a random subset is chosen from  */
/* the whole dataset and then k-means initialization is applied.          */
/* In the second scheme, random samples from the data set are used as     */
/* the initial Gaussian means.                                            */
/*------------------------------------------------------------------------*/
void hmmfit_minit(double **u, int nseq, int nb, int *bdim, int **var, int *numst, CondChain **md, 
		  double *loglikehd, double *lhsumpt, double epsilon, double *wt, 
		  int ninit0, int ninit1, int ninit2, int randomseed)
// *md has no space allocated, to be allocated inside this subroutine
{
  int t,i,j,k,l,m,n;
  double *thiswt;
  CondChain **mymd;
  double *lhsum, *mylikehd;
  int ninit;
  int nseq_total; // total number of samples
 
  MPI_Allreduce(&nseq, &nseq_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
 
  // Order the variables according to the info in bdim and var
  ordervar(u,nseq,nb,bdim,var);
  //fprintf(stdout, "variables ordered\n");

  ninit=ninit1+ninit2+ninit0;
  if (ninit==0) {ninit=ninit0=1;}

  mymd=(CondChain **)calloc(ninit,sizeof(CondChain *));
  for (i=0;i<ninit;i++)
    mymd[i]=(CondChain *)calloc(1,sizeof(CondChain));

  lhsum=(double *)calloc(ninit,sizeof(double));
  mylikehd=(double *)calloc(ninit*nseq_total,sizeof(double));

  if (wt==NULL) {
    thiswt=(double *)calloc(nseq, sizeof(double));
    for (i=0;i<nseq;i++) thiswt[i]=1.0;
  } else {thiswt=wt;}
  
  for (m=0;m<ninit;m++) {  
    newccm(mymd[m], nb,bdim,var,numst);

    if (m<ninit0) initial_ccm(u,nseq,mymd[m]);
    else {
      if (m<ninit0+ninit1) initial_ccm1(u,nseq,mymd[m], randomseed+(m-ninit0)*100); //use seed m*100
      else initial_ccm2(u,nseq,mymd[m], randomseed+(m-ninit0)*100); //use m*100 as seed
    }

    baumwelch(u, nseq, mymd[m], mylikehd+m*nseq_total, lhsum+m, epsilon, thiswt);

    //fprintf(stdout, "Initial %d, likelihood: %e\n",m, lhsum[m]);
  }

  *lhsumpt=lhsum[0];
  n=0;
  for (m=1;m<ninit;m++) {
    if (lhsum[m]>*lhsumpt) {
      *lhsumpt=lhsum[m];
      n=m;
    }
  }
  
  *md=mymd[n];
  for (i=0;i<nseq;i++) loglikehd[i]=mylikehd[n*nseq+i];

  //fprintf(stdout, "Choose initialiation: %d, likelihood: %e\n", n,*lhsumpt);

  if (wt==NULL) free(thiswt);
  free(lhsum);
  free(mylikehd);
  // Release each mymd[]
  for (m=0;m<ninit;m++) { if (m==n) continue; freeccm(&(mymd[m])); }
  free(mymd);
}

/*------------------------------------------------------------------------*/
/* Revisd from hmmfit_minit() to perform greedy selection of the variable */
/* block structure.                                                       */
/*------------------------------------------------------------------------*/
void hmmfit_minit2(double **u, int nseq, int nb, int *bdim, int **var, int *numst, CondChain **md, 
		   double *loglikehd, double *lhsumpt, double epsilon, double *wt, int ninit0,
		   int ninit1, int ninit2, int randomseed)
// *md has no space allocated, to be allocated inside this subroutine
{
  int t,i,j,k,l,m,n;
  double *thiswt;
  CondChain **mymd;
  double *lhsum, *mylikehd;
  int ninit;
  double **u2;
  int nseq_total; // total number of samples
 
  MPI_Allreduce(&nseq, &nseq_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  // Order the variables according to the info in bdim and var
  ordervar2(u, &u2,nseq,nb,bdim,var);
  //fprintf(stdout, "variables ordered\n");

  ninit=ninit1+ninit2+ninit0;
  if (ninit==0) { ninit=ninit0=1;}

  mymd=(CondChain **)calloc(ninit,sizeof(CondChain *));
  for (i=0;i<ninit;i++)
    mymd[i]=(CondChain *)calloc(1,sizeof(CondChain));

  lhsum=(double *)calloc(ninit,sizeof(double));
  mylikehd=(double *)calloc(ninit*nseq_total,sizeof(double));

  if (wt==NULL) {
    thiswt=(double *)calloc(nseq, sizeof(double));
    for (i=0;i<nseq;i++) thiswt[i]=1.0;
  } else {thiswt=wt;}
  
  for (m=0;m<ninit;m++) {
    newccm(mymd[m], nb,bdim,var,numst);

    if (m<ninit0) initial_ccm(u2,nseq,mymd[m]);
    else {
      if (m<ninit0+ninit1) initial_ccm1(u2,nseq,mymd[m], randomseed+(m-ninit0)*100); //use seed m*100
      else initial_ccm2(u2,nseq,mymd[m], randomseed+(m-ninit0)*100); //use m*100 as seed
    }
    
    baumwelch(u2, nseq, mymd[m], mylikehd+m*nseq_total, lhsum+m, epsilon, thiswt);

    //fprintf(stdout, "Initial %d, likelihood: %e\n",m, lhsum[m]);
  }

  *lhsumpt=lhsum[0];
  n=0;
  for (m=1;m<ninit;m++) {
    if (lhsum[m]>*lhsumpt) {
      *lhsumpt=lhsum[m];
      n=m;
    }
  }
  
  *md=mymd[n];
  for (i=0;i<nseq;i++) loglikehd[i]=mylikehd[n*nseq+i];

  //fprintf(stdout, "Choose initialiation: %d, likelihood: %e\n", n,*lhsumpt);

  if (wt==NULL) free(thiswt);
  free(lhsum);
  free(mylikehd);
  // Release each mymd[]
  for (m=0;m<ninit;m++) { if (m==n) continue; freeccm(&(mymd[m])); }
  free(mymd);

  for (i=0;i<nseq;i++) free(u2[i]);
  free(u2);
}


void permutevar(int dim, int np, int **vlist)
{
  int i,j,k,l,m,n,ii;
  double *buf,*buf2;
  int rank; // rank of MPI process
  //int v[10]={0,1};//v[10]={2,5,7,4,3,0,1,6,0,0};
  //int v[10]={2,5,7,4,3,0,1,6,0,0};
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  buf=(double *)calloc(dim,sizeof(double));
  buf2=(double *)calloc(dim,sizeof(double));

  srand48(0);
  
  // permute variables on master node and send permutation to slaves
  if (rank == 0){
	  for (ii=0;ii<np;ii++) {
		for (i=0;i<dim;i++) buf[i]=drand48();
		SortDouble(buf,buf2,vlist[ii],dim);
		
		for (j=0;j<ii;j++) {
		  for (i=0,m=0;i<dim;i++) {
			if (vlist[ii][i]==vlist[j][i]) m++;
		  }
		  
		  if (m==dim) {
			fprintf(stderr, "Warning: duplicate permutation of variables\n");
		  }
		}

		for (i=0;i<dim;i++) fprintf(stderr, "%d ",vlist[ii][i]);
		fprintf(stderr, "\n");
	  }
  }
  
  // send permutations to slaves
  for (ii=0;ii<np;ii++)
	MPI_Bcast(vlist[ii], dim, MPI_INT, 0, MPI_COMM_WORLD);
	
  /** debug
  for (i=0;i<dim;i++) vlist[0][i]=v[i];
  for (i=0;i<dim;i++) fprintf(stderr, "%d ",vlist[0][i]);
  fprintf(stderr, "\n"); 
  ***/
  
  free(buf);
  free(buf2);
}

void setnumstate(int nb, int *bdim, int *numst, int *Numst0)
{
  int i;

  if (Numst0==NULL) {
    for (i=0;i<nb;i++) {
      if (bdim[i]<=2) { numst[i]=5;}
      else{
        if(bdim[i]<=5) {numst[i]=12;}
        else{
          if (bdim[i]<=10) {numst[i]=20;}
          else{numst[i]=10+bdim[i];}
        }
      }
    }
  }
  else {
    for (i=0;i<nb;i++) numst[i]=Numst0[bdim[i]-1];
  }
}

//promote nested model
void setnumstate2(int nb, int *bdim, int *numst, int *Numst0)
{
  int i;
  
  if (Numst0==NULL) {
    for (i=0;i<nb;i++) {
      if (bdim[i]==1) numst[i]=3;
      if (bdim[i]==2) numst[i]=9;
      if (bdim[i]>=3) numst[i]=16;
    }
  }
  else {
    for (i=0;i<nb;i++) numst[i]=Numst0[bdim[i]-1];
  }
}

void printvbinfo(int nb, int *bdim, int **var, double lhsum)
{
  int i,j;

  fprintf(stdout, "nb=%d, ",nb);
  for (i=0;i<nb;i++) fprintf(stdout, "%d ", bdim[i]);
  fprintf(stdout, ", ");
  for (i=0;i<nb;i++) {
    for (j=0;j<bdim[i];j++){
      fprintf(stdout, "%d ", var[i][j]);
    }
    fprintf(stdout, " * ");
  }
  fprintf(stdout, ", --- ");
  fprintf(stdout, "likelihood/BIC=%f\n",lhsum);
}

int computenp(int nb, int *bdim, int *numst)
{
  int i,m,res;

  if (nb==0) return(0);

  if (DIAGCOV==1) {//Diagonal covariance matrix induce smaller number of parameters
    res=numst[0]-1+numst[0]*bdim[0]+numst[0]*bdim[0];
    for (i=1;i<nb;i++) {
      m=numst[i-1]*(numst[i]-1)+numst[i]*bdim[i]+numst[i]*bdim[i];
      res+=m;
    }
  }
  else {
    res=numst[0]-1+numst[0]*bdim[0]+numst[0]*bdim[0]*(bdim[0]+1)/2;
    for (i=1;i<nb;i++) {
      m=numst[i-1]*(numst[i]-1)+numst[i]*bdim[i]+numst[i]*bdim[i]*(bdim[i]+1)/2;
      res+=m;
    }
  }
  
  return(res);
}

void findbuddy(int *buddy, int *skipped, int *bdim0, int nb0, int **var0,
	       double *lhsum, int **vlist, int k1, int k2, int mm)
{
  int kk,i,j;
  double v3;
  
  kk=nb0;
  for (i=0;i<nb0;i++) {
    if (skipped[i]==0) {
      v3=lhsum[i];
      kk=i;
    }
  }
  if (kk<nb0){//at least one possible buddy exists
    for (i=0;i<nb0;i++) {
      if (skipped[i]==0 && lhsum[i]>v3) {
		v3=lhsum[i];
		kk=i;
      }
    }
  }
  
  if (mm<nb0) {
    buddy[vlist[k1][k2]]=mm;//buddy is itself since the block is not singular
  }
  else {
    buddy[vlist[k1][k2]]=kk;//buddy is forced to be itself if all precedents disqualifies
  }
}

void mergeblock(int *bdim0, int **var0, int *nb0pt, int blk, int mybuddy)
{
  int j,jj,nb0;

  nb0=*nb0pt;
  
  var0[mybuddy][bdim0[mybuddy]]=var0[blk][0];
  bdim0[mybuddy]++;
  for (j=blk; j<nb0-1;j++) {//shift later blocks one block ahead
    bdim0[j]=bdim0[j+1];
    for (jj=0;jj<bdim0[j];jj++) {
      var0[j][jj]=var0[j+1][jj];
    }
  }
  nb0--;

  *nb0pt=nb0;
}


int removeminimumblock(int mindim, int maxdim,int *bdim0, int *buddy, int **var0,
			int *nb0pt, int relaxsearch)
{
  int i,j,k,kk,jj,k4,nb0,mybuddy;
  int res=0;

  nb0=*nb0pt;
 
  if (mindim==2 && maxdim>=2) {
    for (i=0,kk=bdim0[0];i<nb0;i++) {
      if (bdim0[i]<kk) kk=bdim0[i]; //find minimum block dimension
    }
    if (kk==1) {//minimum block is too small
      for (i=nb0-1;i>=1;i--) { //process singular block, first block is skipped since
		//it has no preceding buddy
		//recursive loop since nb0, bdim0, var0 can be changed in the loop
		//The visit of the blocks has to be backwards and mindim has to be 2, not working for
		//more than 2
		if (bdim0[i]==1) {
		  if (buddy[var0[i][0]]<i) {
			if (bdim0[buddy[var0[i][0]]]<maxdim) { //join buddy
			  //update bdim0, nb0, var0
			  mergeblock(bdim0, var0, &nb0, i, buddy[var0[i][0]]);
			  res=1;
			}
			else {//join minimum preceding block
			  if (relaxsearch==1) {
				for (j=0,k4=bdim0[0],mybuddy=0;j<i;j++) {
					if (bdim0[j]<k4) {
						k4=bdim0[j]; //find minimum block dimension
						mybuddy=j;
					}
				}
				//update bdim0, nb0, var0
				if (bdim0[mybuddy]<maxdim) {
					mergeblock(bdim0, var0, &nb0, i, mybuddy);
					res=1;
				}
			  }
			}
		  }
		}
      }
    }
  }

  *nb0pt=nb0;
  return(res);
}


//Wrapper for hmmfit_minit2 with variable block selection by a greedy procedure
void hmmfit_vb(double **u, int nseq, int dim, int *snb, int **sbdim, int ***svar,
	       int nperm, int nperm0, int **vlist0,
	       CondChain **md, double *loglikehd, double *lhsumpt, double epsilon, double *wt,
	       int ninit0, int ninit1, int ninit2, int randomseed, int *Numst0,
	       int maxdim, int mindim, int relaxsearch)
// *md has no space allocated, to be allocated inside this subroutine
//nperm0 is the number of pre-given permutations, and the permutations are stored in vlist0[nperm0][dim]
{
  int i,j,k,m,n,kk,k1,k2,k3,k4;
  int nb, nb0, *nb1, *bdim, *bdim0, **bdim1, **var, **var0, **var1, *numst;
  CondChain *thismd=NULL;
  double *lhsum, *lhsum1, v0,v1,v2,v3;
  int **vlist;
  int *skipped,*buddy,mybuddy;
  int rank; // rank of MPI process
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  lhsum=(double *)calloc(dim,sizeof(double));
  vlist=(int **)calloc(nperm,sizeof(int *));
  for (i=0;i<nperm;i++) vlist[i]=(int *)calloc(dim,sizeof(int));
  for (i=0;i<(nperm0<nperm?nperm0:nperm);i++)
    for (j=0;j<dim;j++)
      vlist[i][j]=vlist0[i][j];
  //pre-given permutations
  
  if (nperm>nperm0) permutevar(dim,nperm-nperm0,vlist+nperm0);//randomly generated permutations

  //Allocate space to get ready for recursive fitting
  numst=(int *)calloc(dim,sizeof(int));
  bdim=(int *)calloc(dim,sizeof(int));
  bdim0=(int *)calloc(dim,sizeof(int));
  var=(int **)calloc(dim,sizeof(int *));
  var0=(int **)calloc(dim,sizeof(int *));
  for (i=0;i<dim;i++) {
    var[i]=(int *)calloc(dim,sizeof(int));
    var0[i]=(int *)calloc(dim,sizeof(int));
  }

  lhsum1=(double *)calloc(nperm,sizeof(double));
  nb1=(int *)calloc(nperm,sizeof(int));
  bdim1=(int **)calloc(nperm,sizeof(int *));
  for (i=0;i<nperm;i++) bdim1[i]=(int *)calloc(dim,sizeof(int));
  var1=(int **)calloc(nperm,sizeof(int *));
  for (i=0;i<nperm;i++) var1[i]=(int *)calloc(dim,sizeof(int));
  
  skipped=(int *)calloc(dim+1,sizeof(int));
  buddy=(int *)calloc(dim,sizeof(int)); // best block for variable if we don't want to add a new block
  //we only really need lhsum for selection purpose
  for (k1=0;k1<nperm;k1++) {
    nb0=1;
    bdim0[0]=1;
    var0[0][0]=vlist[k1][0]; //// !!!!!!!!!!!!!!!!!!! //// 

    buddy[vlist[k1][0]]=0;//buddy is indexed by original variable id
    for (k2=1;k2<dim;k2++) {
		for (k3=0; k3<nb0;k3++) {
			nb=nb0;
			for (i=0;i<nb0;i++) {
				bdim[i]=bdim0[i];
				for (j=0;j<bdim0[i];j++)
					var[i][j]=var0[i][j]; //// !!!!!!!!!!!!!!!!!! //// 
			}
	
			if ((relaxsearch==1 && bdim[k3]==maxdim) || 
				(relaxsearch==0 && (k3<nb0-1 || bdim[k3]==maxdim))) {	//new line of code for the new variable block structure search, more restricted than original
				skipped[k3]=1;
				continue; //upper bound block dimension
			} else {skipped[k3]=0;}
	
			bdim[k3]++;
			var[k3][bdim[k3]-1]=vlist[k1][k2];
	
			setnumstate(nb,bdim,numst,Numst0);
			hmmfit_minit2(u, nseq, nb, bdim, var, numst, &thismd, loglikehd, lhsum+k3,
					(double)epsilon, wt, ninit0, ninit1, ninit2, randomseed);
			freeccm(&thismd);
			lhsum[k3]-=(double)(computenp(nb, bdim,numst))*log((double)nseq)*0.5; //BIC
	
			//printvbinfo(nb, bdim, var, lhsum[k3]); //debugging
		}
		nb=nb0+1; //add a new block
		for (i=0;i<nb0;i++) {
			bdim[i]=bdim0[i];
			for (j=0;j<bdim0[i];j++)
				var[i][j]=var0[i][j];
		}
		bdim[nb-1]=1;
		var[nb-1][0]=vlist[k1][k2];
		setnumstate(nb,bdim,numst,Numst0);
		hmmfit_minit2(u, nseq, nb, bdim, var, numst, &thismd, loglikehd, lhsum+nb0,
				(double)epsilon, wt, ninit0, ninit1, ninit2, randomseed);
		freeccm(&thismd);
		lhsum[nb0]-=(double)(computenp(nb, bdim,numst))*log((double)nseq)*0.5; //BIC
		//printvbinfo(nb, bdim, var, lhsum[nb0]); //debugging
		skipped[nb0]=0;
 
		// Generate the current best variable block structure up to dimension k2
		v0=lhsum[nb0];//additional block by the new variable is always an option
		m=nb0;
		for (i=0;i<nb0;i++) {
			if (skipped[i]==0 && lhsum[i]>v0) {
				v0=lhsum[i];
				m=i;
			}
		}

		//Define buddy
		findbuddy(buddy, skipped, bdim0, nb0, var0, lhsum, vlist, k1,k2,m);

		if (m<nb0) {//variable appended to existing block
			bdim0[m]++;
			var0[m][bdim0[m]-1]=vlist[k1][k2];
		}
		else {//a new block is formed
			bdim0[nb0]=1;
			var0[nb0][0]=vlist[k1][k2];
			nb0++;
		}//k3      
    } //k2
    

    if (removeminimumblock(mindim,maxdim,bdim0,buddy,var0,&nb0,relaxsearch)==1 && nperm>1)
    {//Fit again using new block structure
		setnumstate(nb0,bdim0,numst,Numst0);
		hmmfit_minit2(u, nseq, nb0, bdim0, var0, numst, &thismd, loglikehd, lhsum1+k1,
		      (double)epsilon, wt, ninit0, ninit1, ninit2, randomseed);
		freeccm(&thismd);
		lhsum1[k1]-=(double)(computenp(nb0, bdim0,numst))*log((double)nseq)*0.5; //BIC
	//printvbinfo(nb0, bdim0, var0, lhsum1[k1]); //debugging
    }
    else {
      lhsum1[k1]=v0; //no change occurred, record the best BIC
    }

    //Record the best for this permutation
    nb1[k1]=nb0;
    for (i=0;i<nb0;i++) {
      bdim1[k1][i]=bdim0[i];
    }
    for (i=0,m=0;i<nb0;i++) {
      for (j=0;j<bdim0[i];j++){
	var1[k1][m]=var0[i][j];
	m++;
      }
    }    
  } //k1
  
  //Output snb, sbdim, svar
  //Choose the best permutation based on each permutation's best model
  v1=lhsum1[0];
  m=0;
  for (i=1;i<nperm;i++) {
    if (lhsum1[i]>v1) {
      v1=lhsum1[i];
      m=i;
    }
  }

  *snb=nb1[m];
  *sbdim=(int *)calloc(*snb,sizeof(int));
  for (i=0;i<*snb;i++) (*sbdim)[i]=bdim1[m][i];
  *svar=(int **)calloc(*snb,sizeof(int *));
  for (i=0;i<*snb;i++) {
    (*svar)[i]=(int *)calloc((*sbdim)[i],sizeof(int));
  }

  for (i=0,n=0;i<nb1[m];i++) {
    for (j=0;j<bdim1[m][i];j++){
      (*svar)[i][j]=var1[m][n];
      n++;
    }
  }
  
  if (rank == 0){
	fprintf(stdout, "Final chosen model:\n");
	printvbinfo(*snb, *sbdim, *svar, v1); //debugging
  }
  
  // Do a final fitting with snb, sbdim, svar
  setnumstate(*snb,*sbdim,numst,Numst0);
  hmmfit_minit2(u, nseq, *snb, *sbdim, *svar, numst, md, loglikehd, lhsumpt,
		(double)epsilon, wt, ninit0, ninit1, ninit2, randomseed);
  (*lhsumpt)-=(double)(computenp(*snb, *sbdim,numst))*log((double)nseq)*0.5; //BIC
  
  for (i=0;i<nperm;i++) free(vlist[i]);
  free(vlist);
  free(lhsum);

  for (i=0;i<dim;i++) {
    free(var[i]);
    free(var0[i]);
  }
  free(var); free(var0); free(bdim); free(bdim0);
  free(numst);

  free(lhsum1);
  free(nb1);
  for (i=0;i<nperm;i++) {
    free(bdim1[i]);
    free(var1[i]);
  }
  free(bdim1); free(var1);
  
  free(skipped);
  free(buddy);
}
