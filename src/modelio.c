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

unsigned char write_hmm(HmmModel *md, FILE *outfile)
{
  int i,j,k,m,n;
  int dim,numst,prenumst;

  dim=md->dim;
  numst=md->numst;
  prenumst=md->prenumst;
  
  if (fwrite(&dim, sizeof(int), 1, outfile)!=1)
    {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }

  if (fwrite(&numst, sizeof(int), 1, outfile)!=1)
    {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }

  if (fwrite(&prenumst, sizeof(int), 1, outfile)!=1)
    {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }


  if (fwrite(md->a00, sizeof(double),numst, outfile)!=numst)
    {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }

  for (i=0; i<prenumst; i++) {
    if (fwrite(md->a[i], sizeof(double),numst, outfile)!=numst)
      {
	fprintf(stderr, "**ERROR writing out data\n");
	return(0);
      }    
  }

  for (i=0; i<numst; i++) {
    if (fwrite(&(md->stpdf[i]->exist), sizeof(int),1,outfile)!=1) {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }

    if (fwrite(&(md->stpdf[i]->dim), sizeof(int), 1, 
	       outfile)!=1) {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    } 
    
    if (fwrite(md->stpdf[i]->mean, sizeof(double), md->stpdf[i]->dim, 
	       outfile)!=md->stpdf[i]->dim) {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }
 
    if (fwrite(&(md->stpdf[i]->sigma_det_log), sizeof(double),1,
	       outfile)!=1) {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }
 
    for (m=0; m<md->stpdf[i]->dim; m++) {
      if (fwrite(md->stpdf[i]->sigma[m], sizeof(double), md->stpdf[i]->dim,
		 outfile)!=md->stpdf[i]->dim) {
	fprintf(stderr, "**ERROR writing out data\n");
	return(0);
      } 
    }

    for (m=0; m<md->stpdf[i]->dim; m++) {
      if (fwrite(md->stpdf[i]->sigma_inv[m], sizeof(double), 
		 md->stpdf[i]->dim, outfile)!=md->stpdf[i]->dim) {
	fprintf(stderr, "**ERROR writing out data\n");
	return(0);
      } 
    }
  }
  
  return(1);
}

/*------------------------------------------------------------*/

unsigned char read_hmm(HmmModel *md, FILE *infile)
{
  int i,j,k,m,n;
  int dim,numst,prenumst;

  if (fread(&dim, sizeof(int), 1, infile)!=1)
    {
      fprintf(stderr, "**ERROR reading in model\n");
      return(0);
    }
  
  if (fread(&numst, sizeof(int), 1, infile)!=1)
    {
      fprintf(stderr, "**ERROR reading in model\n");
      return(0);
    }

  if (fread(&prenumst, sizeof(int), 1, infile)!=1)
    {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }

  md->dim=dim;
  md->numst=numst;
  md->prenumst=prenumst;
  
  md->a00=(double *)calloc(numst,sizeof(double));
  if (md->a00==NULL) {
    fprintf(stderr, "Error allocate space while reading in model\n");
    exit(1);
  }

  if (fread(md->a00, sizeof(double),numst, infile)!=numst)
    {
      fprintf(stderr, "**ERROR reading in model\n");
      return(0);
    }

  md->a=(double **)calloc(prenumst,sizeof(double *));
  for (i=0; i<prenumst; i++) {
    md->a[i]=(double *)calloc(numst,sizeof(double));
    if (fread(md->a[i], sizeof(double),numst, infile)!=numst)
      {
	fprintf(stderr, "**ERROR reading in model\n");
	return(0);
      }    
  }

  md->stpdf=(GaussModel **)calloc(numst, sizeof(GaussModel *));
  for (i=0; i<numst; i++)
    md->stpdf[i]=(GaussModel *)calloc(1, sizeof(GaussModel));

  for (i=0; i<numst; i++) {
    if (fread(&(md->stpdf[i]->exist), sizeof(int),1,infile)!=1) {
      fprintf(stderr, "**ERROR reading in model\n");
      return(0);
    }

    if (fread(&(md->stpdf[i]->dim), sizeof(int), 1,infile)!=1) {
      fprintf(stderr, "**ERROR reading in model\n");
      return(0);
    } 
    
    md->stpdf[i]->mean=(double *)calloc(md->stpdf[i]->dim,sizeof(double));
    if (fread(md->stpdf[i]->mean, sizeof(double), md->stpdf[i]->dim, 
	       infile)!=md->stpdf[i]->dim) {
      fprintf(stderr, "**ERROR reading in model\n");
      return(0);
    }
 
    if (fread(&(md->stpdf[i]->sigma_det_log), sizeof(double),1,infile)!=1) {
      fprintf(stderr, "**ERROR reading in model\n");
      return(0);
    }
 
    md->stpdf[i]->sigma=(double **)calloc(md->stpdf[i]->dim,sizeof(double *));
    for (m=0; m<md->stpdf[i]->dim; m++) {
      md->stpdf[i]->sigma[m]=(double *)calloc(md->stpdf[i]->dim, 
					      sizeof(double));
      if (fread(md->stpdf[i]->sigma[m], sizeof(double), md->stpdf[i]->dim,
		infile)!=md->stpdf[i]->dim) {
	fprintf(stderr, "**ERROR reading in model\n");
	return(0);
      } 
    }

    md->stpdf[i]->sigma_inv=(double **)calloc(md->stpdf[i]->dim,
					      sizeof(double *));
    for (m=0; m<md->stpdf[i]->dim; m++) {
      md->stpdf[i]->sigma_inv[m]=(double *)calloc(md->stpdf[i]->dim, 
						  sizeof(double));
      if (fread(md->stpdf[i]->sigma_inv[m], sizeof(double), md->stpdf[i]->dim,
		infile)!=md->stpdf[i]->dim) {
	fprintf(stderr, "**ERROR reading in model\n");
	return(0);
      } 
    }
  }
  
  return(1);
}


/*------------------------------------------------------------*/
unsigned char print_hmm(HmmModel *md, FILE *outfile)
{
  int i,j,k,m,n;
  int dim,numst,prenumst;

  dim=md->dim;
  numst=md->numst;
  prenumst=md->prenumst;
  
  fprintf(outfile, "dim=%d\n", dim);
  fprintf(outfile, "numst=%d\n", numst);
  fprintf(outfile, "prenumst=%d\n", prenumst);

  fprintf(outfile, "\nTransition probability a00:\n");
  for (i=0; i<numst; i++)
    fprintf(outfile, "%8e ", md->a00[i]);
  fprintf(outfile, "\n");

  fprintf(outfile, "Transition probability a:\n");
  for (i=0; i<prenumst; i++) {
    for (j=0; j<numst; j++)
      fprintf(outfile, "%8e ", md->a[i][j]);
    fprintf(outfile, "\n");
  }

  fprintf(outfile, "\nThe Gaussian distributions of states:\n");
  for (i=0; i<numst; i++) {
    fprintf(outfile, "\nState %d =============================\n", i);
    fprintf(outfile, "exist=%d, dim=%d\n", md->stpdf[i]->exist, 
	    md->stpdf[i]->dim);

    fprintf(outfile, "Mean vector:\n");
    for (j=0; j<dim; j++)
      fprintf(outfile, "%.5e ", md->stpdf[i]->mean[j]);
    fprintf(outfile, "\n");

    fprintf(outfile, "Sigma_det_log=%e\n",md->stpdf[i]->sigma_det_log);

    fprintf(outfile, "Covariance matrix Sigma:\n");
 
    for (m=0; m<md->stpdf[i]->dim; m++) {
      for (n=0; n<md->stpdf[i]->dim; n++)
	fprintf(outfile, "%.5e ", md->stpdf[i]->sigma[m][n]);
      fprintf(outfile, "\n");
    }

    fprintf(outfile, "Covariance matrix inverse Sigma_inv:\n");
 
    for (m=0; m<md->stpdf[i]->dim; m++) {
      for (n=0; n<md->stpdf[i]->dim; n++)
	fprintf(outfile, "%.5e ", md->stpdf[i]->sigma_inv[m][n]);
      fprintf(outfile, "\n");
    }
  }
  
  return(1);
}

/*------------------------------------------------------------*/
unsigned char readascii_hmm(HmmModel *md, FILE *infile)
{
  int i,j,k,m,n,ii;
  int dim,numst,prenumst;
  float v1,v2,v3;
  
  fscanf(infile, "dim=%d\n", &(md->dim));
  fscanf(infile, "numst=%d\n", &(md->numst));
  fscanf(infile, "prenumst=%d\n", &(md->prenumst));

  dim=md->dim;
  numst=md->numst;
  prenumst=md->prenumst;

  md->a00=(double *)calloc(numst,sizeof(double));
  if (md->a00==NULL) {
    fprintf(stderr, "Error allocate space while reading in model\n");
    exit(1);
  }
  fscanf(infile, "\nTransition probability a00:\n");
  for (i=0; i<numst; i++){
    fscanf(infile, "%f ", &v1);
    md->a00[i]=v1;
  }
  fscanf(infile, "\n");

  md->a=(double **)calloc(prenumst,sizeof(double *));
  for (i=0; i<prenumst; i++) {
    md->a[i]=(double *)calloc(numst,sizeof(double));
  }
  fscanf(infile, "Transition probability a:\n");
  for (i=0; i<prenumst; i++) {
    for (j=0; j<numst; j++){
      fscanf(infile, "%f ", &v1);
      md->a[i][j]=v1;
    }
    fscanf(infile, "\n");
  }

  md->stpdf=(GaussModel **)calloc(numst, sizeof(GaussModel *));
  for (i=0; i<numst; i++)
    md->stpdf[i]=(GaussModel *)calloc(1, sizeof(GaussModel));

  fscanf(infile, "\nThe Gaussian distributions of states:\n");
  for (i=0; i<numst; i++) {
    fscanf(infile, "\nState %d =============================\n", &ii);
    fscanf(infile, "exist=%d, dim=%d\n", &(md->stpdf[i]->exist), 
	   &(md->stpdf[i]->dim));

    md->stpdf[i]->mean=(double *)calloc(md->stpdf[i]->dim,sizeof(double));
    fscanf(infile, "Mean vector:\n");
    for (j=0; j<dim; j++) {
      fscanf(infile, "%f ", &v1);
      md->stpdf[i]->mean[j]=v1;
    }
    fscanf(infile, "\n");

    fscanf(infile, "Sigma_det_log=%f\n",&v1);
    md->stpdf[i]->sigma_det_log=v1;
    
    fscanf(infile, "Covariance matrix Sigma:\n");

    md->stpdf[i]->sigma=(double **)calloc(md->stpdf[i]->dim,sizeof(double *));
    for (m=0; m<md->stpdf[i]->dim; m++) {
      md->stpdf[i]->sigma[m]=(double *)calloc(md->stpdf[i]->dim, sizeof(double)); 
    }
    for (m=0; m<md->stpdf[i]->dim; m++) {
      for (n=0; n<md->stpdf[i]->dim; n++) {
	fscanf(infile, "%f ", &v1);
	md->stpdf[i]->sigma[m][n]=v1;
      }
      fscanf(infile, "\n");
    }

    fscanf(infile, "Covariance matrix inverse Sigma_inv:\n");

    md->stpdf[i]->sigma_inv=(double **)calloc(md->stpdf[i]->dim, sizeof(double *));
    for (m=0; m<md->stpdf[i]->dim; m++) {
      md->stpdf[i]->sigma_inv[m]=(double *)calloc(md->stpdf[i]->dim, sizeof(double));
    }

    for (m=0; m<md->stpdf[i]->dim; m++) {
      for (n=0; n<md->stpdf[i]->dim; n++){
	fscanf(infile, "%f ", &v1);
	md->stpdf[i]->sigma_inv[m][n]=v1;
      }
      fscanf(infile, "\n");
    }
  }
  
  return(1);
}

/*------------------------------------------------------------*/
unsigned char readascii2_hmm(HmmModel *md, FILE *infile)
{
  int i,j,k,m,n,ii;
  int dim,numst,prenumst;
  float v1,v2,v3;
  
  fscanf(infile, "%d\n", &(md->dim));
  fscanf(infile, "%d\n", &(md->numst));
  fscanf(infile, "%d\n", &(md->prenumst));

  dim=md->dim;
  numst=md->numst;
  prenumst=md->prenumst;

  md->a00=(double *)calloc(numst,sizeof(double));
  if (md->a00==NULL) {
    fprintf(stderr, "Error allocate space while reading in model\n");
    exit(1);
  }
  for (i=0; i<numst; i++){
    fscanf(infile, "%f ", &v1);
    md->a00[i]=v1;
  }
  fscanf(infile, "\n");

  md->a=(double **)calloc(prenumst,sizeof(double *));
  for (i=0; i<prenumst; i++) {
    md->a[i]=(double *)calloc(numst,sizeof(double));
  }
  for (i=0; i<prenumst; i++) {
    for (j=0; j<numst; j++){
      fscanf(infile, "%f ", &v1);
      md->a[i][j]=v1;
    }
    fscanf(infile, "\n");
  }

  md->stpdf=(GaussModel **)calloc(numst, sizeof(GaussModel *));
  for (i=0; i<numst; i++)
    md->stpdf[i]=(GaussModel *)calloc(1, sizeof(GaussModel));

  for (i=0; i<numst; i++) {
    fscanf(infile, "%d %d\n", &(md->stpdf[i]->exist), 
	   &(md->stpdf[i]->dim));

    md->stpdf[i]->mean=(double *)calloc(md->stpdf[i]->dim,sizeof(double));
    for (j=0; j<dim; j++) {
      fscanf(infile, "%f ", &v1);
      md->stpdf[i]->mean[j]=v1;
    }
    fscanf(infile, "\n");

    fscanf(infile, "%f\n",&v1);
    md->stpdf[i]->sigma_det_log=v1;
    
    md->stpdf[i]->sigma=(double **)calloc(md->stpdf[i]->dim,sizeof(double *));
    for (m=0; m<md->stpdf[i]->dim; m++) {
      md->stpdf[i]->sigma[m]=(double *)calloc(md->stpdf[i]->dim, sizeof(double)); 
    }
    for (m=0; m<md->stpdf[i]->dim; m++) {
      for (n=0; n<md->stpdf[i]->dim; n++) {
	fscanf(infile, "%f ", &v1);
	md->stpdf[i]->sigma[m][n]=v1;
      }
      fscanf(infile, "\n");
    }

    md->stpdf[i]->sigma_inv=(double **)calloc(md->stpdf[i]->dim, sizeof(double *));
    for (m=0; m<md->stpdf[i]->dim; m++) {
      md->stpdf[i]->sigma_inv[m]=(double *)calloc(md->stpdf[i]->dim, sizeof(double));
    }

    for (m=0; m<md->stpdf[i]->dim; m++) {
      for (n=0; n<md->stpdf[i]->dim; n++){
	fscanf(infile, "%f ", &v1);
	md->stpdf[i]->sigma_inv[m][n]=v1;
      }
      fscanf(infile, "\n");
    }
  }
  
  return(1);
}

//-----------------------------------------------------
//-----------------------------------------------------
unsigned char write_ccm(CondChain *md, FILE *outfile)
{
  int i,j,k;

  if (fwrite(&(md->dim), sizeof(int), 1, outfile)!=1)
    {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }

  if (fwrite(&(md->nb), sizeof(int), 1, outfile)!=1)
    {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }

  if (fwrite(&(md->maxnumst), sizeof(int), 1, outfile)!=1)
    {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }

  if (fwrite(md->bdim, sizeof(int),md->nb, outfile)!=md->nb)
    {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }
  if (fwrite(md->numst, sizeof(int),md->nb, outfile)!=md->nb)
    {
      fprintf(stderr, "**ERROR writing out data\n");
      return(0);
    }
  for (i=0;i<md->nb;i++) {
    if (fwrite(md->var[i], sizeof(int),md->bdim[i], outfile)!=md->bdim[i])
      {
	fprintf(stderr, "**ERROR writing out data\n");
	return(0);
      }
  }
  for (i=0;i<md->nb;i++) {
    k=write_hmm(md->mds[i],outfile);
    if (k==0) return(0);
  }
  
  return(1);

}

unsigned char read_ccm(CondChain *md, FILE *infile)
{
  int i,j,k;

  if (fread(&(md->dim), sizeof(int), 1, infile)!=1)
    {
      fprintf(stderr, "**ERROR reading in data\n");
      return(0);
    }

  if (fread(&(md->nb), sizeof(int), 1, infile)!=1)
    {
      fprintf(stderr, "**ERROR reading in data\n");
      return(0);
    }

  if (fread(&(md->maxnumst), sizeof(int), 1, infile)!=1)
    {
      fprintf(stderr, "**ERROR reading in data\n");
      return(0);
    }

  md->bdim=(int *)calloc(md->nb,sizeof(int));
  if (fread(md->bdim, sizeof(int),md->nb, infile)!=md->nb)
    {
      fprintf(stderr, "**ERROR reading in data\n");
      return(0);
    }

  md->numst=(int *)calloc(md->nb,sizeof(int));
  if (fread(md->numst, sizeof(int),md->nb, infile)!=md->nb)
    {
      fprintf(stderr, "**ERROR reading in data\n");
      return(0);
    }
  md->var=(int **)calloc(md->nb,sizeof(int *));
  for (i=0;i<md->nb;i++) {
    md->var[i]=(int *)calloc(md->bdim[i],sizeof(int));
    if (fread(md->var[i], sizeof(int),md->bdim[i], infile)!=md->bdim[i])
      {
	fprintf(stderr, "**ERROR reading in data\n");
	return(0);
      }
  }

  // Set up the redundant fields for convenience of computation
  md->cbdim=(int *)calloc(md->nb,sizeof(int));
  md->cbdim[0]=0;
  md->cnumst=(int *)calloc(md->nb,sizeof(int));
  md->cnumst[0]=0;
  for (i=0;i<md->nb-1;i++) {
    md->cbdim[i+1]=md->cbdim[i]+md->bdim[i];
    md->cnumst[i+1]=md->cnumst[i]+md->numst[i];
  }

  md->mds=(HmmModel **)calloc(md->nb,sizeof(HmmModel *));
  for (i=0;i<md->nb;i++) {
    md->mds[i]=(HmmModel *)calloc(1,sizeof(HmmModel));
    k=read_hmm(md->mds[i],infile);
    if (k==0) return(0);
  }
  
  return(1);

}


unsigned char print_ccm(CondChain *md, FILE *outfile)
{
  int i,j,k;
  
  fprintf(outfile, "dim=%d\n", md->dim);
  fprintf(outfile, "nb=%d\n", md->nb);
  fprintf(outfile, "maxnumst=%d\n", md->maxnumst);

  fprintf(outfile, "Dimension of each block: \n");
  for (i=0;i<md->nb;i++) fprintf(outfile, "%d ", md->bdim[i]);
  fprintf(outfile, "\n");

  fprintf(outfile, "Variables in each block: \n");
  for (i=0;i<md->nb;i++) {
    for (j=0;j<md->bdim[i];j++)
      fprintf(outfile, "%d ", md->var[i][j]);
    fprintf(outfile, "\n");
  }
  
  fprintf(outfile, "Number of states of each block: \n");
  for (i=0;i<md->nb;i++) fprintf(outfile, "%d ", md->numst[i]);
  fprintf(outfile, "\n");

  for (i=0;i<md->nb;i++) {
    fprintf(outfile, "\n!!!!!!!==== HMM for variable block %d =====!!!!!!\n",i);
    k=print_hmm(md->mds[i],outfile);
    if (k==0) return(0);
  }
  
  return(1);

}

unsigned char readascii_ccm(CondChain *md, FILE *infile)
{
  int i,j,k;
  int ii,jj;
  
  fscanf(infile, "dim=%d\n", &(md->dim));
  fscanf(infile, "nb=%d\n", &(md->nb));
  fscanf(infile, "maxnumst=%d\n", &(md->maxnumst));

  md->bdim=(int *)calloc(md->nb,sizeof(int));
  fscanf(infile, "Dimension of each block: \n");
  for (i=0;i<md->nb;i++) fscanf(infile, "%d ", md->bdim+i);
  fscanf(infile, "\n");

  md->var=(int **)calloc(md->nb,sizeof(int *));
  for (i=0;i<md->nb;i++) {
    md->var[i]=(int *)calloc(md->bdim[i],sizeof(int));
  }
  fscanf(infile, "Variables in each block: \n");
  for (i=0;i<md->nb;i++) {
    for (j=0;j<md->bdim[i];j++)
      fscanf(infile, "%d ", md->var[i]+j);
    fscanf(infile, "\n");
  }
  
  md->numst=(int *)calloc(md->nb,sizeof(int));
  fscanf(infile, "Number of states of each block: \n");
  for (i=0;i<md->nb;i++) fscanf(infile, "%d ", md->numst+i);
  fscanf(infile, "\n");

  md->cbdim=(int *)calloc(md->nb,sizeof(int));
  md->cbdim[0]=0;
  md->cnumst=(int *)calloc(md->nb,sizeof(int));
  md->cnumst[0]=0;
  for (i=0;i<md->nb-1;i++) {
    md->cbdim[i+1]=md->cbdim[i]+md->bdim[i];
    md->cnumst[i+1]=md->cnumst[i]+md->numst[i];
  }

  md->mds=(HmmModel **)calloc(md->nb,sizeof(HmmModel *));
  for (i=0;i<md->nb;i++) {
    md->mds[i]=(HmmModel *)calloc(1,sizeof(HmmModel));
    fscanf(infile, "\n!!!!!!!==== HMM for variable block %d =====!!!!!!\n",&ii);
    k=readascii_hmm(md->mds[i],infile);
    if (k==0) return(0);
  }
  
  return(1);

}

unsigned char readascii2_ccm(CondChain *md, FILE *infile)
{
  int i,j,k;
  int ii,jj;
  
  fscanf(infile, "%d\n", &(md->dim));
  fscanf(infile, "%d\n", &(md->nb));
  fscanf(infile, "%d\n", &(md->maxnumst));

  md->bdim=(int *)calloc(md->nb,sizeof(int));
  for (i=0;i<md->nb;i++) fscanf(infile, "%d", md->bdim+i);
  fscanf(infile, "\n");

  md->var=(int **)calloc(md->nb,sizeof(int *));
  for (i=0;i<md->nb;i++) {
    md->var[i]=(int *)calloc(md->bdim[i],sizeof(int));
  }

  for (i=0;i<md->nb;i++) {
    for (j=0;j<md->bdim[i];j++)
      fscanf(infile, "%d ", md->var[i]+j);
    fscanf(infile, "\n");
  }
  
  md->numst=(int *)calloc(md->nb,sizeof(int));
  for (i=0;i<md->nb;i++) fscanf(infile, "%d ", md->numst+i);
  fscanf(infile, "\n");

  md->cbdim=(int *)calloc(md->nb,sizeof(int));
  md->cbdim[0]=0;
  md->cnumst=(int *)calloc(md->nb,sizeof(int));
  md->cnumst[0]=0;
  for (i=0;i<md->nb-1;i++) {
    md->cbdim[i+1]=md->cbdim[i]+md->bdim[i];
    md->cnumst[i+1]=md->cnumst[i]+md->numst[i];
  }

  md->mds=(HmmModel **)calloc(md->nb,sizeof(HmmModel *));
  for (i=0;i<md->nb;i++) {
    md->mds[i]=(HmmModel *)calloc(1,sizeof(HmmModel));
    k=readascii2_hmm(md->mds[i],infile);
    if (k==0) return(0);
  }
  
  return(1);

}


void printclsinfo(FILE *outfile, int *clsid, int nseq, int numcls)
{
  int *ct;
  int i;

  ct=(int *)calloc(numcls,sizeof(int));
  for (i=0;i<numcls;i++) ct[i]=0;

  for (i=0;i<nseq;i++) {
    ct[clsid[i]]++;
  }

  for (i=0;i<numcls;i++){
    fprintf(outfile, "%d ", ct[i]);
  }
  fprintf(outfile, "\n");

  for (i=0;i<nseq;i++) {
    fprintf(outfile, "%d\n", clsid[i]);
  }

  free(ct);
}

void readrefcls(FILE *reffile, int *rncls, int *rnseq, double ***refmode, double **refsigma, int ***refpath,
		int **refcls, int dim, int nb, int *nseq, int **ct)
{
  int i,j;
  float tp;
  
  fscanf(reffile, "%d %d %d\n", rncls, rnseq, nseq);
  *ct=(int *)calloc(*rncls,sizeof(int));
  for (i=0;i<*rncls;i++) fscanf(reffile, "%d ", (*ct)+i);
  fscanf(reffile, "\n");

  *refmode=(double **)calloc(*rncls,sizeof(double *));
  for (i=0;i<*rncls;i++)
    (*refmode)[i]=(double *)calloc(dim,sizeof(double));
  *refsigma=(double *)calloc(dim,sizeof(double));
  for (i=0;i<dim;i++){
    fscanf(reffile, "%e ", &tp);
    (*refsigma)[i]=(double)tp;
  }
  fscanf(reffile, "\n");
  for (i=0;i<*rncls;i++) {
    for (j=0;j<dim;j++) {
      fscanf(reffile, "%e ", &tp);
      (*refmode)[i][j]=tp;
    } 
    fscanf(reffile, "\n");
  }
  
  *refpath=(int **)calloc(*rnseq,sizeof(int *));
  for (i=0;i<*rnseq;i++)
    (*refpath)[i]=(int *)calloc(nb,sizeof(int));
  *refcls=(int *)calloc(*rnseq,sizeof(int));
  
  for (i=0;i<*rnseq;i++) {
    for (j=0;j<nb;j++) {
      fscanf(reffile, "%d ", (*refpath)[i]+j);
    } 
    fscanf(reffile, "%d", (*refcls)+i);
    fscanf(reffile, "\n");
  }
}

void printrefcls(FILE *reffile, int rncls, int rnseq, double **refmode, double *refsigma, int **refpath,
		 int *refcls, int dim, int nb, int nseq, int *ct)
{
  int i,j;
  
  fprintf(reffile, "%d %d %d\n", rncls, rnseq, nseq);
  for (i=0;i<rncls;i++) fprintf(reffile, "%d ", ct[i]);
  fprintf(reffile, "\n");
  for (i=0;i<dim;i++){
    fprintf(reffile, "%e ", refsigma[i]);
  }
  fprintf(reffile, "\n");
  for (i=0;i<rncls;i++) {
    for (j=0;j<dim;j++) {
      fprintf(reffile, "%e ", refmode[i][j]);
    } 
    fprintf(reffile, "\n");
  }
  
  for (i=0;i<rnseq;i++) {
    for (j=0;j<nb;j++) {
      fprintf(reffile, "%d ", refpath[i][j]);
    } 
    fprintf(reffile, "%d", refcls[i]);
    fprintf(reffile, "\n");
  }
}
