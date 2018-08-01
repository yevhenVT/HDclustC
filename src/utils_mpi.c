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

void scatter_doubleVectors(double **u, double ***u_local, int ntotal, int nlocal, int dim, int *nperproc)
{
  int i, j;
  int numproc; // number of MPI processes
  int rank; // rank of MPI process
  int cumn;
  MPI_Status status;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  *u_local =(double **)calloc(nlocal,sizeof(double *));
  
  for (i = 0; i < nlocal; i++)
	(*u_local)[i] = (double *)calloc(dim, sizeof(double));
	
  // copy data from global array to local for master process
  if (rank == 0){
	for (i = 0; i < nlocal; i++)
		for (j = 0; j < dim; j++)
			(*u_local)[i][j] = u[i][j];
  }
  
  // send data from master to slave processes
  if (rank == 0){
	cumn = nlocal;
	for (i = 1; i < numproc; i++){  
		for (j = 0; j < nperproc[i]; j++)
			MPI_Send(u[cumn+j], dim, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			
		cumn += nperproc[i];
	  }
  }
  else{	
	for (j = 0; j < nlocal; j++)
		MPI_Recv((*u_local)[j], dim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
  }  
}

void scatter_intVectors(int **u, int ***u_local, int ntotal, int nlocal, int dim, int *nperproc)
{
  int i, j;
  int numproc; // number of MPI processes
  int rank; // rank of MPI process
  int cumn;
  MPI_Status status;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  *u_local =(int **)calloc(nlocal,sizeof(int *));
  
  for (i = 0; i < nlocal; i++)
	(*u_local)[i] = (int *)calloc(dim, sizeof(int));
	
  // copy data from global array to local for master process
  if (rank == 0){
	for (i = 0; i < nlocal; i++)
		for (j = 0; j < dim; j++)
			(*u_local)[i][j] = u[i][j];
  }
  
  // send data from master to slave processes
  if (rank == 0){
	cumn = nlocal;
	for (i = 1; i < numproc; i++){  
		for (j = 0; j < nperproc[i]; j++)
			MPI_Send(u[cumn+j], dim, MPI_INT, i, 0, MPI_COMM_WORLD);
			
		cumn += nperproc[i];
	  }
  }
  else{	
	for (j = 0; j < nlocal; j++)
		MPI_Recv((*u_local)[j], dim, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
  }  
}

void gather_intVectors(int ***u, int **u_local, int ntotal, int nlocal, int dim, int *nperproc)
{
  int i, j;
  int numproc; // number of MPI processes
  int rank; // rank of MPI process
  int cumn;
  MPI_Status status;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0){
	*u = (int **)calloc(ntotal,sizeof(int *));
	for (i=0;i<ntotal;i++) (*u)[i]=(int *)calloc(dim,sizeof(int));
  }
  
  // copy data from local array to global for master process
  if (rank == 0){
	for (i = 0; i < nlocal; i++)
		for (j = 0; j < dim; j++)
			(*u)[i][j] = u_local[i][j];
  }
  
  // send data from master to slave processes
  if (rank == 0){
	cumn = nlocal;
	for (i = 1; i < numproc; i++){  
		for (j = 0; j < nperproc[i]; j++)
			MPI_Recv((*u)[cumn+j], dim, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			
		cumn += nperproc[i];
	  }
  }
  else{	
	for (j = 0; j < nlocal; j++)
		MPI_Send(u_local[j], dim, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }  
}

void distribute_work(int ntotal, int* nlocal, int **nperproc)
{
  int i;
  int numproc; // number of MPI processes
  int rank; // rank of MPI process
  int nremain;
  
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  
  *nperproc = (int *)calloc(numproc,sizeof(int));
  
  nremain = ntotal;
  for (i = 0; i < numproc; i++){
	  (*nperproc)[i] = ntotal / numproc;
	  nremain -= (*nperproc)[i];
  }
  
  i = 0;
  while (nremain > 0){
	  (*nperproc)[i] += 1;
	  nremain -= 1;
	  i++;
  }
  
  *nlocal = (*nperproc)[rank]; // number of sequences assigned to process
}

void send_new_hmm(HmmModel *md)
{
  int i,m;
  
  int rank; // MPI process rank

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  MPI_Bcast(&(md->dim), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(md->numst), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(md->prenumst), 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (rank != 0){
	md->a00=(double *)calloc(md->numst,sizeof(double));
    md->a=(double **)calloc(md->prenumst,sizeof(double *));
  
    for (i=0; i<md->prenumst; i++)
		md->a[i]=(double *)calloc(md->numst,sizeof(double));
  
	md->stpdf=(GaussModel **)calloc(md->numst, sizeof(GaussModel *));
	
	for (i=0; i<md->numst; i++)
		md->stpdf[i]=(GaussModel *)calloc(1, sizeof(GaussModel));
  }
  
  MPI_Bcast(md->a00, md->numst, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  for (i=0; i<md->prenumst; i++) 
	MPI_Bcast(md->a[i], md->numst, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  for (i=0; i<md->numst; i++) {
	MPI_Bcast(&(md->stpdf[i]->exist), 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&(md->stpdf[i]->dim), 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if (rank != 0){
		md->stpdf[i]->mean=(double *)calloc(md->stpdf[i]->dim,sizeof(double));
		md->stpdf[i]->sigma=(double **)calloc(md->stpdf[i]->dim,sizeof(double *));
		md->stpdf[i]->sigma_inv=(double **)calloc(md->stpdf[i]->dim,
					      sizeof(double *));
		
		for (m=0; m<md->stpdf[i]->dim; m++) {
			md->stpdf[i]->sigma[m]=(double *)calloc(md->stpdf[i]->dim, 
					      sizeof(double));
			
			md->stpdf[i]->sigma_inv[m]=(double *)calloc(md->stpdf[i]->dim, 
						  sizeof(double));
		}
    }
    
    MPI_Bcast(md->stpdf[i]->mean, md->stpdf[i]->dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	MPI_Bcast(&(md->stpdf[i]->sigma_det_log), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	for (m=0; m<md->stpdf[i]->dim; m++){
		MPI_Bcast(md->stpdf[i]->sigma[m], md->stpdf[i]->dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(md->stpdf[i]->sigma_inv[m], md->stpdf[i]->dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
  }
}

void send_new_ccm(CondChain *md)
{
  int i;
  
  int rank; // rank of MPI process
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Bcast(&(md->dim), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(md->nb), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(md->maxnumst), 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (rank != 0){
	md->bdim=(int *)calloc(md->nb,sizeof(int));
	md->numst=(int *)calloc(md->nb,sizeof(int));
	md->var=(int **)calloc(md->nb,sizeof(int *));
  } 
  
  MPI_Bcast(md->bdim, md->nb, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(md->numst, md->nb, MPI_INT, 0, MPI_COMM_WORLD);
  
  if (rank != 0){
	for (i=0;i<md->nb;i++)
		md->var[i]=(int *)calloc(md->bdim[i],sizeof(int));
  }
  
  for (i=0;i<md->nb;i++)
	MPI_Bcast(md->var[i], md->bdim[i], MPI_INT, 0, MPI_COMM_WORLD);
  

  // Set up the redundant fields for convenience of computation
  if (rank != 0){
	md->cbdim=(int *)calloc(md->nb,sizeof(int));
	md->cbdim[0]=0;
	md->cnumst=(int *)calloc(md->nb,sizeof(int));
	md->cnumst[0]=0;
	
	for (i=0;i<md->nb-1;i++) {
		md->cbdim[i+1]=md->cbdim[i]+md->bdim[i];
		md->cnumst[i+1]=md->cnumst[i]+md->numst[i];
	}
    
    md->mds=(HmmModel **)calloc(md->nb,sizeof(HmmModel *));
	for (i=0;i<md->nb;i++)
		md->mds[i]=(HmmModel *)calloc(1,sizeof(HmmModel));
  }

  for (i=0;i<md->nb;i++)
    send_new_hmm(md->mds[i]);
  
}

void send_comp_mode(CompMode *cmp_local, int newnseq_local, CompMode *cmp, int newnseq, int dim, int *newnseq_proc)
{
	int i,j;
	int cumnseq; // cumulative number of sequences
	int rank; // MPI process rank
	int numproc; // number of MPI processes
	MPI_Status status;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	
	// copy data from local to global array on master process
	if (rank == 0){
		for (i=0; i<newnseq_local; i++)
			for (j=0; j<dim; j++)
				cmp[i].mode[j] = cmp_local[i].mode[j];
	}
	
	// send data to master process
	if (rank == 0){
		cumnseq = newnseq_local;
		for (i=1; i<numproc; i++){
			for (j=0; j<newnseq_proc[i]; j++)
				MPI_Recv(cmp[cumnseq+j].mode, dim, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
		
			cumnseq += newnseq_proc[i];
		}	
	}
	else{	
		for (j = 0; j < newnseq_local; j++)
			MPI_Send(cmp_local[j].mode, dim, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}  
}
