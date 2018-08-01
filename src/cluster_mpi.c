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
#include "cluster_mpi.h"


void split(double *cdwd, double *newcdwd, int dim, double *stddev)
{
  double mult_offset=0.1;
  int i;
  int rank; // rank of MPI process
  /** if cdwd is way off zero and the variance is small     **/
  /** the offset may be out of the data range and generates **/
  /** empty cells.                                          **/
  /*****
  for (i=0; i<dim; i++) {
    newcdwd[i] = cdwd[i]*(1+mult_offset*drand48());
  }
  *****/
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // sample new centroid on master process
  /* set random range to [0.25, 0.75] */
  if (rank == 0){
	for (i=0; i<dim; i++) {
		newcdwd[i] = cdwd[i]+stddev[i]*mult_offset*(0.25+drand48()/2.0);
	}
  }
  
  // send coordinates of a new centroid to all processes
  MPI_Bcast(newcdwd, dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// index - array with cluster id
void centroid(double *cdbk, int dim, int numcdwd, double *vc, 
	 int *index, int numdata)
{
  int i,j,k,m,n;
  int *ct, *ct_local;
  double *cdbk_local;
  int rank; // rank of MPI process
  int numdata_total;

  ct_local=(int *)calloc(numcdwd, sizeof(int)); // number of points in clusters in process
  ct=(int *)calloc(numcdwd, sizeof(int)); // total number of points in clusters
  cdbk_local=(double *)calloc(numcdwd*dim, sizeof(double)); // local centroid coordinates
 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
  // calculate total number of datapoints
  MPI_Reduce(&numdata, &numdata_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
  if (index==NULL)
  {  	
    for (i=0; i<numdata; i++) 
		for (k=0; k<dim; k++) 
			cdbk_local[k]+=vc[i*dim+k];
    
    if (rank == 0)
		for (k=0; k<dim; k++)
			cdbk[k] = 0.0;
	
    MPI_Reduce(cdbk_local, cdbk, dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
		for (k=0; k<dim; k++)
			cdbk[k] /= ((double)numdata_total);
  }
  else
  {
      
    for (i=0; i<numdata; i++) {
		for (k=0; k<dim; k++) 
			cdbk_local[index[i]*dim+k]+=vc[i*dim+k];
			ct_local[index[i]]++;
    }
    
    // calculate total number of datapoints in each cluster
    MPI_Reduce(ct_local, ct, numcdwd, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    MPI_Reduce(cdbk_local, cdbk, numcdwd*dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
    { 
		for (j=0; j<numcdwd; j++) 
			for (k=0; k<dim; k++)
				cdbk[j*dim+k] /= ((double)ct[j]);
  
	}
  }
  
  // send centroid coordinates to all processes
  MPI_Bcast(cdbk, numcdwd*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  free(ct);
  free(ct_local);
  free(cdbk_local);
}  

void cellstdv(double *cdbk, double *stddev, int dim, int numcdwd, double *vc,
	      int *index,  int numdata)
{
  int i,j,k,m,n;
  int rank; // rank of MPI process
  int *ct, *ct_local;
  double *stddev_local;

  ct_local=(int *)calloc(numcdwd, sizeof(int)); // number of datapoints in cluster in process
  ct=(int *)calloc(numcdwd, sizeof(int)); // total number of datapoints in cluster

  stddev_local=(double *)calloc(numcdwd*dim, sizeof(double)); // standard deviations from cluster center in process
   
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
  // calculate deviation from cluster center on each process   
  for (i=0; i<numdata; i++) {
    for (k=0; k<dim; k++) 
      stddev_local[index[i]*dim+k]+=((vc[i*dim+k]-cdbk[index[i]*dim+k])*
	(vc[i*dim+k]-cdbk[index[i]*dim+k]));
    ct_local[index[i]]++;
  }
  
  
  // calculate total number of datapoints in each cluster
  MPI_Reduce(ct_local, ct, numcdwd, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  
  // get data to master process  
  MPI_Reduce(stddev_local, stddev, numcdwd*dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
  if (rank == 0){
	for (j=0; j<numcdwd; j++) { 
		if (ct[j]>0) {
		  for (k=0; k<dim; k++) {
			stddev[j*dim+k] /= ((double)ct[j]);
			stddev[j*dim+k]=sqrt(stddev[j*dim+k]);
		  }
		}
		else {
		  for (k=0; k<dim; k++) stddev[j*dim+k]=1.0;
		}
	  }
  }
  
  free(ct);
  free(ct_local);
  free(stddev_local);
}  


double mse_dist(double *cdwd, double *vc, int dim)
{
  double mse= 0.0;
  int i;

  for (i=0; i<dim; i++)
    mse += (vc[i]-cdwd[i])*(vc[i]-cdwd[i]);

  return(mse);
}

// find cluster id as id of the nearest cluster to datapoint
void encode(double *cdbk, int dim, int numcdwd, double *vc, int *code,
	    int numdata)
{
  int i,j,k,m,n;
  double *dist,minv;

  dist=(double *)calloc(numcdwd,sizeof(double));

  for (i=0; i<numdata; i++) {
    for (j=0; j<numcdwd;j++)
      dist[j]=mse_dist(cdbk+j*dim, vc+i*dim, dim);    
    code[i]=0;
    minv=dist[0];
    for (j=1;j<numcdwd;j++)
      if (dist[j]<minv) {
		minv=dist[j];
		code[i]=j;
      }
  }

  free(dist);
}


double lloyd(double *cdbk, int dim, int numcdwd, double *vc, int numdata, 
	    double threshold)
     // cdbk - coordinates of centroids
     // vc - array with data in variable block
     // numcdwd - number of states in variable block
     // dim - dimensionality of data in variable block
     // index - array with cluster id
     /* return the value of the mean squared distance, i.e., average */
     /* squared distance between a sample and its centroid           */
     /* threshold is for controling when to stop the loop */
{
  int i,j,k,m,n;
  int ite;
  double dist_local, dist, olddist, minmse, mse;
  int min_iteration=2;
  /*double threshold = 0.005;*/
  int *index; // array with state id of sample points
  double *tp;
  int cdbksz2; // number of times clusters split
  int temp_int, new_cdwds, cdbksz;
  int numdata_total; // total number of samples in dataset
  double *stddev; // standard deviation for appropriate split

  srand48(0);//5/26/2017, to ensure identical result when rerun
  
  // calculate total number of samples in dataset
  MPI_Allreduce(&numdata, &numdata_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  cdbksz2 = 0;
  temp_int = 1;
  while (temp_int < numcdwd) {
    cdbksz2++;
    temp_int += temp_int;
  }  


  index = (int *)calloc(numdata, sizeof(int));
  stddev = (double *)calloc(numcdwd*dim,sizeof(double));

  centroid(cdbk, dim, 1, vc, NULL, numdata); // centroid coordinate for single cluster

  /* compute standard deviation for each cell */
  for (i=0;i<numdata;i++) index[i]=0;
  cellstdv(cdbk,stddev,dim,numcdwd,vc,index,numdata);

  if (numcdwd==1) {
    dist_local = 0.0;
    for (i=0, k=0; i<numdata; i++)
      for (j=0; j<dim; j++, k++)
		dist_local += (cdbk[j]-vc[k])*(cdbk[j]-vc[k]);
		
	MPI_Allreduce(&dist_local, &dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
    dist /= ((double)numdata_total);
  }

  cdbksz = 1;

  for (ite=0; ite<cdbksz2; ite++) {
    new_cdwds = ((numcdwd - 2*cdbksz) >= 0) ? cdbksz : numcdwd - cdbksz;

    for (k=0; k<new_cdwds; k++)
      split(cdbk+k*dim, cdbk+(cdbksz+k)*dim, dim, stddev+k*dim);

    cdbksz += new_cdwds;

    dist=HUGE;
    m = 0;

    while (m < min_iteration || 
	   (fabs((double)(olddist-dist))/olddist > threshold
	    && dist < olddist))
    {
		m++;
		olddist = dist;
		tp = vc;
		dist_local = 0.0;
		// assign cluster id as id of the nearest cluster
		for (i=0; i<numdata; i++){
			minmse = mse_dist(cdbk, tp, dim);
			index[i]= 0;
			
			for (j=1; j<cdbksz; j++){
				mse = mse_dist(cdbk+j*dim, tp, dim);
				if (mse<minmse){
					minmse=mse;
					index[i]=j;
				}
			}
				
			dist_local += minmse;
			tp += dim;
		}
		
		// sum distances across all processes
		MPI_Allreduce(&dist_local, &dist, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
   
		dist /= ((double)numdata_total);

		centroid(cdbk, dim, cdbksz, vc, index, numdata);
	}
	cellstdv(cdbk,stddev,dim,cdbksz,vc,index,numdata);

  }

  free(index);
  free(stddev);

  return(dist);

}	    


/** The kmeans algorithm is close to lloyd, except that the number **/
/** codewords is not preselected.  Instead, it is determined by    **/
/** the minimum number of codewords that lead to a mean squared    **/
/** error below a given threshold.  The number of codewords is     **/
/** upper bounded by the given maxnumcdwd.                         **/

double kmeans(double *cdbk, int dim, int maxnumcdwd, int *fnumcdwd, 
	     double *vc, int numdata, double threshold, double distthred)
{
  int i,j,k,m,n;
  int ite, splitwd;
  double dist, dist_local, olddist, minmse, mse;
  int min_iteration=2;
  int numcdwd;
  int numdata_total; // total number of samples in dataset
  
  // calculate total number of samples in dataset
  MPI_Allreduce(&numdata, &numdata_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  centroid(cdbk, dim, numcdwd, vc, NULL, numdata);

  dist_local = 0.0;
  for (i=0, k=0; i<numdata; i++)
    for (j=0; j<dim; j++, k++)
      dist += (cdbk[j]-vc[k])*(cdbk[j]-vc[k]);
  
  MPI_Allreduce(&dist_local, &dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  dist /= ((double)numdata_total);
  
  if (dist<distthred) {
    *fnumcdwd=1;
    return(dist);
  }

  numcdwd=2;
  do {
    dist=lloyd(cdbk, dim, numcdwd, vc, numdata, threshold);
    numcdwd++;
    //fprintf(stderr, "numcdwd=%d, dist=%f\n", numcdwd,dist);
  } while (numcdwd<=maxnumcdwd && dist > distthred);
  
  *fnumcdwd=numcdwd-1;
  
  return(dist);
}	    

