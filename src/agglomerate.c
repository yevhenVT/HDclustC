/**************************************************************************/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*                                                                        */
/*                        Jia  Li                                         */
/*  Copyright: Jia Li, Feb, 2004                                          */
/*  Revised October, 2016                                                 */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*                                                                        */
/* Function:                                                              */
/*   Subroutines for agglomerative clustering                             */
/*   The main agglomerative clustering algorithm is performed by          */
/*   aggcluster().  The flag argument of aggcluster determines the        */
/*   way distance is agglomerated.  The other subroutines are called      */
/*   by aggcluster().                                                     */
/*   Subroutines for cutting a tree that has grown to a single root       */
/*   node: cuttree_numcls(), cuttree_dist(). The first cut by a given     */
/*   number of desired clusters and the second by a stopping distance     */
/*   threshold.                                                           */
/*                                                                        */
/*-------------------------------------------------------------------------
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/**************************************************************************/

#include "agglomerate.h"
#include "cluster_mpi.h"

void newtreenode(TreeNode **node_pt, TreeNode *left, TreeNode *right, 
		 TreeNode *parent, int id, int numsamples, int *sps)
{
  int i,j,k,m,n;
  TreeNode *node;

  node=(TreeNode *)calloc(1,sizeof(TreeNode));
  node->left=left;
  node->right=right;
  node->parent=parent;

  node->depth=0;
  node->numsps=numsamples;

  node->lvdes=0;
  node->id=id;
  node->mdist=0.0;
  
  if (sps!=NULL) {
    node->sps=(int *)calloc(numsamples,sizeof(int));
    for (i=0;i<numsamples;i++) node->sps[i]=sps[i];
  }
  else {
    node->sps=NULL;
  }

  *node_pt=node;

}

void freetreenode(TreeNode **node_pt)
{
  int i,j,k,m,n;
  TreeNode *node;

  node=*node_pt;
  if (node->left!=NULL) freetreenode(&(node->left));
  if (node->right!=NULL) freetreenode(&(node->right));

  if (node->sps!=NULL) free(node->sps);
  free(node);

  *node_pt=NULL;
}

int countleaf(TreeNode *root)
{
  int i,j,k,m,n, numleaves;
  TreeNode *cur, *pre;

  cur=root;
  pre=NULL;

  numleaves=0;
  
  while (cur!=NULL) {
    if (cur->left==NULL) {
      numleaves++;
      pre=cur;
      cur=cur->parent;
    }
    else {
      if (pre==cur->left) {
	pre=cur;
	cur=cur->right;
      }
      else {
	if (pre==cur->right) {
	  pre=cur;
	  cur=cur->parent;
	}
	else { // pre==cur->parent
	  pre=cur;
	  cur=cur->left;
	}
      }
    }
  }

  return(numleaves);
}



void setdepth(TreeNode *root)
{
  int i,j,k,m,n, depth;
  TreeNode *cur, *pre;

  cur=root;
  pre=NULL;

  depth=0;
  
  while (cur!=NULL) {
    if (cur->left==NULL) {
      cur->depth=depth;
      pre=cur;
      cur=cur->parent;
      depth--;
    }
    else {
      if (pre==cur->left) {
	pre=cur;
	cur=cur->right;
	depth++;
      }
      else {
	if (pre==cur->right) {
	  cur->depth=depth;
	  pre=cur;
	  cur=cur->parent;
	  depth--;
	}
	else { // pre==cur->parent
	  pre=cur;
	  cur=cur->left;
	  depth++;
	}
      }
    }
  }
}


void setlvdes(TreeNode *root)
{
  int i,j,k,m,n, depth;
  TreeNode *cur, *pre;

  cur=root;
  pre=NULL;

  while (cur!=NULL) {
    if (cur->left==NULL) {
      cur->lvdes=0;
      pre=cur;
      cur=cur->parent;
    }
    else {
      if (pre==cur->left) {
	pre=cur;
	cur=cur->right;
      }
      else {
	if (pre==cur->right) {
	  cur->lvdes=((cur->left->lvdes > cur->right->lvdes)? 
	    (cur->left->lvdes):(cur->right->lvdes))+1;
	  pre=cur;
	  cur=cur->parent;
	}
	else { // pre==cur->parent
	  pre=cur;
	  cur=cur->left;
	}
      }
    }
  }
}


void setid(TreeNode *root)
{
  int i,j,k,m,n, depth,id,numleaves, numlv;
  TreeNode *cur, *pre;
  int **treemt;

  numleaves=countleaf(root);
  numlv=root->lvdes+1;

  depth=root->lvdes;

  id=1;
  for (m=0; m<=depth; m++) {
    cur=root;
    pre=NULL;
    while (cur!=NULL) {
      if (cur->lvdes<=m) {
	if (cur->lvdes==m) {
	  cur->id=id;
	  id++;
	}
	pre=cur;
	cur=cur->parent;
      }
      else {
	if (pre==cur->left) {
	  pre=cur;
	  cur=cur->right;
	}
	else {
	  if (pre==cur->right) {
	    pre=cur;
	    cur=cur->parent;
	  }
	  else { // pre==cur->parent
	    pre=cur;
	    cur=cur->left;
	  }
	}
      }
    }
    
  }

}

void idbottomup(TreeNode *root, int ***treemt_pt, int *nr, int *nc)
{
  int i,j,k,m,n, id, depth,numleaves, numlv;
  TreeNode *cur, *pre;
  int **treemt;

  numleaves=countleaf(root);
  numlv=root->lvdes+1;

  treemt=(int **)calloc(numlv,sizeof(int *));
  for (i=0; i<numlv; i++)
    treemt[i]=(int *)calloc(numleaves,sizeof(int));
  for (i=0; i<numlv; i++)
    for (j=0; j<numleaves; j++)
      treemt[i][j]=0;  

  depth=root->lvdes;

  id=1;
  for (m=0; m<=depth; m++) {
    if (m>0) {
      for (i=0; i<numleaves; i++)
	treemt[m][i]=treemt[m-1][i];
    }

    cur=root;
    pre=NULL;
    while (cur!=NULL) {
      if (cur->lvdes<=m) {
	if (cur->lvdes==m) {
	  if (m==0) 
	    { treemt[m][id-1]=cur->id; }
	  else {
	    for (i=0; i<numleaves; i++) {
	      if (treemt[m-1][i]==cur->left->id || 
		  treemt[m-1][i]==cur->right->id)
		treemt[m][i]=cur->id;
	    }
	  }
	  id++;
	}
	pre=cur;
	cur=cur->parent;
      }
      else {
	if (pre==cur->left) {
	  pre=cur;
	  cur=cur->right;
	}
	else {
	  if (pre==cur->right) {
	    pre=cur;
	    cur=cur->parent;
	  }
	  else { // pre==cur->parent
	    pre=cur;
	    cur=cur->left;
	  }
	}
      }
    }
    
  }


  *treemt_pt=treemt;
  *nr=numlv;
  *nc=numleaves;
}

void print_leafdat(TreeNode *root)
//Modified from the original dendrogram clustering package
//not assuming vector data
{
  int i,j,k,m,n;
  TreeNode *cur, *pre;

  cur=root;
  pre=NULL;

  while (cur!=NULL) {
    if (cur->left==NULL) {
      for (i=0; i<cur->numsps; i++) {
	fprintf(stdout, "%d ", cur->sps[i]);
      }
      fprintf(stdout, "\n");
      pre=cur;
      cur=cur->parent;
    }
    else {
      if (pre==cur->left) {
	pre=cur;
	cur=cur->right;
      }
      else {
	if (pre==cur->right) {
	  pre=cur;
	  cur=cur->parent;
	}
	else { // pre==cur->parent
	  pre=cur;
	  cur=cur->left;
	}
      }
    }
  }
}


void leafdatid(TreeNode *root, int *cls)
{
  int i,j,k,m,n,ct;
  TreeNode *cur, *pre;

  cur=root;
  pre=NULL;

  ct=0;
  while (cur!=root->parent) {
    //old code cur!=NULL is only suitable for root being the real root of a whole tree and is invalide if root is any node in a tree
    if (cur->left==NULL) {
      cls[ct]=cur->sps[0];
      ct++;
      pre=cur;
      cur=cur->parent;
    }
    else {
      if (pre==cur->left) {
	pre=cur;
	cur=cur->right;
      }
      else {
	if (pre==cur->right) {
	  pre=cur;
	  cur=cur->parent;
	}
	else { // pre==cur->parent
	  pre=cur;
	  cur=cur->left;
	}
      }
    }
  }

}


float updatedist(float dist1, float dist2, float drs, 
		 TreeNode *cls1, TreeNode *cls2, TreeNode *thiscls,
		 int flag)
{
  int i,j,k,m,n;
  float res,p,p1,p2;

  if (flag>6 || flag<0) {
    fprintf(stderr, "WARNING: algorithm flag is not properly set, 0 used.\n");
    flag=0;
  }

  if (flag==0) { //single link 
    res=(dist1<dist2)?dist1:dist2;
    return(res);
  }

  if (flag==1) { // complete link
    res=(dist1<dist2)?dist2:dist1;
    return(res);
  }

  if (flag==2) { // average link, unweighted
    p=((float)cls1->numsps)/((float)(cls1->numsps+cls2->numsps));
    res=p*dist1+(1-p)*dist2;
    return(res);
  }

  if (flag==3) { // average link, weighted
    res=0.5*dist1+0.5*dist2;
    return(res);
  }

  if (flag==4) { // centroid clustering, unweighted
    p=((float)cls1->numsps)/((float)(cls1->numsps+cls2->numsps));
    res=p*dist1+(1-p)*dist2-p*((float)cls2->numsps)*drs;
    return(res);
  }

  if (flag==5) { // centroid clustering, weighted
    p=((float)cls1->numsps)/((float)(cls1->numsps+cls2->numsps));
    res=0.5*dist1+0.5*dist2-0.25*drs;
    return(res);
  }

  if (flag==6) { // Ward's clustering
    p1=((float)(cls1->numsps+thiscls->numsps))/
	((float)(cls1->numsps+cls2->numsps+thiscls->numsps));
    p2=((float)(cls2->numsps+thiscls->numsps))/
	((float)(cls1->numsps+cls2->numsps+thiscls->numsps));
    p=((float)thiscls->numsps)/
	((float)(cls1->numsps+cls2->numsps+thiscls->numsps));
    res=p1*dist1+p2*dist2-p*drs;
    return(res);
  }
  return(res);
}

void aggcluster(float **dist, int numdata, float disthred,
		int flag, int numroots, int *cls, TreeNode ***ndlist, 
		int *numnd)
{
  int i,j,k,m,n,m1,m2,ndlen, idct;
  TreeNode *root;
  TreeNode **curndlist, *newnd;
  float mindist,v1,v2;
  
  /** leaf nodes **/
  curndlist=(TreeNode **)calloc(numdata,sizeof(TreeNode *));
  for (i=0; i<numdata; i++)
    newtreenode(curndlist+i, NULL, NULL, NULL, i, 1, &i);
  for (i=0; i<numdata; i++) cls[i]=i;
  ndlen=numdata;

  mindist=0.0;
  idct=numdata;
  while (ndlen>numroots && mindist < disthred) {
    /** Find the closest pair of clusters **/
    m1=0; m2=1; // m1<m2 is guaranteed 
    mindist=dist[m1][m2];
    for (i=0; i<ndlen; i++)
      for (j=i+1; j<ndlen; j++){
	if (dist[i][j]<mindist){
	  mindist=dist[i][j];
	  m1=i;
	  m2=j;
	}
      }
    if (m1>m2) {k=m1; m1=m2; m2=k;}  // should not happen

    if (mindist>=disthred) break;

    /** generate a new node by merging the two nodes pointed **/

    newtreenode(&newnd, curndlist[m1], curndlist[m2], NULL, idct,
		curndlist[m1]->numsps+curndlist[m2]->numsps, NULL);
    newnd->mdist=mindist; //the distance at which merge happens
    
    idct++;
    curndlist[m1]->parent=newnd;
    curndlist[m2]->parent=newnd;

    /** update dist **/    
    v2=dist[m1][m2];
    for (j=0; j<ndlen; j++) {
      if (j==m1 || j==m2) continue;
      v1=updatedist(dist[j][m1],dist[j][m2], v2, curndlist[m1], 
		    curndlist[m2], curndlist[j], flag);
      dist[m1][j]=dist[j][m1]=v1;
    }

    /** delete the m2 th row and column **/
    for (i=m2; i<ndlen-1; i++) 
      for (j=0; j<ndlen; j++)
	dist[i][j]=dist[i+1][j];
    
    for (j=m2; j<ndlen-1; j++)
      for (i=0; i<ndlen-1; i++)
	dist[i][j]=dist[i][j+1];

    curndlist[m1]=newnd;
    for (i=m2; i<ndlen-1; i++)
      curndlist[i]=curndlist[i+1];
    ndlen-=1;

    for (i=0; i<numdata; i++) {
      if (cls[i]==m2) {
	cls[i]=m1;
      }
      else {
	if (cls[i]>m2)
	  cls[i]-=1;
      }
    }
  }


  *ndlist = (TreeNode **)calloc(ndlen, sizeof(TreeNode *));
  for (i=0; i<ndlen; i++)
    (*ndlist)[i]=curndlist[i];
  *numnd=ndlen;

  /*------------- release memory -------------*/
  free(curndlist);

}


/*----- Cutting out a tree (stop criteria) given a tree root -----*/
/*--- provide clustering result at a desired number of clusters --*/
void cuttree_numcls(TreeNode *rt, int ncls, int *cls)
//cls[rt->numsps]
{
  int i,j,k,m,n;
  int nleaf, idthred;
  TreeNode **clsnode, *cur, *pre;
  int *cls2;
  int ct;

  //needed if not assuming each leaf contains a single data point
  nleaf=countleaf(rt); 
  clsnode=(TreeNode **)calloc(nleaf,sizeof(TreeNode *));
  
  if (ncls<1 || ncls>nleaf) {
    fprintf(stderr, "Number of clusters required %d is impossible\n",ncls);
    exit(0);
  }

  idthred=2*nleaf-ncls-1;
  
  cur=rt;
  pre=NULL;
  ct=0;
  while (cur!=NULL) {
    if (cur->left==NULL) {
      if ((cur->id)>idthred) {
	fprintf(stderr, "Error: improper number of clusters requested\n");
	fprintf(stderr, "leaf id: %d, #clusters: %d, id threshold: %d\n",
		cur->id, ncls, idthred);
	exit(0);
      }
      else {
	clsnode[ct]=cur;
	ct++;
      }
      pre=cur;
      cur=cur->parent;
    }
    else {
      if (pre==cur->left) {
	pre=cur;
	cur=cur->right;
      }
      else {
	if (pre==cur->right) {
	  //will leave this node and all its descendents permanently
	  //all done with this branch
	  //any process to be done on this node should be done now
	  pre=cur;
	  cur=cur->parent;
	}
	else { // pre==cur->parent, first time this node is visited
	  if ((cur->id) <= idthred) {
	    clsnode[ct]=cur;
	    ct++;
	    pre=cur; //visits to descendents can be terminated 
	    cur=cur->parent;
	  }
	  else {
	    pre=cur;
	    cur=cur->left;
	  }
	}
      }
    }
  }

  //clsnode[ct] contains pointers to all the nodes that are valid clusters
  for (i=0,m=0;i<ct;i++) {
    if (clsnode[i]->numsps > m) m=clsnode[i]->numsps;
  }
  cls2=(int *)calloc(m,sizeof(int)); //????? why this doesn't work?
  
  for (i=0;i<ct;i++) {
    leafdatid(clsnode[i],cls2);
    for (j=0;j<clsnode[i]->numsps;j++)
      cls[cls2[j]]=i;
  }
  
  free(clsnode);
  free(cls2);
  
  return;
}

// Find all the merge distances at all the internal nodes
void find_distances(TreeNode *rt, int *nd, float **mydist, int **myids)
{
  int i,j,k,m,n;
  int nleaf, idthred;
  int *ids;
  float *dist;
  TreeNode *cur, *pre;
  int ct;

  //needed if not assuming each leaf contains a single data point
  *nd=countleaf(rt)-1;
  dist=(float *)calloc(*nd,sizeof(float));
  ids=(int *)calloc(*nd,sizeof(int));

  cur=rt;
  pre=NULL;
  ct=0;
  while (cur!=NULL) {
    if (cur->left==NULL) {
      pre=cur;
      cur=cur->parent;
    }
    else {
      if (pre==cur->left) {
	pre=cur;
	cur=cur->right;
      }
      else {
	if (pre==cur->right) {
	  //will leave this node and all its descendents permanently
	  //all done with this branch
	  //any process to be done on this node should be done now
	  dist[ct]=cur->mdist;
	  ids[ct]=cur->id;
	  ct++;
	  pre=cur;
	  cur=cur->parent;
	}
	else { // pre==cur->parent, first time this node is visited
	  pre=cur;
	  cur=cur->left;
	}
      }
    }
  }

  //Order according to values in ids[nd]
  //It's known that ids[] contains consecutive integers according to the way
  //a linkage tree is built. Hence we only need to find the minimum value

  *myids=(int *)calloc(*nd,sizeof(int));
  *mydist=(float *)calloc(*nd,sizeof(float));
  for (i=1,m=ids[0];i<*nd;i++){
    if (ids[i]<m) m=ids[i];
  }
  for (i=0;i<*nd;i++) {
    n=ids[i]-m;
    (*myids)[n]=ids[i];
    (*mydist)[n]=dist[i];
  }

  free(dist);
  free(ids);
  return;
}

//Cut tree according to a distance threshold given
void cuttree_dist(TreeNode *rt, float dthred, int *cls)
{
  int i,j,k,m,n;
  float *dist;
  int *ids, nd, ncls, nleaf, idthred;

  nleaf=countleaf(rt);
  find_distances(rt, &nd, &dist, &ids);

  if (dist[0]>=dthred) {
    for (i=0;i<(rt->numsps); i++)
      cls[i]=i;
    return;
  }

  idthred=ids[0];
  for (i=1;i<nd;i++) {
    if (dist[i]>=dthred) { break; }
    else {idthred=ids[i];}
  }

  ncls=2*nleaf-idthred-1;
  cuttree_numcls(rt, ncls, cls);//cls[rt->numsps]
  
  free(dist);
  free(ids);
  
  return;
}

void print_cluster(int *cls, int len, int *code, FILE *outfile)
{
  int i,j,k,m,n;
  int ncls, *ct;

  for (i=0, ncls=0;i<len;i++){
    if (cls[i]>ncls) ncls=cls[i];
  }
  ncls++;

  ct=(int *)calloc(ncls,sizeof(int));
  for (i=0;i<ncls;i++) ct[i]=0;
  for (i=0;i<len;i++) ct[cls[i]]++;

  fprintf(outfile,"Number of clusters: %d\n",ncls);
  fprintf(outfile, "Cluster sizes (ID, Size): ");
  for (i=0;i<ncls;i++)
    fprintf(outfile, "(%d, %d) ", i, ct[i]);
  fprintf(outfile, "\n");

  fprintf(outfile, "Members in each cluster:\n");
  for (i=0;i<ncls;i++) {
    fprintf(outfile, "%d: ",i);
    for (j=0;j<len;j++) {
      if (cls[j]==i) //code[] is the original symbols for samples 0, ..., len-1
	fprintf(outfile, "%d ", code[j]);
    }
    fprintf(outfile, "\n");
  }

  free(ct);
}
