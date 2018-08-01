/**************************************************************************/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*                                                                        */
/*                        Jia  Li                                         */
/*  Copyright: Jia Li, Feb, 2004                                          */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*                                                                        */
/* Function:                                                              */
/*   Header file for agglomerate.c                                        */
/*                                                                        */
/*------------------------------------------------------------------------*/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/**************************************************************************/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct tree_struct
{
  struct tree_struct *left;
  struct tree_struct *right;
  struct tree_struct *parent;
  int depth;
  int numsps;
  int *sps;
  int lvdes;
  int id; //1, ..., n (#data points) for the nodes formed by the original data
          //n+1 for the first merged nodes, so on so forth, n+n-1 for the root
  float mdist; //distance at which the merge for creating this node happens
               //mdist is not meaningful for the nodes containing only the
               //original data points (that is, the leaf nodes)
} TreeNode;


/*----------------------------------------------------------*/
/* agglomerate.c                                            */
/*----------------------------------------------------------*/

extern void newtreenode(TreeNode **node_pt, TreeNode *left, TreeNode *right, 
		 TreeNode *parent, int id, int numsamples, int *sps);
extern void freetreenode(TreeNode **node_pt);
extern int countleaf(TreeNode *root);
extern void setdepth(TreeNode *root);
extern void setlvdes(TreeNode *root);
extern void setid(TreeNode *root);
extern void print_leafdat(TreeNode *rt);
extern void idbottomup(TreeNode *root, int ***treemt_pt, int *nr, int *nc);
extern float updatedist(float dist1, float dist2, float drs, 
		 TreeNode *cls1, TreeNode *cls2, TreeNode *thiscls,
		 int flag);
extern void aggcluster(float **dist, int numdata, float disthred, 
		       int flag, int numroots, int *cls, 
		       TreeNode ***ndlist, int *numnd);
extern void cuttree_numcls(TreeNode *rt, int ncls, int *cls);
extern void find_distances(TreeNode *rt, int *nd, float **mydist, int **myids);
extern void cuttree_dist(TreeNode *rt, float dthred, int *cls);
extern void print_cluster(int *cls, int len, int *code, FILE *outfile);

