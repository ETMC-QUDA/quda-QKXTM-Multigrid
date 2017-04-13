/*
# There is an unsigned integer "k" running from 1 unitl ...
# From this integer we can specify several important quantities regarding the coloring
# The total number of colors is given by N_{hc} = 2 * 2^{d(k-1)} where d is the number of dimensions
# The distance seperating neighbors carrying the same color is D=2^k
# The size of the elementary coloring is given L_u=2^{k-1}
# A condition must be fulfilled in order to be able to do the coloring for a specific k
# The condition must be that the number of blocks in each direction must be even
# And that Ls%(2Lu)=0 and Lt%(2Lu)=0
 */


/*
Description: This function computes the colors for the elementary color block
Inputs:
lc: Pointer to the array where we want to store the colors
Nc: The number of colors we want to put in the block
Lu: The extent of the color block
d: Dimension, either 2 or 3
 */

void fcb(unsigned short int lc[][2], const int Nc,const  int Lu, const int d){
  if(d==3)
    for(int i = 0 ; i < Lu; i++)
      for(int j = 0 ; j < Lu ; j++)
	for(int k = 0 ; k < Lu ; k++)
	  for(int eo = 0 ; eo < 2 ; eo++)
	    lc[i*Lu*Lu+j*Lu+k][eo] = (i*Lu*Lu+j*Lu+k)*2+eo;
  if(d==4)
    for(int i = 0 ; i < Lu; i++)
      for(int j = 0 ; j < Lu ; j++)
	for(int k = 0 ; k < Lu ; k++)
	  for(int l = 0 ; l < Lu ; l++)
	    for(int eo = 0 ; eo < 2 ; eo++)
	      lc[i*Lu*Lu*Lu+j*Lu*Lu+k*Lu+l][eo] = (i*Lu*Lu*Lu+j*Lu*Lu+k*Lu+l)*2+eo;
}

/*
  Description: This function gets a lexicographic index and return a position vector
x: The pointer to the vector
idx: the index
L: The spatial total extent (temporal dimension is the slowest in memory)
 */
void get_ind2Vec(int *x, const long int idx, const long int *L, const int d){
  
  if(d == 3){
    x[2]=idx/(L[0]*L[1]);
    x[1]=idx/L[0]-x[2]*L[1];
    x[0]=idx - x[2]*L[0]*L[1] - x[1]*L[0];
    }
  if(d==4){
    x[3]=idx/(L[0]*L[1]*L[2]);
    x[2]=idx/(L[0]*L[1]) - x[3]*L[2];
    x[1]=idx/L[0] - x[3]*L[1]*L[2] - x[2]*L[1];
    x[0]=idx-(x[3]*L[0]*L[1]*L[2]+x[2]*L[0]*L[1]+x[1]*L[0]);
  }
}

/*
  Description: Does the opposite of get_ind2Vec
 */

inline long int get_vec2Idx(const int *x, const long int *L, const int d){
  if (d==3)
    return x[0]+x[1]*L[0]+x[2]*L[0]*L[1];
  else
    return x[0]+x[1]*L[0]+x[2]*L[0]*L[1]+x[3]*L[0]*L[1]*L[2]; 
}

void create_hch_coloring(unsigned short int *Vc, long int lenVc, int Nc, int Lu, int d){
  unsigned short int (*lc)[2] = (unsigned short int(*)[2]) malloc(Nc*sizeof(unsigned short int));
  fcb(lc,Nc,Lu,d);

  int *x = (int*) malloc(d*sizeof(int));
  long int *GL = (long int *) malloc(d*sizeof(long int));
  long int *lu = (long int*) malloc(d*sizeof(long int));
  int *bx = (int*) malloc(d*sizeof(int));
  int *lx = (int*) malloc(d*sizeof(int));

  for(int i = 0 ; i < d ; i++) GL[i] = GK_localL[i];
  for(int i = 0 ; i < d ; i++) lu[i] = Lu;
  int eo;
  for(long int i=0; i < lenVc; i++){
    get_ind2Vec(x,i,GL,d);
    for(int j = 0 ; j < d ; j++)
      bx[j] =  x[j]/Lu; // find the position of each block
    eo=0;
    for(int j = 0 ; j < d ; j++) eo += bx[j]; // find if the block is even or odd
    eo=eo & 1;
    for(int j = 0 ; j < d ; j++)
      lx[j] = x[j] - Lu*bx[j]; // find the position inside block
    Vc[i] = lc[get_vec2Idx(lx,lu,d)][eo];
  }
  
  free(x);
  free(GL);
  free(lu);
  free(bx);
  free(lx);
}


//#define CHECK_COLORING

#ifdef CHECK_COLORING

inline static int brt(int x, int L){
  int y=x;
  if (y >= L)
    y=y%L;
  if (y < 0)
    y=y+L;
  return y;
}

void check_coloring(unsigned short int *Vc, int D, int d){
  long int *GL = (long int *) malloc(d*sizeof(long int));
  for(int i = 0 ; i < d ; i++) GL[i] = GK_localL[i];
  int *xx = (int*) malloc(d*sizeof(int));

  if(d==3){
    for(int z = 0 ; z < GK_totalL[2] ; z++)
      for(int y = 0 ; y < GK_totalL[1] ; y++)
	for(int x = 0 ; x < GK_totalL[0] ; x++){
	  xx[0] = x; xx[1] = y; xx[2] = z;
	  int c1 = Vc[get_vec2Idx(xx, GL,d)];
	  for(int dx = -D+1 ; dx < D ; dx++)
	    for(int dy = -D+1 ; dy < D ; dy++)
	      for(int dz = -D+1 ; dz < D ; dz++){
		int ds = abs(dx) + abs(dy) + abs(dz);
		if ((ds<D) && (ds != 0)){
		  int xn = x + dx;
		  xn = brt(xn,GK_totalL[0]);
		  int yn = y + dy;
		  yn = brt(yn,GK_totalL[1]);
		  int zn = z + dz;
		  zn = brt(zn,GK_totalL[2]);
		  xx[0] = xn; xx[1] = yn; xx[2] = zn;
		  int c2 = Vc[get_vec2Idx(xx, GL,d)];
		  if(c1 == c2){
		    errorQuda("Mistake found in the coloring with (%d,%d,%d) and (%d,%d,%d)",x,y,z,xn,yn,zn);
		  }
		}
	      }
	}
  }
  else{
    for(int t = 0 ; t < GK_totalL[3] ; t++)
      for(int z = 0 ; z < GK_totalL[2] ; z++)
	for(int y = 0 ; y < GK_totalL[1] ; y++)
	  for(int x = 0 ; x < GK_totalL[0] ; x++){
	    xx[0] = x; xx[1] = y; xx[2] = z; xx[3] = t;
	    int c1 = Vc[get_vec2Idx(xx, GL,d)];
	    for(int dx = -D+1 ; dx < D ; dx++)
	      for(int dy = -D+1 ; dy < D ; dy++)
		for(int dz = -D+1 ; dz < D ; dz++)
		  for(int dt = -D+1 ; dt < D ; dt++){
		    int ds = abs(dx) + abs(dy) + abs(dz) + abs(dt);
		    if ((ds<D) && (ds != 0)){
		      int xn = x + dx;
		      xn = brt(xn,GK_totalL[0]);
		      int yn = y + dy;
		      yn = brt(yn,GK_totalL[1]);
		      int zn = z + dz;
		      zn = brt(zn,GK_totalL[2]);
		      int tn = t + dt;
		      tn = brt(tn,GK_totalL[3]);
		      xx[0] = xn; xx[1] = yn; xx[2] = zn; xx[3] = tn;
		      int c2 = Vc[get_vec2Idx(xx, GL,d)];
		      if(c1 == c2){
			printfQuda("Colors (%d,%d)\n",c1,c2);
			errorQuda("Mistake found in the coloring with (%d,%d,%d,%d) and (%d,%d,%d,%d)",x,y,z,t,xn,yn,zn,tn);
		      }
		    }
		  }
	  }
  }

  free(xx);
  free(GL);
  printfQuda("Check for coloring passed!!!!");
}

#endif


// k is the integer number related with the distance
// d is the number of dimensions 2 or 3
unsigned short int* hch_coloring(int k, int d){
  if ((d != 3) && (d != 4) )
    errorQuda("Only 3 and 4 dimensions of coloring are allowed");
  if( k < 1)
    errorQuda("k must be greater than 1");
  int Nc = 2*pow(2,d*(k-1));
  int D = pow(2,k);
  int Lu = pow(2,k-1);
  printfQuda("Number of colors for hierarchical probing is %d\n",Nc);
  printfQuda("Distance of neigbors is %d\n",D);
  printfQuda("The extent of the symmetric color unit block is %d\n",Lu);
  for(int i = 0 ; i < d ; i++)
    if( (GK_localL[i] % (2*Lu)) != 0 )
      errorQuda("2*Lu cannot fit in the local lattice extent");
  if (Nc > 65536)
    errorQuda("Exceeded maximum number of colors");
  unsigned short int *Vc;
  
  long int lenVc;
  if(d==3)
    lenVc=GK_localL[0]*GK_localL[1]*GK_localL[2];
  else
    lenVc=GK_localL[0]*GK_localL[1]*GK_localL[2]*GK_localL[3];

  Vc = (unsigned short int *) malloc(sizeof(unsigned short int)*lenVc);
  create_hch_coloring(Vc, lenVc, Nc, Lu, d);
#ifdef CHECK_COLORING
  // check for colors is implemented only for 1 process
  for(int i = 0 ; i < d ; i++)
    if(GK_localL[i] != GK_totalL[i])
      errorQuda("Test coloring is available for only 1 MPI task");

  check_coloring(Vc,D,d);
#endif
  return Vc;
}


int HadamardElements(int i, int j){
  int sum=0;
    for(int k = 0 ; k < 32 ; k++){
      sum += (i%2)*(j%2);
      i=i>>1;
      j=j>>1;
    }
  if( (sum%2) ==0 )
    return 1;
  else
    return -1;
}

template <typename Float>
void get_probing4D_spinColor_dilution(void *probing_input_vector, void *input_vector, unsigned short int *Vc, int ih, int sc){
  memset(probing_input_vector,0,GK_localVolume*12*2*sizeof(Float));
  int c;
  int signProbing;
  for(int i = 0 ; i < GK_localVolume ; i++){
    c = Vc[i];
    int signProbing = HadamardElements(c,ih);
    for(int ri = 0 ; ri < 2 ; ri++)
      ((Float*)probing_input_vector)[i*12*2+sc*2+ri] = signProbing * ((Float*)input_vector)[i*12*2+sc*2+ri];
  }
}

//DMH
template <typename Float>
void get_Blocked_spinColor_dilution(void *blocked_vector, void *input_vector, 
				    int *element, int *Blk_scheme, 
				    int ih, int sc){

  int i = 0;
  //Zero out the blocked vector
  memset(blocked_vector,0,GK_localVolume*12*2*sizeof(Float));

  // Deduce element to be populated

  //The x-coord in the block
  element[0] = ih % Blk_scheme[0];
  
  //The y-coord in the block
  ih -= element[0];
  element[1] = (ih/Blk_scheme[0]) % Blk_scheme[1];
  
	  //The z-coord in the block
  ih -= element[1]*Blk_scheme[0];
  element[2] = ( ih/(Blk_scheme[0]*Blk_scheme[1]) ) % Blk_scheme[2];
	  
  //The t-coord in the block
  ih -= element[2]*Blk_scheme[0]*Blk_scheme[1];
  element[3] = ( ih/(Blk_scheme[0]*Blk_scheme[1]*Blk_scheme[2]) ) % Blk_scheme[3];

  //Loop over local lattice sites
  for(int x = 0 ; x < GK_localL[0] ; x++)
  for(int y = 0 ; y < GK_localL[1] ; y++)
  for(int z = 0 ; z < GK_localL[2] ; z++)
  for(int t = 0 ; t < GK_localL[3] ; t++){
	  
    // This assumes (for the moment) that the
    // blocking scheme borders and the local 
    // volume borders always superimpose.
    //
    // I.e. GK_localVolume_X / Block_scheme_X = whole number.
    //
    // E.g.
    // GK_localVolume = {32,32,16,32} Block_scheme = {2,2,2,2} will WORK
    // and
    // GK_localVolume = {24,24,12,24} Block_scheme = {3,3,3,3} will WORK
    // but
    // GK_localVolume = {32,32,16,32} Block_scheme = {3,3,3,3} will FAIL
    
    if(x%Blk_scheme[0] == element[0] &&
       y%Blk_scheme[1] == element[1] &&
       z%Blk_scheme[2] == element[2] &&
       t%Blk_scheme[3] == element[3]){
      
      //Reconstruct local index
      i = (((t*GK_localL[2] +z)*GK_localL[1] + y)*GK_localL[0] + x);
      
      ((Float*)blocked_vector)[i*12*2+sc*2+0] = ((Float*)input_vector)[i*12*2+sc*2+0];
      ((Float*)blocked_vector)[i*12*2+sc*2+1] = ((Float*)input_vector)[i*12*2+sc*2+1];
    }
  }
}
