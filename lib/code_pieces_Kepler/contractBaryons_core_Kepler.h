
int sid = blockIdx.x*blockDim.x + threadIdx.x;
int cacheIndex = threadIdx.x;
__shared__ FLOAT2 shared_cache[2*4*4*THREADS_PER_BLOCK];

int c_stride_spatial = c_stride/c_localL[3];

register FLOAT2 accum1[4][4];
register FLOAT2 accum2[4][4];
int x_id, y_id , z_id;
int r1,r2;

r1 = sid / c_localL[0];
x_id = sid - r1 * c_localL[0];
r2 = r1 / c_localL[1];
y_id = r1 - r2*c_localL[1];
z_id = r2;

int x,y,z;

x = x_id + c_procPosition[0] * c_localL[0] - x0;
y = y_id + c_procPosition[1] * c_localL[1] - y0;
z = z_id + c_procPosition[2] * c_localL[2] - z0;

FLOAT phase;
FLOAT2 expon;
int i;
int ii = -1;
//register FLOAT2 prop1_1,prop1_2,prop1_3,prop1_4,prop1_5;
//register FLOAT2 prop2_1,prop2_2,prop2_3,prop2_4,prop2_5;

/*
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  prop1[mu][nu][c1][c2] = FETCH_FLOAT2(prop1Tex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride_spatial);
	  prop2[mu][nu][c1][c2] = FETCH_FLOAT2(prop2Tex,sid + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride_spatial);
	}
*/
#define PROP(tex,mu,nu,a,b) ( FETCH_FLOAT2(tex,sid + it*c_stride_spatial + ( (mu*4+nu)*3*3 + a*3 + b ) * c_stride) ) 
if(ip == 0){
  /////////////////////////////////////// NTN //////////////////////////////////////
  for(int i = 0 ; i < 4 ; i++)
    for(int j = 0 ; j < 4 ; j++){
      accum1[i][j].x = 0.; accum1[i][j].y = 0.;
      accum2[i][j].x = 0.; accum2[i][j].y = 0.;
    }

  if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume

    for(int gamma = 0 ; gamma < 4 ; gamma++)
      for(int gammap = 0 ; gammap < 4 ; gammap++)
	for(int idx = 0 ; idx < 16 ; idx++){
	  short int alpha = c_NTN_indices[idx][0];
	  short int beta = c_NTN_indices[idx][1];
	  short int betap = c_NTN_indices[idx][2];
	  short int alphap = c_NTN_indices[idx][3];
	  for(int cc1 = 0 ; cc1 < 6 ; cc1++){
	    short int a = c_eps[cc1][0];
	    short int b = c_eps[cc1][1];
	    short int c = c_eps[cc1][2];
	    for(int cc2 = 0 ; cc2 < 6 ; cc2++){
	      short int a1 = c_eps[cc2][0];
	      short int b1 = c_eps[cc2][1];
	      short int c1 = c_eps[cc2][2];
	      FLOAT factor = c_sgn_eps[cc1] * c_sgn_eps[cc2] * c_NTN_values[idx];
	      accum1[gamma][gammap] = accum1[gamma][gammap] + factor*PROP(prop2Tex,beta,betap,b,b1)*(PROP(prop1Tex,alpha,alphap,a,a1) * PROP(prop1Tex,gamma,gammap,c,c1) - PROP(prop1Tex,alpha,gammap,a,c1) * PROP(prop1Tex,gamma,alphap,c,a1) );
	      accum2[gamma][gammap] = accum2[gamma][gammap] + factor*PROP(prop1Tex,beta,betap,b,b1)*(PROP(prop2Tex,alpha,alphap,a,a1) * PROP(prop2Tex,gamma,gammap,c,c1) - PROP(prop2Tex,alpha,gammap,a,c1) * PROP(prop2Tex,gamma,alphap,c,a1));
	    }
	  }
	}

  }
  __syncthreads();
  for(int imom = 0 ; imom < c_Nmoms ; imom++){
    phase = ( ((FLOAT) c_moms[imom][0]*x)/c_totalL[0] + ((FLOAT) c_moms[imom][1]*y)/c_totalL[1] + ((FLOAT) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
    expon.x = cos(phase);
    expon.y = -sin(phase);
    for(int gamma = 0 ; gamma < 4 ; gamma++)
      for(int gammap = 0 ; gammap < 4 ; gammap++){
	shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum1[gamma][gammap] * expon; 
	shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum2[gamma][gammap] * expon;
      }
    __syncthreads();
    i = blockDim.x/2;
    while (i != 0){
      if(cacheIndex < i){
	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int gammap = 0 ; gammap < 4 ; gammap++){
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	    
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	  }
      }
      __syncthreads();
      i /= 2;
    }
    
    if(cacheIndex == 0){
      for(int gamma = 0 ; gamma < 4 ; gamma++)
	for(int gammap = 0 ; gammap < 4 ; gammap++){
	  block[imom*2*4*4*gridDim.x + 0*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	  block[imom*2*4*4*gridDim.x + 1*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	}
    }
  } // close momentum
  
 }

 else if(ip == 1){
   
   /////////////////////////////////////// NTR //////////////////////////////////////
   for(int i = 0 ; i < 4 ; i++)
     for(int j = 0 ; j < 4 ; j++){
       accum1[i][j].x = 0.; accum1[i][j].y = 0.;
       accum2[i][j].x = 0.; accum2[i][j].y = 0.;
     }
   
   if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume
     for(int gamma = 0 ; gamma < 4 ; gamma++)
       for(int idx = 0 ; idx < 64 ; idx++){
       short int alpha = c_NTR_indices[idx][0];
       short int beta = c_NTR_indices[idx][1];
       short int betap = c_NTR_indices[idx][2];
       short int alphap = c_NTR_indices[idx][3];
       short int gammap = c_NTR_indices[idx][4];
       short int deltap = c_NTR_indices[idx][5];
	 for(int cc1 = 0 ; cc1 < 6 ; cc1++){
	   short int a = c_eps[cc1][0];
	   short int b = c_eps[cc1][1];
	   short int c = c_eps[cc1][2];
	   for(int cc2 = 0 ; cc2 < 6 ; cc2++){
	     short int a1 = c_eps[cc2][0];
	     short int b1 = c_eps[cc2][1];
	     short int c1 = c_eps[cc2][2];
	     FLOAT factor = c_sgn_eps[cc1] * c_sgn_eps[cc2] * c_NTR_values[idx];

	      accum1[gamma][gammap] = accum1[gamma][gammap] - factor*PROP(prop2Tex,beta,betap,b,b1)*(PROP(prop1Tex,alpha,alphap,a,a1) * PROP(prop1Tex,gamma,deltap,c,c1) - PROP(prop1Tex,alpha,deltap,a,c1) * PROP(prop1Tex,gamma,alphap,c,a1) );
	      accum2[gamma][gammap] = accum2[gamma][gammap] - factor*PROP(prop1Tex,beta,betap,b,b1)*(PROP(prop2Tex,alpha,alphap,a,a1) * PROP(prop2Tex,gamma,deltap,c,c1) - PROP(prop2Tex,alpha,deltap,a,c1) * PROP(prop2Tex,gamma,alphap,c,a1));
	   }
	 }
       }
   }

  __syncthreads();
  for(int imom = 0 ; imom < c_Nmoms ; imom++){
    phase = ( ((FLOAT) c_moms[imom][0]*x)/c_totalL[0] + ((FLOAT) c_moms[imom][1]*y)/c_totalL[1] + ((FLOAT) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
    expon.x = cos(phase);
    expon.y = -sin(phase);
    for(int gamma = 0 ; gamma < 4 ; gamma++)
      for(int gammap = 0 ; gammap < 4 ; gammap++){
	shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum1[gamma][gammap] * expon; 
	shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum2[gamma][gammap] * expon;
      }
    __syncthreads();
    i = blockDim.x/2;
    while (i != 0){
      if(cacheIndex < i){
	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int gammap = 0 ; gammap < 4 ; gammap++){
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	    
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	  }
      }
      __syncthreads();
      i /= 2;
    }
    
    if(cacheIndex == 0){
      for(int gamma = 0 ; gamma < 4 ; gamma++)
	for(int gammap = 0 ; gammap < 4 ; gammap++){
	  block[imom*2*4*4*gridDim.x + 0*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	  block[imom*2*4*4*gridDim.x + 1*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	}
    }
  } // close momentum

 }
 else if(ip == 2){
   /////////////////////////////////////// RTN //////////////////////////////////////
   for(int i = 0 ; i < 4 ; i++)
     for(int j = 0 ; j < 4 ; j++){
       accum1[i][j].x = 0.; accum1[i][j].y = 0.;
       accum2[i][j].x = 0.; accum2[i][j].y = 0.;
     }
   
   if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume
     for(int gammap = 0 ; gammap < 4 ; gammap++)
       for(int idx = 0 ; idx < 64 ; idx++){
	 short int alpha = c_RTN_indices[idx][0];
	 short int beta = c_RTN_indices[idx][1];
	 short int betap = c_RTN_indices[idx][2];
	 short int alphap = c_RTN_indices[idx][3];
	 short int gamma = c_RTN_indices[idx][4];
	 short int delta = c_RTN_indices[idx][5];
	 for(int cc1 = 0 ; cc1 < 6 ; cc1++){
	   short int a = c_eps[cc1][0];
	   short int b = c_eps[cc1][1];
	   short int c = c_eps[cc1][2];
	   for(int cc2 = 0 ; cc2 < 6 ; cc2++){
	     short int a1 = c_eps[cc2][0];
	     short int b1 = c_eps[cc2][1];
	     short int c1 = c_eps[cc2][2];
	     FLOAT factor = c_sgn_eps[cc1] * c_sgn_eps[cc2] * c_RTN_values[idx];

	      accum1[gamma][gammap] = accum1[gamma][gammap] + factor*PROP(prop2Tex,beta,betap,b,b1)*(PROP(prop1Tex,alpha,alphap,a,a1) * PROP(prop1Tex,delta,gammap,c,c1) - PROP(prop1Tex,alpha,gammap,a,c1) * PROP(prop1Tex,delta,alphap,c,a1) );
	      accum2[gamma][gammap] = accum2[gamma][gammap] + factor*PROP(prop1Tex,beta,betap,b,b1)*(PROP(prop2Tex,alpha,alphap,a,a1) * PROP(prop2Tex,delta,gammap,c,c1) - PROP(prop2Tex,alpha,gammap,a,c1) * PROP(prop2Tex,delta,alphap,c,a1));

	   }
	 }
       }
   }
  __syncthreads();
  for(int imom = 0 ; imom < c_Nmoms ; imom++){
    phase = ( ((FLOAT) c_moms[imom][0]*x)/c_totalL[0] + ((FLOAT) c_moms[imom][1]*y)/c_totalL[1] + ((FLOAT) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
    expon.x = cos(phase);
    expon.y = -sin(phase);
    for(int gamma = 0 ; gamma < 4 ; gamma++)
      for(int gammap = 0 ; gammap < 4 ; gammap++){
	shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum1[gamma][gammap] * expon; 
	shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum2[gamma][gammap] * expon;
      }
    __syncthreads();
    i = blockDim.x/2;
    while (i != 0){
      if(cacheIndex < i){
	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int gammap = 0 ; gammap < 4 ; gammap++){
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	    
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	  }
      }
      __syncthreads();
      i /= 2;
    }
    
    if(cacheIndex == 0){
      for(int gamma = 0 ; gamma < 4 ; gamma++)
	for(int gammap = 0 ; gammap < 4 ; gammap++){
	  block[imom*2*4*4*gridDim.x + 0*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	  block[imom*2*4*4*gridDim.x + 1*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	}
    }
  } // close momentum

 }
 else if(ip==3){
   
   /////////////////////////////////////// RTR //////////////////////////////////////
   for(int i = 0 ; i < 4 ; i++)
     for(int j = 0 ; j < 4 ; j++){
       accum1[i][j].x = 0.; accum1[i][j].y = 0.;
       accum2[i][j].x = 0.; accum2[i][j].y = 0.;
     }
   
   if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume
     
     for(int idx = 0 ; idx < 256 ; idx++){
       short int alpha = c_RTR_indices[idx][0];
       short int beta = c_RTR_indices[idx][1];
       short int betap = c_RTR_indices[idx][2];
       short int alphap = c_RTR_indices[idx][3];
       short int gamma = c_RTR_indices[idx][4];
       short int delta = c_RTR_indices[idx][5];
       short int gammap = c_RTR_indices[idx][6];
       short int deltap = c_RTR_indices[idx][7];
       for(int cc1 = 0 ; cc1 < 6 ; cc1++){
	 short int a = c_eps[cc1][0];
	 short int b = c_eps[cc1][1];
	 short int c = c_eps[cc1][2];
	 for(int cc2 = 0 ; cc2 < 6 ; cc2++){
	   short int a1 = c_eps[cc2][0];
	   short int b1 = c_eps[cc2][1];
	   short int c1 = c_eps[cc2][2];
	   FLOAT factor = c_sgn_eps[cc1] * c_sgn_eps[cc2] * c_RTR_values[idx];

	      accum1[gamma][gammap] = accum1[gamma][gammap] - factor*PROP(prop2Tex,beta,betap,b,b1)*(PROP(prop1Tex,alpha,alphap,a,a1) * PROP(prop1Tex,delta,deltap,c,c1) - PROP(prop1Tex,alpha,deltap,a,c1) * PROP(prop1Tex,delta,alphap,c,a1) );
	      accum2[gamma][gammap] = accum2[gamma][gammap] - factor*PROP(prop1Tex,beta,betap,b,b1)*(PROP(prop2Tex,alpha,alphap,a,a1) * PROP(prop2Tex,delta,deltap,c,c1) - PROP(prop2Tex,alpha,deltap,a,c1) * PROP(prop2Tex,delta,alphap,c,a1));

	 }
       }
     }
   }

  __syncthreads();
  for(int imom = 0 ; imom < c_Nmoms ; imom++){
    phase = ( ((FLOAT) c_moms[imom][0]*x)/c_totalL[0] + ((FLOAT) c_moms[imom][1]*y)/c_totalL[1] + ((FLOAT) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
    expon.x = cos(phase);
    expon.y = -sin(phase);
    for(int gamma = 0 ; gamma < 4 ; gamma++)
      for(int gammap = 0 ; gammap < 4 ; gammap++){
	shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum1[gamma][gammap] * expon; 
	shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum2[gamma][gammap] * expon;
      }
    __syncthreads();
    i = blockDim.x/2;
    while (i != 0){
      if(cacheIndex < i){
	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int gammap = 0 ; gammap < 4 ; gammap++){
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	    
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	  }
      }
      __syncthreads();
      i /= 2;
    }
    
    if(cacheIndex == 0){
      for(int gamma = 0 ; gamma < 4 ; gamma++)
	for(int gammap = 0 ; gammap < 4 ; gammap++){
	  block[imom*2*4*4*gridDim.x + 0*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	  block[imom*2*4*4*gridDim.x + 1*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	}
    }
  } // close momentum
   
   ///////////////////////////////////////////////////////
 }

 else if((ip==4) || (ip==5) || (ip==6)){
   
   /////////////////////////////////////// DELTAISO1 //////////////////////////////////////
   ii = ip-4;
   for(int i = 0 ; i < 4 ; i++)
     for(int j = 0 ; j < 4 ; j++){
       accum1[i][j].x = 0.; accum1[i][j].y = 0.;
       accum2[i][j].x = 0.; accum2[i][j].y = 0.;
     }
   
   if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume
     for(int gamma = 0 ; gamma < 4 ; gamma++)
       for(int gammap = 0 ; gammap < 4 ; gammap++)
	 for(int idx = 0 ; idx < 16 ; idx++){
	   short int alpha = c_Delta_indices[ii][idx][0];
	   short int beta = c_Delta_indices[ii][idx][1];
	   short int betap = c_Delta_indices[ii][idx][2];
	   short int alphap = c_Delta_indices[ii][idx][3];
	   for(int cc1 = 0 ; cc1 < 6 ; cc1++){
	     short int a = c_eps[cc1][0];
	     short int b = c_eps[cc1][1];
	     short int c = c_eps[cc1][2];
	     for(int cc2 = 0 ; cc2 < 6 ; cc2++){
	       short int a1 = c_eps[cc2][0];
	       short int b1 = c_eps[cc2][1];
	       short int c1 = c_eps[cc2][2];
	       FLOAT factor = c_sgn_eps[cc1] * c_sgn_eps[cc2] * c_Delta_values[ii][idx];
	       accum1[gamma][gammap] = accum1[gamma][gammap] + factor*(
								       PROP(prop1Tex,alpha,betap,a,b1)*PROP(prop1Tex,beta,gammap,b,c1)*PROP(prop1Tex,gamma,alphap,c,a1)
								       -PROP(prop1Tex,alpha,gammap,a,c1)*PROP(prop1Tex,beta,betap,b,b1)*PROP(prop1Tex,gamma,alphap,c,a1)
								       +PROP(prop1Tex,alpha,gammap,a,c1)*PROP(prop1Tex,beta,alphap,b,a1)*PROP(prop1Tex,gamma,betap,c,b1)
								       -PROP(prop1Tex,alpha,alphap,a,a1)*PROP(prop1Tex,beta,gammap,b,c1)*PROP(prop1Tex,gamma,betap,c,b1)
								       -PROP(prop1Tex,alpha,betap,a,b1)*PROP(prop1Tex,beta,alphap,b,a1)*PROP(prop1Tex,gamma,gammap,c,c1)
								       +PROP(prop1Tex,alpha,alphap,a,a1)*PROP(prop1Tex,beta,betap,b,b1)*PROP(prop1Tex,gamma,gammap,c,c1) );
	       
	       accum2[gamma][gammap] = accum2[gamma][gammap] + factor*(
								       PROP(prop2Tex,alpha,betap,a,b1)*PROP(prop2Tex,beta,gammap,b,c1)*PROP(prop2Tex,gamma,alphap,c,a1)
								       -PROP(prop2Tex,alpha,gammap,a,c1)*PROP(prop2Tex,beta,betap,b,b1)*PROP(prop2Tex,gamma,alphap,c,a1)                           
								       +PROP(prop2Tex,alpha,gammap,a,c1)*PROP(prop2Tex,beta,alphap,b,a1)*PROP(prop2Tex,gamma,betap,c,b1)
								       -PROP(prop2Tex,alpha,alphap,a,a1)*PROP(prop2Tex,beta,gammap,b,c1)*PROP(prop2Tex,gamma,betap,c,b1)
								       -PROP(prop2Tex,alpha,betap,a,b1)*PROP(prop2Tex,beta,alphap,b,a1)*PROP(prop2Tex,gamma,gammap,c,c1)
								       +PROP(prop2Tex,alpha,alphap,a,a1)*PROP(prop2Tex,beta,betap,b,b1)*PROP(prop2Tex,gamma,gammap,c,c1) );
	       
	     }
	   }
	 }
   }

  __syncthreads();
  for(int imom = 0 ; imom < c_Nmoms ; imom++){
    phase = ( ((FLOAT) c_moms[imom][0]*x)/c_totalL[0] + ((FLOAT) c_moms[imom][1]*y)/c_totalL[1] + ((FLOAT) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
    expon.x = cos(phase);
    expon.y = -sin(phase);
    for(int gamma = 0 ; gamma < 4 ; gamma++)
      for(int gammap = 0 ; gammap < 4 ; gammap++){
	shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum1[gamma][gammap] * expon; 
	shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum2[gamma][gammap] * expon;
      }
    __syncthreads();
    i = blockDim.x/2;
    while (i != 0){
      if(cacheIndex < i){
	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int gammap = 0 ; gammap < 4 ; gammap++){
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	    
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	  }
      }
      __syncthreads();
      i /= 2;
    }
    
    if(cacheIndex == 0){
      for(int gamma = 0 ; gamma < 4 ; gamma++)
	for(int gammap = 0 ; gammap < 4 ; gammap++){
	  block[imom*2*4*4*gridDim.x + 0*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	  block[imom*2*4*4*gridDim.x + 1*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	}
    }
  } // close momentum

 }

 else if((ip==7) || (ip==8) || (ip==9)){
   /////////////////////////////////////// DELTAISO1O2 //////////////////////////////////////
   ii = ip - 7;
   for(int i = 0 ; i < 4 ; i++)
     for(int j = 0 ; j < 4 ; j++){
       accum1[i][j].x = 0.; accum1[i][j].y = 0.;
       accum2[i][j].x = 0.; accum2[i][j].y = 0.;
     }
   
   if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume
     for(int gamma = 0 ; gamma < 4 ; gamma++)
       for(int gammap = 0 ; gammap < 4 ; gammap++)
	 for(int idx = 0 ; idx < 16 ; idx++){
	   short int alpha = c_Delta_indices[ii][idx][0];
	   short int beta = c_Delta_indices[ii][idx][1];
	   short int betap = c_Delta_indices[ii][idx][2];
	   short int alphap = c_Delta_indices[ii][idx][3];
	   for(int cc1 = 0 ; cc1 < 6 ; cc1++){
	     short int a = c_eps[cc1][0];
	     short int b = c_eps[cc1][1];
	     short int c = c_eps[cc1][2];
	     for(int cc2 = 0 ; cc2 < 6 ; cc2++){
	       short int a1 = c_eps[cc2][0];
	       short int b1 = c_eps[cc2][1];
	       short int c1 = c_eps[cc2][2];
	       FLOAT factor = c_sgn_eps[cc1] * c_sgn_eps[cc2] * c_Delta_values[ii][idx];
	       accum1[gamma][gammap] = accum1[gamma][gammap] + (1./3.)*factor*(
								       -4*PROP(prop1Tex,alpha,gammap,a,c1)*PROP(prop2Tex,beta,betap,b,b1)*PROP(prop1Tex,gamma,alphap,c,a1)
								       +2*PROP(prop1Tex,alpha,betap,a,b1)*PROP(prop2Tex,beta,gammap,b,c1)*PROP(prop1Tex,gamma,alphap,c,a1)
								       +2*PROP(prop1Tex,alpha,gammap,a,c1)*PROP(prop1Tex,beta,alphap,b,a1)*PROP(prop2Tex,gamma,betap,c,b1)
								       -2*PROP(prop1Tex,alpha,alphap,a,a1)*PROP(prop1Tex,beta,gammap,b,c1)*PROP(prop2Tex,gamma,betap,c,b1)
								       -2*PROP(prop1Tex,alpha,alphap,a,a1)*PROP(prop2Tex,beta,gammap,b,c1)*PROP(prop1Tex,gamma,betap,c,b1)
								       -PROP(prop1Tex,alpha,betap,a,b1)*PROP(prop1Tex,beta,alphap,b,a1)*PROP(prop2Tex,gamma,gammap,c,c1)
								       +PROP(prop1Tex,alpha,alphap,a,a1)*PROP(prop1Tex,beta,betap,b,b1)*PROP(prop2Tex,gamma,gammap,c,c1)
								       +4*PROP(prop1Tex,alpha,alphap,a,a1)*PROP(prop2Tex,beta,betap,b,b1)*PROP(prop1Tex,gamma,gammap,c,c1));
	       
	       accum2[gamma][gammap] = accum2[gamma][gammap] + (1./3.)*factor*(
								       -4*PROP(prop2Tex,alpha,gammap,a,c1)*PROP(prop1Tex,beta,betap,b,b1)*PROP(prop2Tex,gamma,alphap,c,a1)
								       +2*PROP(prop2Tex,alpha,betap,a,b1)*PROP(prop1Tex,beta,gammap,b,c1)*PROP(prop2Tex,gamma,alphap,c,a1)
								       +2*PROP(prop2Tex,alpha,gammap,a,c1)*PROP(prop2Tex,beta,alphap,b,a1)*PROP(prop1Tex,gamma,betap,c,b1)
								       -2*PROP(prop2Tex,alpha,alphap,a,a1)*PROP(prop2Tex,beta,gammap,b,c1)*PROP(prop1Tex,gamma,betap,c,b1)
								       -2*PROP(prop2Tex,alpha,alphap,a,a1)*PROP(prop1Tex,beta,gammap,b,c1)*PROP(prop2Tex,gamma,betap,c,b1)
								       -PROP(prop2Tex,alpha,betap,a,b1)*PROP(prop2Tex,beta,alphap,b,a1)*PROP(prop1Tex,gamma,gammap,c,c1)
								       +PROP(prop2Tex,alpha,alphap,a,a1)*PROP(prop2Tex,beta,betap,b,b1)*PROP(prop1Tex,gamma,gammap,c,c1)
								       +4*PROP(prop2Tex,alpha,alphap,a,a1)*PROP(prop1Tex,beta,betap,b,b1)*PROP(prop2Tex,gamma,gammap,c,c1));
	     }
	   }
	 }
   }

  __syncthreads();
  for(int imom = 0 ; imom < c_Nmoms ; imom++){
    phase = ( ((FLOAT) c_moms[imom][0]*x)/c_totalL[0] + ((FLOAT) c_moms[imom][1]*y)/c_totalL[1] + ((FLOAT) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
    expon.x = cos(phase);
    expon.y = -sin(phase);
    for(int gamma = 0 ; gamma < 4 ; gamma++)
      for(int gammap = 0 ; gammap < 4 ; gammap++){
	shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum1[gamma][gammap] * expon; 
	shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex] = accum2[gamma][gammap] * expon;
      }
    __syncthreads();
    i = blockDim.x/2;
    while (i != 0){
      if(cacheIndex < i){
	for(int gamma = 0 ; gamma < 4 ; gamma++)
	  for(int gammap = 0 ; gammap < 4 ; gammap++){
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	    
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].x;
	    shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + i].y;
	  }
      }
      __syncthreads();
      i /= 2;
    }
    
    if(cacheIndex == 0){
      for(int gamma = 0 ; gamma < 4 ; gamma++)
	for(int gammap = 0 ; gammap < 4 ; gammap++){
	  block[imom*2*4*4*gridDim.x + 0*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[0*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	  block[imom*2*4*4*gridDim.x + 1*4*4*gridDim.x + gamma*4*gridDim.x + gammap*gridDim.x + blockIdx.x] = shared_cache[1*4*4*THREADS_PER_BLOCK + gamma*4*THREADS_PER_BLOCK + gammap*THREADS_PER_BLOCK + cacheIndex + 0];
	}
    }
  } // close momentum
   
 }

