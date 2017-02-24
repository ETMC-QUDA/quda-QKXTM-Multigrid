
// The convention is that x runs fastest, then y, then z

int sid  = blockIdx.x*blockDim.x + threadIdx.x;
int locV = blockDim.x * gridDim.x;
int c_stride_spatial = c_stride/c_localL[3];

register FLOAT2 accum1[4][4];
register FLOAT2 accum2[4][4];

int ii = -1;

#define PROP(tex,mu,nu,a,b) ( FETCH_FLOAT2(tex,sid + it*c_stride_spatial + ( (mu*4+nu)*3*3 + a*3 + b ) * c_stride) ) 


if(ip == 0){ // Nucleon to Nucleon
  
  for(int i = 0 ; i < 4 ; i++)
    for(int j = 0 ; j < 4 ; j++){
      accum1[i][j].x = 0.; accum1[i][j].y = 0.;
      accum2[i][j].x = 0.; accum2[i][j].y = 0.;
    }
  
  if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume
    
    for(int gamma = 0 ; gamma < 4 ; gamma++){
      for(int gammap = 0 ; gammap < 4 ; gammap++){
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
	      accum1[gamma][gammap] = accum1[gamma][gammap] +
		factor*PROP(prop2Tex,beta,betap,b,b1)*(PROP(prop1Tex,alpha,alphap,a,a1) * PROP(prop1Tex,gamma,gammap,c,c1) - PROP(prop1Tex,alpha,gammap,a,c1) * PROP(prop1Tex,gamma,alphap,c,a1) );
	      
	      accum2[gamma][gammap] = accum2[gamma][gammap] +
		factor*PROP(prop1Tex,beta,betap,b,b1)*(PROP(prop2Tex,alpha,alphap,a,a1) * PROP(prop2Tex,gamma,gammap,c,c1) - PROP(prop2Tex,alpha,gammap,a,c1) * PROP(prop2Tex,gamma,alphap,c,a1) );
	    }}}}
    }
    
  }//-if sid 
  __syncthreads();  
 }//-if ip==0

 else if(ip == 1){ // Nucleon to Roper
   
   for(int i = 0 ; i < 4 ; i++)
     for(int j = 0 ; j < 4 ; j++){
       accum1[i][j].x = 0.; accum1[i][j].y = 0.;
       accum2[i][j].x = 0.; accum2[i][j].y = 0.;
     }
   
   if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume

     for(int gamma = 0 ; gamma < 4 ; gamma++){
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
	     
	     accum1[gamma][gammap] = accum1[gamma][gammap] -
	       factor*PROP(prop2Tex,beta,betap,b,b1)*(PROP(prop1Tex,alpha,alphap,a,a1) * PROP(prop1Tex,gamma,deltap,c,c1) - PROP(prop1Tex,alpha,deltap,a,c1) * PROP(prop1Tex,gamma,alphap,c,a1) );
	     
	     accum2[gamma][gammap] = accum2[gamma][gammap] -
	       factor*PROP(prop1Tex,beta,betap,b,b1)*(PROP(prop2Tex,alpha,alphap,a,a1) * PROP(prop2Tex,gamma,deltap,c,c1) - PROP(prop2Tex,alpha,deltap,a,c1) * PROP(prop2Tex,gamma,alphap,c,a1) );
	   }}}
     }
     
   }//-if sid
   __syncthreads();
 }//-if ip==1

 else if(ip == 2){ // Roper to Nucleon

   for(int i = 0 ; i < 4 ; i++)
     for(int j = 0 ; j < 4 ; j++){
       accum1[i][j].x = 0.; accum1[i][j].y = 0.;
       accum2[i][j].x = 0.; accum2[i][j].y = 0.;
     }
   
   if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume

     for(int gammap = 0 ; gammap < 4 ; gammap++){
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

	     accum1[gamma][gammap] = accum1[gamma][gammap] +
	       factor*PROP(prop2Tex,beta,betap,b,b1)*(PROP(prop1Tex,alpha,alphap,a,a1) * PROP(prop1Tex,delta,gammap,c,c1) - PROP(prop1Tex,alpha,gammap,a,c1) * PROP(prop1Tex,delta,alphap,c,a1) );

	     accum2[gamma][gammap] = accum2[gamma][gammap] +
	       factor*PROP(prop1Tex,beta,betap,b,b1)*(PROP(prop2Tex,alpha,alphap,a,a1) * PROP(prop2Tex,delta,gammap,c,c1) - PROP(prop2Tex,alpha,gammap,a,c1) * PROP(prop2Tex,delta,alphap,c,a1) );
	   }}}
     }
   }//-if sid
   __syncthreads();
 }//-if ip==2

 else if(ip==3){ // Roper to Roper
   
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

	   accum1[gamma][gammap] = accum1[gamma][gammap] -
	     factor*PROP(prop2Tex,beta,betap,b,b1)*(PROP(prop1Tex,alpha,alphap,a,a1) * PROP(prop1Tex,delta,deltap,c,c1) - PROP(prop1Tex,alpha,deltap,a,c1) * PROP(prop1Tex,delta,alphap,c,a1) );
	      
	   accum2[gamma][gammap] = accum2[gamma][gammap] -
	     factor*PROP(prop1Tex,beta,betap,b,b1)*(PROP(prop2Tex,alpha,alphap,a,a1) * PROP(prop2Tex,delta,deltap,c,c1) - PROP(prop2Tex,alpha,deltap,a,c1) * PROP(prop2Tex,delta,alphap,c,a1) );
	 }}
     }
   }//-if sid
   __syncthreads();
 }//-if ip==3

 else if((ip==4) || (ip==5) || (ip==6)){ // Deltapp and deltamm, 11,22,33
   
   ii = ip-4;
   for(int i = 0 ; i < 4 ; i++)
     for(int j = 0 ; j < 4 ; j++){
       accum1[i][j].x = 0.; accum1[i][j].y = 0.;
       accum2[i][j].x = 0.; accum2[i][j].y = 0.;
     }
   
   if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume

     for(int gamma = 0 ; gamma < 4 ; gamma++){
       for(int gammap = 0 ; gammap < 4 ; gammap++){
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
	     }}}}
     }
   }//-if sid
   __syncthreads();
 }//if ip==4,5,6

 else if((ip==7) || (ip==8) || (ip==9)){  // Deltap and deltaz, 11,22,33
   
   ii = ip - 7;
   for(int i = 0 ; i < 4 ; i++)
     for(int j = 0 ; j < 4 ; j++){
       accum1[i][j].x = 0.; accum1[i][j].y = 0.;
       accum2[i][j].x = 0.; accum2[i][j].y = 0.;
     }
   
   if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume

     for(int gamma = 0 ; gamma < 4 ; gamma++){
       for(int gammap = 0 ; gammap < 4 ; gammap++){
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
	     }}}}
     }
   }//-if sid
   __syncthreads();   
 }//-if ip==7,8,9


if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume

  for(int ga=0;ga<4;ga++){
    for(int gap=0;gap<4;gap++){
      block[sid + locV*gap + locV*4*ga + locV*4*4*0 ] = accum1[ga][gap];
      block[sid + locV*gap + locV*4*ga + locV*4*4*1 ] = accum2[ga][gap];
    }
  }
 }
__syncthreads();
