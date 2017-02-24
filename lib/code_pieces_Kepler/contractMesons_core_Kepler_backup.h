#include <g5_prop_G_dag_g5.h>
int sid = blockIdx.x*blockDim.x + threadIdx.x;
int cacheIndex = threadIdx.x;
__shared__ FLOAT2 shared_cache[2*10*THREADS_PER_BLOCK];

int c_stride_spatial = c_stride/c_localL[3];

FLOAT2 prop[4][4][3][3];
FLOAT2 g5propDagG5[4][4][3][3];
FLOAT2 accum1[10];
FLOAT2 accum2[10];

for(int i = 0 ; i < 10 ; i++){
  accum1[i].x = 0.; accum1[i].y = 0.;
  accum2[i].x = 0.; accum2[i].y = 0.;
 }

if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume

  /////////////////////////////////////// using prop1 //////////////////////////////////////

  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  prop[mu][nu][c1][c2] = FETCH_FLOAT2(prop1Tex,sid + it*c_stride_spatial + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
	}



  g5_prop_G_dag_g5(g5propDagG5,prop);
  for(int ip = 0 ; ip < 10 ; ip++)
    for(int is = 0 ; is < 16 ; is++){
      short int beta = c_mesons_indices[ip][is][0];
      short int gamma = c_mesons_indices[ip][is][1];
      short int delta = c_mesons_indices[ip][is][2];
      short int alpha = c_mesons_indices[ip][is][3];
      float value = c_mesons_values[ip][is];
      for(int a = 0 ; a < 3 ; a++)
	for(int b = 0 ; b < 3 ; b++)
	  accum1[ip] = accum1[ip] + value * prop[alpha][beta][a][b] * g5propDagG5[gamma][delta][b][a];
    }
  
  /////////////////////////////////////// using prop1 //////////////////////////////////////


  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++)
      for(int c1 = 0 ; c1 < 3 ; c1++)
	for(int c2 = 0 ; c2 < 3 ; c2++){
	  prop[mu][nu][c1][c2] = FETCH_FLOAT2(prop2Tex,sid + it*c_stride_spatial + ( (mu*4+nu)*3*3 + c1*3 + c2 ) * c_stride);
	}

  g5_prop_G_dag_g5(g5propDagG5,prop);
  
  for(int ip = 0 ; ip < 10 ; ip++)
    for(int is = 0 ; is < 16 ; is++){
      short int beta = c_mesons_indices[ip][is][0];
      short int gamma = c_mesons_indices[ip][is][1];
      short int delta = c_mesons_indices[ip][is][2];
      short int alpha = c_mesons_indices[ip][is][3];
      float value = c_mesons_values[ip][is];
      for(int a = 0 ; a < 3 ; a++)
	for(int b = 0 ; b < 3 ; b++)
	  accum2[ip] = accum2[ip] + value * prop[alpha][beta][a][b] * g5propDagG5[gamma][delta][b][a];
    }

  
 }
//////////////////////////////////////////////////////////////////////////////////////

__syncthreads();

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

//x=0;
//y=0;
//z=0;

FLOAT phase;
FLOAT2 expon;
int i;
for(int imom = 0 ; imom < c_Nmoms ; imom++){
  phase = ( ((FLOAT) c_moms[imom][0]*x)/c_totalL[0] + ((FLOAT) c_moms[imom][1]*y)/c_totalL[1] + ((FLOAT) c_moms[imom][2]*z)/c_totalL[2] ) * 2. * PI;
  expon.x = cos(phase);
  expon.y = -sin(phase);
  for(int ip = 0 ; ip < 10 ; ip++){
    shared_cache[0*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + cacheIndex] = accum1[ip] * expon; 
    shared_cache[1*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + cacheIndex] = accum2[ip] * expon;
  }
  __syncthreads();
  i = blockDim.x/2;
  while (i != 0){
    if(cacheIndex < i){
      for(int ip = 0 ; ip < 10 ; ip++){
	shared_cache[0*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[0*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + cacheIndex + i].x;
	shared_cache[0*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[0*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + cacheIndex + i].y;

	shared_cache[1*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + cacheIndex].x += shared_cache[1*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + cacheIndex + i].x;
	shared_cache[1*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + cacheIndex].y += shared_cache[1*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + cacheIndex + i].y;
      }
    }
    __syncthreads();
    i /= 2;
  }

  if(cacheIndex == 0){
    for(int ip = 0 ; ip < 10 ; ip++){
      block[imom*2*10*gridDim.x + 0*10*gridDim.x + ip*gridDim.x + blockIdx.x] = shared_cache[0*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + 0];
      block[imom*2*10*gridDim.x + 1*10*gridDim.x + ip*gridDim.x + blockIdx.x] = shared_cache[1*10*THREADS_PER_BLOCK + ip*THREADS_PER_BLOCK + 0];
    }
  }

 } // close momentum

#undef g5_prop_G_dag_g5
