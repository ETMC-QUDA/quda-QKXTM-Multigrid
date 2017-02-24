
int sid = blockIdx.x*blockDim.x + threadIdx.x;
int locV = blockDim.x * gridDim.x;
int c_stride_spatial = c_stride/c_localL[3];

register FLOAT2 accum1[10];
register FLOAT2 accum2[10];

for(int i = 0 ; i < 10 ; i++){
  accum1[i].x = 0.; accum1[i].y = 0.;
  accum2[i].x = 0.; accum2[i].y = 0.;
 }

#define PROP(tex,mu,nu,a,b) ( FETCH_FLOAT2(tex,sid + it*c_stride_spatial + ( (mu*4+nu)*3*3 + a*3 + b ) * c_stride) ) 

if (sid < c_threads/c_localL[3]){ // I work only on the spatial volume
  
  for(int ip = 0 ; ip < 10 ; ip++){
    for(int is = 0 ; is < 16 ; is++){
      short int beta = c_mesons_indices[ip][is][0];
      short int gamma = c_mesons_indices[ip][is][1];
      short int delta = c_mesons_indices[ip][is][2];
      short int alpha = c_mesons_indices[ip][is][3];
      float value = c_mesons_values[ip][is];
      for(int a = 0 ; a < 3 ; a++){
	for(int b = 0 ; b < 3 ; b++){
	  accum1[ip] = accum1[ip] + value *PROP(prop1Tex,alpha,beta,a,b) * conj(PROP(prop1Tex,delta,gamma,a,b));
	  accum2[ip] = accum2[ip] + value *PROP(prop2Tex,alpha,beta,a,b) * conj(PROP(prop2Tex,delta,gamma,a,b));
	}}
    }
    __syncthreads();

    block[sid + locV*ip + locV*10*0 ] = accum1[ip];
    block[sid + locV*ip + locV*10*1 ] = accum2[ip];

    __syncthreads();

  }//-ip
 }//-if sid
