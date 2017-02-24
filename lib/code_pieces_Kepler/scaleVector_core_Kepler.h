
int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;

inOut[0*c_stride + sid] = a*inOut[0*c_stride + sid];
inOut[1*c_stride + sid] = a*inOut[1*c_stride + sid];
inOut[2*c_stride + sid] = a*inOut[2*c_stride + sid];
inOut[3*c_stride + sid] = a*inOut[3*c_stride + sid];
inOut[4*c_stride + sid] = a*inOut[4*c_stride + sid];
inOut[5*c_stride + sid] = a*inOut[5*c_stride + sid];
inOut[6*c_stride + sid] = a*inOut[6*c_stride + sid];
inOut[7*c_stride + sid] = a*inOut[7*c_stride + sid];
inOut[8*c_stride + sid] = a*inOut[8*c_stride + sid];
inOut[9*c_stride + sid] = a*inOut[9*c_stride + sid];
inOut[10*c_stride + sid] = a*inOut[10*c_stride + sid];
inOut[11*c_stride + sid] = a*inOut[11*c_stride + sid];
