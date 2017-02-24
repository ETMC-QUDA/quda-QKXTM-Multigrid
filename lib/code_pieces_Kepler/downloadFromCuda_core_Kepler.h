int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads/2) return;

// take indices on 4d lattice


int half_stride = c_stride/2;
int evenSiteBit;
int latt_coord = 2*sid;


int r1,r2,x_id,y_id,z_id,t_id;

r1 = latt_coord/(c_localL[0]);
r2 = r1/(c_localL[1]);
x_id = latt_coord - r1*(c_localL[0]);
y_id = r1 - r2*(c_localL[1]);
t_id = r2/(c_localL[2]);
z_id = r2 - t_id*(c_localL[2]);
evenSiteBit = ((x_id+y_id+z_id+t_id) & 1);

int oddSiteBit  = evenSiteBit ^ 1;


for(int mu = 0 ; mu < c_nSpin ; mu++)
  for(int ic = 0 ; ic < c_nColor ; ic++){

    if(inEven != NULL)out[mu*c_nColor*c_stride + ic*c_stride + latt_coord + evenSiteBit].x = inEven[mu*c_nColor*half_stride + ic*half_stride + sid].x;
    else out[mu*c_nColor*c_stride + ic*c_stride + latt_coord + evenSiteBit].x = 0.;

    if(inOdd != NULL)out[mu*c_nColor*c_stride + ic*c_stride + latt_coord + oddSiteBit].x = inOdd[mu*c_nColor*half_stride + ic*half_stride + sid].x;
    else out[mu*c_nColor*c_stride + ic*c_stride + latt_coord + oddSiteBit].x = 0.;

    if(inEven != NULL)out[mu*c_nColor*c_stride + ic*c_stride + latt_coord + evenSiteBit].y = inEven[mu*c_nColor*half_stride + ic*half_stride + sid].y;
    else out[mu*c_nColor*c_stride + ic*c_stride + latt_coord + evenSiteBit].y = 0.;

    if(inOdd != NULL)out[mu*c_nColor*c_stride + ic*c_stride + latt_coord + oddSiteBit].y = inOdd[mu*c_nColor*half_stride + ic*half_stride + sid].y;
    else out[mu*c_nColor*c_stride + ic*c_stride + latt_coord + oddSiteBit].y = 0.;

  }
