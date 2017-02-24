int sid = blockIdx.x*blockDim.x + threadIdx.x;
if (sid >= c_threads) return;


int x_id, y_id, z_id, t_id;
int r1,r2;

r1 = sid/(c_localL[0]);
r2 = r1/(c_localL[1]);
x_id = sid - r1*(c_localL[0]);
y_id = r1 - r2*(c_localL[1]);
t_id = r2/(c_localL[2]);
z_id = r2 - t_id*(c_localL[2]);

// take forward and backward points index
int pointPlus[4];
int pointMinus[4];
int pointMinusG[4];

pointPlus[0] = LEXIC(t_id,z_id,y_id,(x_id+1)%c_localL[0],c_localL); 
pointPlus[1] = LEXIC(t_id,z_id,(y_id+1)%c_localL[1],x_id,c_localL);
pointPlus[2] = LEXIC(t_id,(z_id+1)%c_localL[2],y_id,x_id,c_localL);
pointPlus[3] = LEXIC((t_id+1)%c_localL[3],z_id,y_id,x_id,c_localL);
pointMinus[0] = LEXIC(t_id,z_id,y_id,(x_id-1+c_localL[0])%c_localL[0],c_localL);
pointMinus[1] = LEXIC(t_id,z_id,(y_id-1+c_localL[1])%c_localL[1],x_id,c_localL);
pointMinus[2] = LEXIC(t_id,(z_id-1+c_localL[2])%c_localL[2],y_id,x_id,c_localL);
pointMinus[3] = LEXIC((t_id-1+c_localL[3])%c_localL[3],z_id,y_id,x_id,c_localL);

pointMinusG[0] = pointMinus[0];
pointMinusG[1] = pointMinus[1];
pointMinusG[2] = pointMinus[2];
pointMinusG[3] = pointMinus[3];

// x direction
if(c_dimBreak[0] == true){
  if(x_id == c_localL[0] -1)
    pointPlus[0] = c_plusGhost[0]*c_nSpin*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);
  if(x_id == 0){
    pointMinus[0] = c_minusGhost[0]*c_nSpin*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);
    pointMinusG[0] = c_minusGhost[0]*c_nDim*c_nColor*c_nColor + LEXIC_TZY(t_id,z_id,y_id,c_localL);
  }
 }
// y direction
if(c_dimBreak[1] == true){
  if(y_id == c_localL[1] -1)
    pointPlus[1] = c_plusGhost[1]*c_nSpin*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);
  if(y_id == 0){
    pointMinus[1] = c_minusGhost[1]*c_nSpin*c_nColor + LEXIC_TZX(t_id,z_id,x_id,c_localL);
    pointMinusG[1] = c_minusGhost[1]*c_nDim*c_nColor*c_nColor +  LEXIC_TZX(t_id,z_id,x_id,c_localL);
  }
 }
// z direction 
if(c_dimBreak[2] == true){
  if(z_id == c_localL[2] -1)
    pointPlus[2] = c_plusGhost[2]*c_nSpin*c_nColor + LEXIC_TYX(t_id,y_id,x_id,c_localL);
  if(z_id == 0){
    pointMinus[2] = c_minusGhost[2]*c_nSpin*c_nColor + LEXIC_TYX(t_id,y_id,x_id,c_localL);
    pointMinusG[2] = c_minusGhost[2]*c_nDim*c_nColor*c_nColor +  LEXIC_TYX(t_id,y_id,x_id,c_localL);
  }

 }
// t direction
if(c_dimBreak[3] == true){
  if(t_id == c_localL[3] -1)
    pointPlus[3] = c_plusGhost[3]*c_nSpin*c_nColor + LEXIC_ZYX(z_id,y_id,x_id,c_localL);
  if(t_id == 0){
    pointMinus[3] = c_minusGhost[3]*c_nSpin*c_nColor + LEXIC_ZYX(z_id,y_id,x_id,c_localL);
    pointMinusG[3] = c_minusGhost[3]*c_nDim*c_nColor*c_nColor + LEXIC_ZYX(z_id,y_id,x_id,c_localL);
  }
 }

FLOAT2 G_0 , G_1 , G_2 , G_3 , G_4 , G_5 , G_6 , G_7 , G_8;
FLOAT2 S_0 , S_1 , S_2 , S_3 , S_4 , S_5 , S_6 , S_7 , S_8 , S_9 , S_10 , S_11;
FLOAT2 P1_0 , P1_1 , P1_2 , P1_3 , P1_4 , P1_5 , P1_6 , P1_7 , P1_8 , P1_9 , P1_10 , P1_11;
FLOAT2 P2_0 , P2_1 , P2_2 , P2_3 , P2_4 , P2_5 , P2_6 , P2_7 , P2_8 , P2_9 , P2_10 , P2_11;
FLOAT2 tmp[12];
/////////////////////////////////// mu = 0 

READGAUGE_FLOAT(G,gaugeTex,0,sid,c_stride);

if(c_dimBreak[0] == true && (x_id == c_localL[0] - 1)){
  READVECTOR_FLOAT(S,vecInTex,pointPlus[0],c_surface[0]);}
 else{
   READVECTOR_FLOAT(S,vecInTex,pointPlus[0],c_stride);}

apply_U_on_S(P1_,G_,S_);

if(c_dimBreak[0] == true && (x_id == 0)){
  READGAUGE_FLOAT(G,gaugeTex,0,pointMinusG[0],c_surface[0]);}
 else{
   READGAUGE_FLOAT(G,gaugeTex,0,pointMinusG[0],c_stride);}

if(c_dimBreak[0] == true && (x_id == 0)){
  READVECTOR_FLOAT(S,vecInTex,pointMinus[0],c_surface[0]);}
 else{
   READVECTOR_FLOAT(S,vecInTex,pointMinus[0],c_stride);}

apply_U_DAG_on_S(P2_,G_,S_);

tmp[0] = P1_0 + P2_0;
tmp[1] = P1_1 + P2_1;
tmp[2] = P1_2 + P2_2;
tmp[3] = P1_3 + P2_3;
tmp[4] = P1_4 + P2_4;
tmp[5] = P1_5 + P2_5;
tmp[6] = P1_6 + P2_6;
tmp[7] = P1_7 + P2_7;
tmp[8] = P1_8 + P2_8;
tmp[9] = P1_9 + P2_9;
tmp[10] = P1_10 + P2_10;
tmp[11] = P1_11 + P2_11;

/////////////////////////////////// mu = 1

READGAUGE_FLOAT(G,gaugeTex,1,sid,c_stride);

if(c_dimBreak[1] == true && (y_id == c_localL[1] - 1)){
  READVECTOR_FLOAT(S,vecInTex,pointPlus[1],c_surface[1]);}
 else{
   READVECTOR_FLOAT(S,vecInTex,pointPlus[1],c_stride);}

  apply_U_on_S(P1_,G_,S_);

if(c_dimBreak[1] == true && (y_id == 0)){
  READGAUGE_FLOAT(G,gaugeTex,1,pointMinusG[1],c_surface[1]);}
 else{
   READGAUGE_FLOAT(G,gaugeTex,1,pointMinusG[1],c_stride);}

if(c_dimBreak[1] == true && (y_id == 0)){
  READVECTOR_FLOAT(S,vecInTex,pointMinus[1],c_surface[1]);}
 else{
   READVECTOR_FLOAT(S,vecInTex,pointMinus[1],c_stride);}

  apply_U_DAG_on_S(P2_,G_,S_);

tmp[0].x += P1_0.x + P2_0.x;
tmp[1].x += P1_1.x + P2_1.x;
tmp[2].x += P1_2.x + P2_2.x;
tmp[3].x += P1_3.x + P2_3.x;
tmp[4].x += P1_4.x + P2_4.x;
tmp[5].x += P1_5.x + P2_5.x;
tmp[6].x += P1_6.x + P2_6.x;
tmp[7].x += P1_7.x + P2_7.x;
tmp[8].x += P1_8.x + P2_8.x;
tmp[9].x += P1_9.x + P2_9.x;
tmp[10].x += P1_10.x + P2_10.x;
tmp[11].x += P1_11.x + P2_11.x;

tmp[0].y += P1_0.y + P2_0.y;
tmp[1].y += P1_1.y + P2_1.y;
tmp[2].y += P1_2.y + P2_2.y;
tmp[3].y += P1_3.y + P2_3.y;
tmp[4].y += P1_4.y + P2_4.y;
tmp[5].y += P1_5.y + P2_5.y;
tmp[6].y += P1_6.y + P2_6.y;
tmp[7].y += P1_7.y + P2_7.y;
tmp[8].y += P1_8.y + P2_8.y;
tmp[9].y += P1_9.y + P2_9.y;
tmp[10].y += P1_10.y + P2_10.y;
tmp[11].y += P1_11.y + P2_11.y;


///////////////////////////////// mu = 2

READGAUGE_FLOAT(G,gaugeTex,2,sid,c_stride);

if(c_dimBreak[2] == true && (z_id == c_localL[2] - 1)){
  READVECTOR_FLOAT(S,vecInTex,pointPlus[2],c_surface[2]);}
 else{
   READVECTOR_FLOAT(S,vecInTex,pointPlus[2],c_stride);}

  apply_U_on_S(P1_,G_,S_);

if(c_dimBreak[2] == true && (z_id == 0)){
  READGAUGE_FLOAT(G,gaugeTex,2,pointMinusG[2],c_surface[2]);}
 else{
   READGAUGE_FLOAT(G,gaugeTex,2,pointMinusG[2],c_stride);}

if(c_dimBreak[2] == true && (z_id == 0)){
  READVECTOR_FLOAT(S,vecInTex,pointMinus[2],c_surface[2]);}
 else{
   READVECTOR_FLOAT(S,vecInTex,pointMinus[2],c_stride);}

  apply_U_DAG_on_S(P2_,G_,S_);

tmp[0].x += P1_0.x + P2_0.x;
tmp[1].x += P1_1.x + P2_1.x;
tmp[2].x += P1_2.x + P2_2.x;
tmp[3].x += P1_3.x + P2_3.x;
tmp[4].x += P1_4.x + P2_4.x;
tmp[5].x += P1_5.x + P2_5.x;
tmp[6].x += P1_6.x + P2_6.x;
tmp[7].x += P1_7.x + P2_7.x;
tmp[8].x += P1_8.x + P2_8.x;
tmp[9].x += P1_9.x + P2_9.x;
tmp[10].x += P1_10.x + P2_10.x;
tmp[11].x += P1_11.x + P2_11.x;

tmp[0].y += P1_0.y + P2_0.y;
tmp[1].y += P1_1.y + P2_1.y;
tmp[2].y += P1_2.y + P2_2.y;
tmp[3].y += P1_3.y + P2_3.y;
tmp[4].y += P1_4.y + P2_4.y;
tmp[5].y += P1_5.y + P2_5.y;
tmp[6].y += P1_6.y + P2_6.y;
tmp[7].y += P1_7.y + P2_7.y;
tmp[8].y += P1_8.y + P2_8.y;
tmp[9].y += P1_9.y + P2_9.y;
tmp[10].y += P1_10.y + P2_10.y;
tmp[11].y += P1_11.y + P2_11.y;

  ////////////////

  READVECTOR_FLOAT(S,vecInTex,sid,c_stride);

  double normalize;
  normalize = 1./(1. + 6. * c_alphaGauss);

out[0*c_nColor*c_stride + 0*c_stride + sid] = normalize * (S_0 + c_alphaGauss * tmp[0]);
out[0*c_nColor*c_stride + 1*c_stride + sid] = normalize * (S_1 + c_alphaGauss * tmp[1]);
out[0*c_nColor*c_stride + 2*c_stride + sid] = normalize * (S_2 + c_alphaGauss * tmp[2]);
out[1*c_nColor*c_stride + 0*c_stride + sid] = normalize * (S_3 + c_alphaGauss * tmp[3]);
out[1*c_nColor*c_stride + 1*c_stride + sid] = normalize * (S_4 + c_alphaGauss * tmp[4]);
out[1*c_nColor*c_stride + 2*c_stride + sid] = normalize * (S_5 + c_alphaGauss * tmp[5]);
out[2*c_nColor*c_stride + 0*c_stride + sid] = normalize * (S_6 + c_alphaGauss * tmp[6]);
out[2*c_nColor*c_stride + 1*c_stride + sid] = normalize * (S_7 + c_alphaGauss * tmp[7]);
out[2*c_nColor*c_stride + 2*c_stride + sid] = normalize * (S_8 + c_alphaGauss * tmp[8]);
out[3*c_nColor*c_stride + 0*c_stride + sid] = normalize * (S_9 + c_alphaGauss * tmp[9]);
out[3*c_nColor*c_stride + 1*c_stride + sid] = normalize * (S_10 + c_alphaGauss * tmp[10]);
out[3*c_nColor*c_stride + 2*c_stride + sid] = normalize * (S_11 + c_alphaGauss * tmp[11]);
