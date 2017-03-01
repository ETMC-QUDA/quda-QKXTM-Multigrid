//-C.K. Interface for performing the loops and the correlation function 
//contractions, including the exact deflation using ARPACK
//#include <qudaQKXTM.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <qudaQKXTM_Kepler.h>
#include <qudaQKXTM_Kepler_utils.h>
#include <errno.h>
#include <mpi.h>
#include <limits>
//#include <mkl.h> //QXKTM: FIXME
#include <cblas.h>
#include <common.h>
#include <omp.h>
#include <hdf5.h>
 
#define PI 3.141592653589793
 
//using namespace quda;
extern Topology *default_topo;
 
/* Block for global variables */
extern float GK_deviceMemory;
extern int GK_nColor;
extern int GK_nSpin;
extern int GK_nDim;
extern int GK_strideFull;
extern double GK_alphaAPE;
extern double GK_alphaGauss;
extern int GK_localVolume;
extern int GK_totalVolume;
extern int GK_nsmearAPE;
extern int GK_nsmearGauss;
extern bool GK_dimBreak[QUDAQKXTM_DIM];
extern int GK_localL[QUDAQKXTM_DIM];
extern int GK_totalL[QUDAQKXTM_DIM];
extern int GK_nProc[QUDAQKXTM_DIM];
extern int GK_plusGhost[QUDAQKXTM_DIM];
extern int GK_minusGhost[QUDAQKXTM_DIM];
extern int GK_surface3D[QUDAQKXTM_DIM];
extern bool GK_init_qudaQKXTM_Kepler_flag;
extern int GK_Nsources;
extern int GK_sourcePosition[MAX_NSOURCES][QUDAQKXTM_DIM];
extern int GK_Nmoms;
extern short int GK_moms[MAX_NMOMENTA][3];
// for mpi use global variables
extern MPI_Group GK_fullGroup , GK_spaceGroup , GK_timeGroup;
extern MPI_Comm GK_spaceComm , GK_timeComm;
extern int GK_localRank;
extern int GK_localSize;
extern int GK_timeRank;
extern int GK_timeSize;

//------------------------------//
// class QKXTM_Deflation_Kelper //
//------------------------------//

//-C.K. Constructor for the even-odd operator functions
template<typename Float>
QKXTM_Deflation_Kepler<Float>::
QKXTM_Deflation_Kepler(int N_EigenVectors, bool isEven): 
  h_elem(NULL), eigenValues(NULL){
  if(GK_init_qudaQKXTM_Kepler_flag == false)
    errorQuda("You must initialize QKXTM library first\n");
  NeV=N_EigenVectors;
  if(NeV == 0){
    warningQuda("You chose zero eigenVectors\n");
    return;
  }

  isEv=isEven;
  isFullOp = false;

  field_length = 4*3;

  total_length_per_NeV = (GK_localVolume/2)*field_length;
  bytes_total_length_per_NeV = total_length_per_NeV*2*sizeof(Float);
  total_length = NeV*(GK_localVolume/2)*field_length;
  bytes_total_length = total_length*2*sizeof(Float);

  h_elem = (Float*)malloc(NeV*bytes_total_length_per_NeV);
  if(h_elem == NULL) errorQuda("Error: Out of memory for eigenVectors.\n");
  memset(h_elem,0,NeV*bytes_total_length_per_NeV);

  eigenValues = (Float*)malloc(2*NeV*sizeof(Float));
  if(eigenValues == NULL)errorQuda("Error with allocation host memory for deflation class\n");
}


//-C.K. Constructor for the full Operator functions
template<typename Float>
QKXTM_Deflation_Kepler<Float>::
QKXTM_Deflation_Kepler(QudaInvertParam *param, 
		       qudaQKXTM_arpackInfo arpackInfo): 
  h_elem(NULL), eigenValues(NULL), diracOp(NULL){
  if(GK_init_qudaQKXTM_Kepler_flag == false)
    errorQuda("You must initialize QKXTM library first\n");

  PolyDeg = arpackInfo.PolyDeg;
  NeV = arpackInfo.nEv;
  NkV = arpackInfo.nKv;
  // for which part of the spectrum we want to solve
  spectrumPart = arpackInfo.spectrumPart; 

  isACC = arpackInfo.isACC;
  tolArpack = arpackInfo.tolArpack;
  maxIterArpack = arpackInfo.maxIterArpack;
  strcpy(arpack_logfile,arpackInfo.arpack_logfile);
  amin = arpackInfo.amin;
  amax = arpackInfo.amax;
  isEv = arpackInfo.isEven;
  isFullOp = arpackInfo.isFullOp;
  flavor_sign = param->twist_flavor;

  if(NeV == 0){
    printfQuda("###############################\n");
    printfQuda("######### Got NeV = 0 #########\n");
    printfQuda("###############################\n");
    return;
  }

  invert_param = param;
  if(isFullOp) invert_param->solve_type = QUDA_NORMOP_SOLVE;
  else invert_param->solve_type = QUDA_NORMOP_PC_SOLVE;

  field_length = 4*3;

  fullorHalf = (isFullOp) ? 1 : 2;

  total_length_per_NeV = (GK_localVolume/fullorHalf)*field_length*2;
  bytes_total_length_per_NeV = total_length_per_NeV*sizeof(Float);
  //NeV*(GK_localVolume/fullorHalf)*field_length;
  total_length =  NeV*total_length_per_NeV;   
  //total_length*2*sizeof(Float);
  bytes_total_length = NeV*bytes_total_length_per_NeV; 
  
  h_elem = (Float*)malloc(NkV*bytes_total_length_per_NeV);
  if(h_elem == NULL) errorQuda("Error: Out of memory for eigenVectors.\n");
  memset(h_elem,0,NkV*bytes_total_length_per_NeV);

  eigenValues = (Float*)malloc(2*NkV*sizeof(Float));
  if(eigenValues == NULL)errorQuda("Error: Out of memory of eigenValues.\n");

  DiracParam diracParam;
  setDiracParam(diracParam,invert_param,!isFullOp);
  diracOp = Dirac::create(diracParam);
}

template<typename Float>
QKXTM_Deflation_Kepler<Float>::~QKXTM_Deflation_Kepler(){
  if(NeV == 0)return;

  free(h_elem);
  free(eigenValues);
  if(diracOp != NULL) delete diracOp;
}

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::printInfo(){
  printfQuda("\n======= DEFLATION INFO =======\n"); 
  if(isFullOp){
    printfQuda(" The EigenVectors are for the Full %smu operator\n", 
	       (flavor_sign==QUDA_TWIST_PLUS) ? "+" : "-");
  }
  else{
    printfQuda(" Will calculate EigenVectors for the %s %smu operator\n", 
	       isEv ? "even-even" : "odd-odd", 
	       (flavor_sign==QUDA_TWIST_PLUS) ? "+" : "-" );
  }

  printfQuda(" Number of requested EigenVectors is %d in precision %d\n",
	     NeV,(int) sizeof(Float));
  printfQuda(" The Size of Krylov space is %d\n",NkV);

  printfQuda(" Allocated Gb for the eigenVectors space for each node are %lf and the pointer is %p\n", NeV * ( (double)bytes_total_length_per_NeV/((double) 1024.*1024.*1024.) ),h_elem);
  printfQuda("==============================\n");
}
//==================================================

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::ApplyMdagM(Float *vec_out, 
					       Float *vec_in, 
					       QudaInvertParam *param){

  bool opFlag;

  if(isFullOp){
    printfQuda("Applying the Full Operator\n");
    opFlag = false;

    cudaColorSpinorField *in    = NULL;
    cudaColorSpinorField *out   = NULL;
    
    QKXTM_Vector_Kepler<double> *Kvec = 
      new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

    ColorSpinorParam cpuParam((void*)vec_in,*param,GK_localL,opFlag);
    ColorSpinorParam cudaParam(cpuParam, *param);

    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    in  = new cudaColorSpinorField(cudaParam);
    out = new cudaColorSpinorField(cudaParam);
    
    Kvec->packVector(vec_in);
    Kvec->loadVector();
    Kvec->uploadToCuda(in,opFlag);
    
    diracOp->MdagM(*out,*in);
    
    Kvec->downloadFromCuda(out,opFlag);
    Kvec->unloadVector();
    Kvec->unpackVector();
    
    memcpy(vec_out,Kvec->H_elem(),bytes_total_length_per_NeV);

    delete in;
    delete out;
    delete Kvec;
  }
  else{
    printfQuda("Applying the %s Operator\n",isEv ? "Even-Even" : "Odd-Odd");

    cudaColorSpinorField *in = NULL;
    cudaColorSpinorField *out = NULL;

    opFlag = isEv;
    bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION) ||
      (param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

    ColorSpinorParam cpuParam((void*)vec_in,*param,GK_localL,pc_solution);

    ColorSpinorField *h_b = 
      (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
      static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
      static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
    
    cpuParam.v = vec_out;
    ColorSpinorField *h_x = 
      (param->output_location == QUDA_CPU_FIELD_LOCATION) ?
      static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) :
      static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

    ColorSpinorParam cudaParam(cpuParam, *param);
    cudaParam.create = QUDA_COPY_FIELD_CREATE;
    in = new cudaColorSpinorField(*h_b, cudaParam);
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    out = new cudaColorSpinorField(cudaParam);

    QKXTM_Vector_Kepler<double> *Kvec = 
      new QKXTM_Vector_Kepler<double>(BOTH,VECTOR);

    Kvec->packVector(vec_in);
    Kvec->loadVector();
    Kvec->uploadToCuda(in,opFlag);
    
    diracOp->MdagM(*out,*in);
    
    Kvec->downloadFromCuda(out,opFlag);
    Kvec->unloadVector();
    Kvec->unpackVector();
    
    memcpy(vec_out,Kvec->H_elem(),bytes_total_length_per_NeV);

    delete in;
    delete out;
    delete h_b;
    delete h_x;
    delete Kvec;
  }

  printfQuda("ApplyMdagM: Completed successfully\n");
}
//==================================================

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::MapEvenOddToFull(){

  if(!isFullOp){ warningQuda("MapEvenOddToFull: This function only works with the Full Operator\n");
    return;
  }

  if(NeV==0) return;

  size_t bytes_eo = bytes_total_length_per_NeV/2;
  int size_eo = total_length_per_NeV/2;

  int site_size = 4*3*2;
  size_t bytes_per_site = site_size*sizeof(Float);

  if((bytes_eo%2)!=0) 
    errorQuda("MapEvenOddToFull: Invalid bytes for eo vector\n");
  if((size_eo%2)!=0) 
    errorQuda("MapEvenOddToFull: Invalid size for eo vector\n");

  Float *vec_odd = (Float*) malloc(bytes_eo);
  Float *vec_evn = (Float*) malloc(bytes_eo);

  if(vec_odd==NULL) 
    errorQuda("MapEvenOddToFull: Check allocation of vec_odd\n");
  if(vec_evn==NULL) 
    errorQuda("MapEvenOddToFull: Check allocation of vec_evn\n");

  printfQuda("MapEvenOddToFull: Vecs allocated\n");

  for(int i=0;i<NeV;i++){
    // Save the even half of the eigenvector
    memcpy(vec_evn,&(h_elem[i*total_length_per_NeV]),bytes_eo); 
    // Save the odd half of the eigenvector
    memcpy(vec_odd,&(h_elem[i*total_length_per_NeV+size_eo]),bytes_eo); 

    int k=0;
    for(int t=0; t<GK_localL[3];t++)
      for(int z=0; z<GK_localL[2];z++)
	for(int y=0; y<GK_localL[1];y++)
	  for(int x=0; x<GK_localL[0];x++){
	    int oddBit     = (x+y+z+t) & 1;
	    if(oddBit) memcpy(&(h_elem[i*total_length_per_NeV+site_size*k]),
			      &(vec_odd[site_size*(k/2)]),bytes_per_site);
	    else       memcpy(&(h_elem[i*total_length_per_NeV+site_size*k]),
			      &(vec_evn[site_size*(k/2)]),bytes_per_site);
	    k++;
	  }	  
  }

  printfQuda("MapEvenOddToFull: Completed successfully\n");
}

//-For a single vector
template<typename Float>
void QKXTM_Deflation_Kepler<Float>::MapEvenOddToFull(int i){

  if(!isFullOp) errorQuda("MapEvenOddToFull: This function only works with the Full Operator\n");

  if(NeV==0) return;

  size_t bytes_eo = bytes_total_length_per_NeV/2;
  int size_eo = total_length_per_NeV/2;

  int site_size = 4*3*2;
  size_t bytes_per_site = site_size*sizeof(Float);

  if((bytes_eo%2)!=0) 
    errorQuda("MapEvenOddToFull: Invalid bytes for eo vector\n");
  if((size_eo%2)!=0) 
    errorQuda("MapEvenOddToFull: Invalid size for eo vector\n");

  Float *vec_odd = (Float*) malloc(bytes_eo);
  Float *vec_evn = (Float*) malloc(bytes_eo);

  if(vec_odd==NULL) 
    errorQuda("MapEvenOddToFull: Check allocation of vec_odd\n");
  if(vec_evn==NULL) 
    errorQuda("MapEvenOddToFull: Check allocation of vec_evn\n");

  printfQuda("MapEvenOddToFull: Vecs allocated\n");

  // Save the even half of the eigenvector
  memcpy(vec_evn,&(h_elem[i*total_length_per_NeV]),bytes_eo); 
  // Save the odd half of the eigenvector
  memcpy(vec_odd,&(h_elem[i*total_length_per_NeV+size_eo]),bytes_eo);

  int k=0;
  for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
      for(int y=0; y<GK_localL[1];y++)
	for(int x=0; x<GK_localL[0];x++){
	  int oddBit     = (x+y+z+t) & 1;
	  if(oddBit) memcpy(&(h_elem[i*total_length_per_NeV+site_size*k]),
			    &(vec_odd[site_size*(k/2)]),bytes_per_site);
	  else       memcpy(&(h_elem[i*total_length_per_NeV+site_size*k]),
			    &(vec_evn[site_size*(k/2)]),bytes_per_site);
	  k++;
	}	  
  
  printfQuda("MapEvenOddToFull: Vector %d completed successfully\n",i);
}


template<typename Float>
void QKXTM_Deflation_Kepler<Float>::
copyEigenVectorToQKXTM_Vector_Kepler(int eigenVector_id, Float *vec){
  if(NeV == 0)return;
  
  if(!isFullOp){
    printfQuda("Copying elements of Eigenvector %d according to %s Operator format\n", eigenVector_id, isEv ? "even-even" : "odd-odd");
    for(int t=0; t<GK_localL[3];t++)
      for(int z=0; z<GK_localL[2];z++)
	for(int y=0; y<GK_localL[1];y++)
	  for(int x=0; x<GK_localL[0];x++)
	    for(int mu=0; mu<4; mu++)
	      for(int c1=0; c1<3; c1++)
		{
		  int oddBit     = (x+y+z+t) & 1;
		  if(oddBit){
		    if(isEv == false){
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 0] = 
			h_elem[eigenVector_id*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
				 z*GK_localL[1]*GK_localL[0] + 
				 y*GK_localL[0] + 
				 x)/2)*4*3*2 + mu*3*2 + c1*2 + 0];
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 1] = 
			h_elem[eigenVector_id*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
				 z*GK_localL[1]*GK_localL[0] + 
				 y*GK_localL[0] + 
				 x)/2)*4*3*2 + mu*3*2 + c1*2 + 1];
		    }
		    else{
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 0] =0.;
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 1] =0.; 
		    }
		  } // if for odd
		  else{
		    if(isEv == true){
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 0] = 
			h_elem[eigenVector_id*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
				 z*GK_localL[1]*GK_localL[0] + 
				 y*GK_localL[0] + 
				 x)/2)*4*3*2 + mu*3*2 + c1*2 + 0];
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 1] = 
			h_elem[eigenVector_id*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
				 z*GK_localL[1]*GK_localL[0] + 
				 y*GK_localL[0] + 
				 x)/2)*4*3*2 + mu*3*2 + c1*2 + 1];
		    }
		    else{
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 0] =0.;
		      vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)*4*3*2 + mu*3*2 + c1*2 + 1] =0.; 
		    }
		  }
		}
  }//-isFullOp check
  else if(isFullOp){
    printfQuda("Copying elements of Eigenvector %d according to Full Operator format\n",eigenVector_id);
    memcpy(vec,&(h_elem[eigenVector_id*total_length_per_NeV]),
	   bytes_total_length_per_NeV);
  }

}

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::
copyEigenVectorFromQKXTM_Vector_Kepler(int eigenVector_id,Float *vec){
  if(NeV == 0)return;
  
  if(!isFullOp){
    for(int t=0; t<GK_localL[3];t++)
      for(int z=0; z<GK_localL[2];z++)
	for(int y=0; y<GK_localL[1];y++)
	  for(int x=0; x<GK_localL[0];x++)
	    for(int mu=0; mu<4; mu++)
	      for(int c1=0; c1<3; c1++)
		{
		  int oddBit     = (x+y+z+t) & 1;
		  if(oddBit){
		    if(isEv == false){
		      h_elem[eigenVector_id*total_length_per_NeV + 
			     ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			       z*GK_localL[1]*GK_localL[0] + 
			       y*GK_localL[0] + 
			       x)/2)*4*3*2 + mu*3*2 + c1*2 + 0] = 
			vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			     z*GK_localL[1]*GK_localL[0] + 
			     y*GK_localL[0] + 
			     x)*4*3*2 + mu*3*2 + c1*2 + 0];
		      h_elem[eigenVector_id*total_length_per_NeV + 
			     ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			       z*GK_localL[1]*GK_localL[0] + 
			       y*GK_localL[0] + 
			       x)/2)*4*3*2 + mu*3*2 + c1*2 + 1] = 
			vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			     z*GK_localL[1]*GK_localL[0] + 
			     y*GK_localL[0] + 
			     x)*4*3*2 + mu*3*2 + c1*2 + 1];
		    }
		  } // if for odd
		  else{
		    if(isEv == true){
		      h_elem[eigenVector_id*total_length_per_NeV + 
			     ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			       z*GK_localL[1]*GK_localL[0] + 
			       y*GK_localL[0] + 
			       x)/2)*4*3*2 + mu*3*2 + c1*2 + 0] = 
			vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			     z*GK_localL[1]*GK_localL[0] + 
			     y*GK_localL[0] + 
			     x)*4*3*2 + mu*3*2 + c1*2 + 0];
		      h_elem[eigenVector_id*total_length_per_NeV + 
			     ((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			       z*GK_localL[1]*GK_localL[0] + 
			       y*GK_localL[0] + 
			       x)/2)*4*3*2 + mu*3*2 + c1*2 + 1] = 
			vec[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			     z*GK_localL[1]*GK_localL[0] + 
			     y*GK_localL[0] + 
			     x)*4*3*2 + mu*3*2 + c1*2 + 1];
		    }
		  }
		}
  }//-isFullOp check
  else if(isFullOp){
    memcpy(&(h_elem[eigenVector_id*total_length_per_NeV]),
	   vec,bytes_total_length_per_NeV);
  }

}

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::copyToEigenVector(Float *vec, 
						      Float *vals){
  memcpy(&(h_elem[0]), vec, bytes_total_length);
  memcpy(&(eigenValues[0]), vals, NeV*2*sizeof(Float));
}


//-C.K: This member function performs the operation vec_defl = U (\Lambda)^(-1) U^dag vec_in
template <typename Float>
void QKXTM_Deflation_Kepler<Float>::
deflateVector(QKXTM_Vector_Kepler<Float> &vec_defl, 
	      QKXTM_Vector_Kepler<Float> &vec_in){
  if(NeV == 0){
    vec_defl.zero_device();
    return;
  }
  
  Float *tmp_vec = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;
  Float *tmp_vec_lex = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float));
  Float *out_vec = (Float*) calloc(NeV*2,sizeof(Float)) ;
  Float *out_vec_reduce = (Float*) calloc(NeV*2,sizeof(Float)) ;
  
  if(tmp_vec        == NULL || 
     tmp_vec_lex    == NULL || 
     out_vec        == NULL || 
     out_vec_reduce == NULL)
    errorQuda("Error with memory allocation in deflation method\n");
  
  Float *tmp_vec_even = tmp_vec;
  Float *tmp_vec_odd = tmp_vec + (GK_localVolume/2)*4*3*2;
  
  if(!isFullOp){
    for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
    for(int y=0; y<GK_localL[1];y++)
    for(int x=0; x<GK_localL[0];x++)
    for(int mu=0; mu<4; mu++)
    for(int c1=0; c1<3; c1++)
      {
	int oddBit = (x+y+z+t) & 1;
	if(oddBit){
	  for(int ipart = 0 ; ipart < 2 ; ipart++)
	    tmp_vec_odd[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			  z*GK_localL[1]*GK_localL[0] + 
			  y*GK_localL[0] + 
			  x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart] = 
	  (Float) vec_in.H_elem()[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] +
				   z*GK_localL[1]*GK_localL[0] + 
				   y*GK_localL[0] + 
				   x)*4*3*2 + mu*3*2 + c1*2 + ipart];
	}
	else{
	  for(int ipart = 0 ; ipart < 2 ; ipart++)
	    tmp_vec_even[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			   z*GK_localL[1]*GK_localL[0] + 
			   y*GK_localL[0] + 
			   x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart] = 
	  (Float) vec_in.H_elem()[(t*GK_localL[2]*GK_localL[1]*GK_localL[0]+ 
				   z*GK_localL[1]*GK_localL[0] + 
				   y*GK_localL[0] + 
				   x)*4*3*2 + mu*3*2 + c1*2 + ipart];
	}
      }  
  }  
  else if(isFullOp){
    memcpy(tmp_vec,vec_in.H_elem(),bytes_total_length_per_NeV);
  }

  Float alpha[2] = {1.,0.};
  Float beta[2] = {0.,0.};
  int incx = 1;
  int incy = 1;
  long int NN = (GK_localVolume/fullorHalf)*4*3;

  Float *ptr_elem = NULL;

  if(!isFullOp){
    if(isEv == true){
      ptr_elem = tmp_vec_even;
    }
    else{
      ptr_elem = tmp_vec_odd;
    }
  }
  else{
    ptr_elem = tmp_vec;
  }


  if( typeid(Float) == typeid(float) ){
    //-C.K: out_vec = H_elem^dag * ptr_elem -> U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasConjTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, ptr_elem, incx, 
		(void*) beta, out_vec, incy ); 
    //-C.K_CHECK: This might not be needed
    memset(ptr_elem,0,NN*2*sizeof(Float));
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_FLOAT,
		  MPI_SUM,MPI_COMM_WORLD);
    for(int i = 0 ; i < NeV ; i++){
      //-Eigenvalues are real!
      out_vec_reduce[i*2+0] /= eigenValues[i*2+0]; 
      //-C.K: out_vec_reduce -> \Lambda^(-1) * U^dag * vec_in
      out_vec_reduce[i*2+1] /= eigenValues[i*2+0]; 
    }
    //-C.K: ptr_elem = H_elem * out_vec_reduce -> ptr_elem = U * \Lambda^(-1) * U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasNoTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, out_vec_reduce, incx, 
		(void*) beta, ptr_elem, incy );
  }
  else if ( typeid(Float) == typeid(double) ){
    cblas_zgemv(CblasColMajor, CblasConjTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, ptr_elem, incx, 
		(void*) beta, out_vec, incy );
    //-C.K_CHECK: This might not be needed
    memset(ptr_elem,0,NN*2*sizeof(Float)); 
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_DOUBLE,
		  MPI_SUM,MPI_COMM_WORLD);
    for(int i = 0 ; i < NeV ; i++){
      out_vec_reduce[i*2+0] /= eigenValues[2*i+0];
      out_vec_reduce[i*2+1] /= eigenValues[2*i+0];
    }
    cblas_zgemv(CblasColMajor, CblasNoTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, out_vec_reduce, incx, 
		(void*) beta, ptr_elem, incy );    
  }
  
  

  if(!isFullOp){
    for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
    for(int y=0; y<GK_localL[1];y++)
    for(int x=0; x<GK_localL[0];x++)
    for(int mu=0; mu<4; mu++)
    for(int c1=0; c1<3; c1++)
      {
	int oddBit     = (x+y+z+t) & 1;
	if(oddBit){
	  for(int ipart = 0 ; ipart < 2 ; ipart++)
	    tmp_vec_lex[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			 z*GK_localL[1]*GK_localL[0] + 
			 y*GK_localL[0] + 
			 x)*4*3*2 + mu*3*2 + c1*2 + ipart] = 
	      tmp_vec_odd[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			    z*GK_localL[1]*GK_localL[0] + 
			    y*GK_localL[0] + 
			    x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart];
	}
	else{
	  for(int ipart = 0 ; ipart < 2 ; ipart++)
	    tmp_vec_lex[(t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			 z*GK_localL[1]*GK_localL[0] + 
			 y*GK_localL[0] + 
			 x)*4*3*2 + mu*3*2 + c1*2 + ipart] = 
	      tmp_vec_even[((t*GK_localL[2]*GK_localL[1]*GK_localL[0] + 
			     z*GK_localL[1]*GK_localL[0] + 
			     y*GK_localL[0] + 
			     x)/2)*4*3*2 + mu*3*2 + c1*2 + ipart];
	}
      }  
  }
  else{
    //    memcpy(tmp_vec_lex,tmp_vec,bytes_total_length_per_NeV);
    memcpy(tmp_vec_lex,ptr_elem,bytes_total_length_per_NeV);
  }

  vec_defl.packVector((Float*) tmp_vec_lex);
  vec_defl.loadVector();


  free(out_vec);
  free(out_vec_reduce);
  free(tmp_vec);
  free(tmp_vec_lex);

  //  printfQuda("deflateVector: Deflation of the initial guess completed succesfully\n");
}


template<typename Float>
void QKXTM_Deflation_Kepler<Float>::
writeEigenVectors_ASCII(char *prefix_path){
  if(NeV == 0)return;
  char filename[257];
  if(comm_rank() != 0) return;
  FILE *fid;
  //int n_elem_write = 240;
  int n_elem_write = (GK_localVolume/fullorHalf)*4*3;
  for(int nev = 0 ; nev < NeV ; nev++){
    sprintf(filename,"%s.%04d.txt",prefix_path,nev);
    fid = fopen(filename,"w");		  
    for(int ir = 0 ; ir < n_elem_write ; ir++)
      fprintf(fid,"%+e %+e\n",h_elem[nev*total_length_per_NeV + ir*2 + 0], 
	      h_elem[nev*total_length_per_NeV + ir*2 + 1]);
    
    fclose(fid);
  }
}


template<typename Float>
void QKXTM_Deflation_Kepler<Float>::
polynomialOperator(cudaColorSpinorField &out, 
		   const cudaColorSpinorField &in){
  
  if(typeid(Float) != typeid(double)) errorQuda("Single precision is not implemented in member function of polynomial operator\n");

  double delta,theta;
  double sigma,sigma1,sigma_old;
  double d1,d2,d3;

  double a = amin;
  double b = amax;

  delta = (b-a)/2.0;
  theta = (b+a)/2.0;

  sigma1 = -delta/theta;
  blas::copy(out,in);

  if( PolyDeg == 0 ){
    printfQuda("Got degree of the polynomial to be 0. Proceeding anyway.\n");
    return;
  }
  
  d1 =  sigma1/delta;
  d2 =  1.0;

  //  (*matDiracOp)(out,in); //!!!!! check if I need (2*k)^2
  diracOp->MdagM(out,in); //!!!!! check if I need (2*k)^2
  blas::axpby(d2, const_cast<cudaColorSpinorField&>(in), d1, out);

  if( PolyDeg == 1 )
    return;

  cudaColorSpinorField *tm1 = new cudaColorSpinorField(in);
  cudaColorSpinorField *tm2 = new cudaColorSpinorField(in);

  blas::copy(*tm1,in);
  blas::copy(*tm2,out);

  sigma_old = sigma1;

  for(int i=2; i <= PolyDeg; i++){
    sigma = 1.0/(2.0/sigma1-sigma_old);
    
    d1 = 2.0*sigma/delta;
    d2 = -d1*theta;
    d3 = -sigma*sigma_old;
    
    //    (*matDiracOp)( out, *tm2); //!!!!! check if I need (2*k)^2
    diracOp->MdagM( out, *tm2); //!!!!! check if I need (2*k)^2
    // axCuda(1./(2.*shift),out);
    
    blas::ax(d3,*tm1);
    std::complex<double> d1c(d1,0);
    std::complex<double> d2c(d2,0);
    blas::cxpaypbz(*tm1,d2c,*tm2,d1c,out);
    blas::copy(*tm1,*tm2);
    blas::copy(*tm2,out);
    sigma_old  = sigma;
  }

  delete tm1;
  delete tm2;

}

#include <sortingFunctions.h>
#include <arpackHeaders.h>

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::eigenSolver(){

  double t1,t2,t_ini,t_fin;

  if(NeV==0){
    printfQuda("eigenSolver: Got NeV=%d. Returning...\n",NeV);
    return;
  }

  //-print the input:

  char *which_evals_req;
  if (spectrumPart==SR)      which_evals_req = strdup("SR");
  else if (spectrumPart==LR) which_evals_req = strdup("LR");
  else if (spectrumPart==SM) which_evals_req = strdup("SM");
  else if (spectrumPart==LM) which_evals_req = strdup("LM");
  else if (spectrumPart==SI) which_evals_req = strdup("SI");
  else if (spectrumPart==LI) which_evals_req = strdup("LI");
  else{    
    errorQuda("eigenSolver: Option for spectrumPart is suspicious\n");
    exit(-1);
  }
  

  char *which_evals;
  if(isACC){
    if (spectrumPart==SR)      which_evals = strdup("LR");
    else if (spectrumPart==LR) which_evals = strdup("SR");
    else if (spectrumPart==SM) which_evals = strdup("LM");
    else if (spectrumPart==LM) which_evals = strdup("SM");
    else if (spectrumPart==SI) which_evals = strdup("LI");
    else if (spectrumPart==LI) which_evals = strdup("SI");
  }
  else{
    if (spectrumPart==SR)      which_evals = strdup("SR");
    else if (spectrumPart==LR) which_evals = strdup("LR");
    else if (spectrumPart==SM) which_evals = strdup("SM");
    else if (spectrumPart==LM) which_evals = strdup("LM");
    else if (spectrumPart==SI) which_evals = strdup("SI");
    else if (spectrumPart==LI) which_evals = strdup("LI");    
  }

  printfQuda("\neigenSolver: Input to ARPACK\n");
  printfQuda("========================================\n");
  printfQuda(" Number of Ritz eigenvalues requested: %d\n", NeV);
  printfQuda(" Size of Krylov space is: %d\n", NkV);
  printfQuda(" Part of the spectrum requested: %s\n", which_evals_req);
  printfQuda(" Part of the spectrum passed to ARPACK (may be different due to Poly. Acc.): %s\n", which_evals);
  printfQuda(" Polynomial acceleration: %s\n", isACC ? "yes" : "no");
  if(isACC) printfQuda(" Chebyshev polynomial paramaters: Degree = %d, amin = %+e, amax = %+e\n",PolyDeg,amin,amax); 
  printfQuda(" The convergence criterion is %+e\n", tolArpack);
  printfQuda(" Maximum number of iterations for ARPACK is %d\n",maxIterArpack);
  printfQuda("========================================\n\n");

  //--------------------------------------------//
  
  //- create the MPI communicator
#ifdef MPI_COMMS
  MPI_Fint mpi_comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);
#endif

  // control of the action taken by reverse communications 
  // (set initially to zero) 
  int ido=0;               
  // Specifies that the right hand side matrix should be the 
  // identity matrix; this makes the problem a standard eigenvalue problem.
  char *bmat=strdup("I");  
                               
  QudaInvertParam *param = invert_param;
 
  //- matrix dimensions 
  int LDV = (GK_localVolume/fullorHalf)*4*3;
  int N   = (GK_localVolume/fullorHalf)*4*3;
  printfQuda("eigenSolver: Number of complex elements: %d\n",LDV);

  //- Define all the necessary pointers
  std::complex<Float> *helem_cplx = NULL;
  helem_cplx = (std::complex<Float>*) &(h_elem[0]);

  std::complex<Float> *evals_cplx = NULL;
  evals_cplx = (std::complex<Float>*) &(eigenValues[0]);

  int *ipntr              = (int *) malloc(14 *sizeof(int));
  //since all Ritz vectors or Schur vectors are computed no need to 
  //initialize this array
  int *select             = (int *) malloc(NkV*sizeof(int)); 
  int *sorted_evals_index = (int *) malloc(NkV*sizeof(int)); 
  int *iparam             = (int *) malloc(11 *sizeof(int));
  // always call the subroutine that computes orthonormal basis for 
  // the eigenvectors
  int rvec = 1;  
  // just allocate more space
  int lworkl = (3*NkV*NkV+5*NkV)*2;

  //always compute orthonormal basis
  char *howmany = strdup("P"); 

  double *rwork        = (double *) malloc(NkV*sizeof(double));
  //will be used to sort the eigenvalues
  double *sorted_evals = (double *) malloc(NkV*sizeof(double)); 
  
  std::complex<Float> *resid  = 
    (std::complex<Float> *) malloc(LDV   *sizeof(std::complex<Float>));
  std::complex<Float> *workd  = 
    (std::complex<Float> *) malloc(3*LDV *sizeof(std::complex<Float>)); 
  std::complex<Float> *workl  = 
    (std::complex<Float> *) malloc(lworkl*sizeof(std::complex<Float>));
  std::complex<Float> *workev = 
    (std::complex<Float> *) malloc(2*NkV *sizeof(std::complex<Float>));
  std::complex<Float> sigma;

  if(resid == NULL)  errorQuda("eigenSolver: not enough memory for resid allocation in eigenSolver.\n");
  if(iparam == NULL) errorQuda("eigenSolver: not enough memory for iparam allocation in eigenSolver.\n");

  if((ipntr == NULL) || 
     (workd==NULL) || 
     (workl==NULL) || 
     (rwork==NULL) || 
     (select==NULL) || 
     (workev==NULL) || 
     (sorted_evals==NULL) || 
     (sorted_evals_index==NULL)){
    errorQuda("eigenSolver: not enough memory for ipntr,workd,workl,rwork,select,workev,sorted_evals,sorted_evals_index in eigenSolver.\n");
  }
  
  iparam[0] = 1;  //use exact shifts
  iparam[2] = maxIterArpack;
  iparam[3] = 1;
  iparam[6] = 1;

  double d1,d2,d3;

  int info;
  //means use a random starting vector with Arnoldi
  info = 0;               
  
  int i,j;

  // Code added to print the log of ARPACK  
  int arpack_log_u = 9999;

#ifndef MPI_COMMS
  if ( NULL != arpack_logfile ) {
    // correctness of this code depends on alignment in Fortran and C 
    // being the same ; if you observe crashes, disable this part 
    
    _AFT(initlog)(&arpack_log_u, arpack_logfile, strlen(arpack_logfile));
    int msglvl0 = 0,
      msglvl1 = 1,
      msglvl2 = 2,
      msglvl3 = 3;
    _AFT(mcinitdebug)(
		      &arpack_log_u,      //logfil
		      &msglvl3,           //mcaupd
		      &msglvl3,           //mcaup2
		      &msglvl0,           //mcaitr
		      &msglvl3,           //mceigh
		      &msglvl0,           //mcapps
		      &msglvl0,           //mcgets
		      &msglvl3            //mceupd
		      );
    
    printfQuda("eigenSolver: Log info:\n");
    printfQuda(" ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
    printfQuda(" output is directed to %s\n",arpack_logfile);
  }
#else  
  if ( NULL != arpack_logfile && (comm_rank() == 0) ) {
    // correctness of this code depends on alignment in Fortran and C 
    // being the same ; if you observe crashes, disable this part 
    _AFT(initlog)(&arpack_log_u, arpack_logfile, strlen(arpack_logfile));
    int msglvl0 = 0,
      msglvl1 = 1,
      msglvl2 = 2,
      msglvl3 = 3;
    _AFT(pmcinitdebug)(
		       &arpack_log_u,      //logfil
		       &msglvl3,           //mcaupd
		       &msglvl3,           //mcaup2
		       &msglvl0,           //mcaitr
		       &msglvl3,           //mceigh
		       &msglvl0,           //mcapps
		       &msglvl0,           //mcgets
		       &msglvl3            //mceupd
		       );
    
    printfQuda("eigenSolver: Log info:\n");
    printfQuda(" ARPACK verbosity set to mcaup2=3 mcaupd=3 mceupd=3; \n");
    printfQuda(" output is directed to %s\n",arpack_logfile);
  }
#endif   

  cpuColorSpinorField *h_v = NULL;
  cudaColorSpinorField *d_v = NULL;

  cpuColorSpinorField *h_v2 = NULL;
  cudaColorSpinorField *d_v2 = NULL;

  int nconv;

  //M A I N   L O O P (Reverse communication)  

  bool checkIdo = true;

  t_ini = MPI_Wtime();

  do{

#ifndef MPI_COMMS 
    _AFT(znaupd)(&ido,"I", &N, which_evals, &NeV, &tolArpack, resid, &NkV,
		 helem_cplx, &N, iparam, ipntr, workd, 
		 workl, &lworkl,rwork,&info,1,2); 
#else
    _AFT(pznaupd)(&mpi_comm_f, &ido,"I", &N, which_evals, 
		  &NeV, &tolArpack, resid, &NkV,
		  helem_cplx, &N, iparam, ipntr, workd, 
		  workl, &lworkl,rwork,&info,1,2);
#endif

    if(checkIdo){
      // !!!!!!! please check that ipntr[0] does not change 
      ColorSpinorParam cpuParam(workd+ipntr[0]-1,*param,GK_localL,!isFullOp);
      h_v = new cpuColorSpinorField(cpuParam);
      cpuParam.v=workd+ipntr[1]-1;
      h_v2 = new cpuColorSpinorField(cpuParam);

      ColorSpinorParam cudaParam(cpuParam, *param);
      cudaParam.create = QUDA_ZERO_FIELD_CREATE;
      d_v = new cudaColorSpinorField( cudaParam);
      d_v2 = new cudaColorSpinorField( cudaParam);
      checkIdo = false;
    }
	
    if (ido == 99 || info == 1)
      break;

    if( (ido==-1) || (ido==1) ){
      *d_v = *h_v;
      if(isACC){
	polynomialOperator(*d_v2,*d_v);
      }
      else{
	diracOp->MdagM(*d_v2,*d_v);
      }
      *h_v2= *d_v2;
    }

  } while (ido != 99);
  
  /*
    Check for convergence 
  */
  if ( (info) < 0 ){
    printfQuda("eigenSolver: Error with _naupd, info = %d\n", info);
  }
  else{ 
    nconv = iparam[4];
    printfQuda("eigenSolver: Number of converged eigenvalues: %d\n", nconv);
    t_fin = MPI_Wtime();
    printfQuda("eigenSolver: TIME_REPORT - Eigenvalue calculation: %f sec\n"
	       ,t_fin-t_ini);
    printfQuda("eigenSolver: Computing eigenvectors...\n");
    t_ini = MPI_Wtime();

    //compute eigenvectors 
#ifndef MPI_COMMS
    _AFT(zneupd) (&rvec,"P", select,evals_cplx,helem_cplx,&N,&sigma, 
		  workev,"I",&N,which_evals,&NeV,&tolArpack,resid,&NkV, 
		  helem_cplx,&N,iparam,ipntr,workd,workl,&lworkl, 
		  rwork,&info,1,1,2);
#else
    _AFT(pzneupd) (&mpi_comm_f,&rvec,"P", select,evals_cplx, 
		   helem_cplx,&N,&sigma, 
		   workev,"I",&N,which_evals,&NeV,&tolArpack, resid,&NkV, 
		   helem_cplx,&N,iparam,ipntr,workd,workl,&lworkl, 
		   rwork,&info,1,1,2);
#endif

    if( (info)!=0){
      printfQuda("eigenSolver: Error with _neupd, info = %d \n",(info));
      printfQuda("eigenSolver: Check the documentation of _neupd. \n");
    }
    else{ //report eiegnvalues and their residuals
      t_fin = MPI_Wtime();
      printfQuda("eigenSolver: TIME_REPORT - Eigenvector calculation: %f sec\n",t_fin-t_ini);
      printfQuda("Ritz Values and their errors\n");
      printfQuda("============================\n");

      /* print out the computed ritz values and their error estimates */
      nconv = iparam[4];
      for(j=0; j< nconv; j++){
	printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		   real(evals_cplx[j]),
		   imag(evals_cplx[j]),
		   std::abs(*(workl+ipntr[10]-1+j)));
	sorted_evals_index[j] = j;
	sorted_evals[j] = std::abs(evals_cplx[j]);
      }

      //SORT THE EIGENVALUES in absolute ascending order
      t1 = MPI_Wtime();
      //quicksort(nconv,sorted_evals,sorted_evals_index);
      sortAbs(sorted_evals,nconv,false,sorted_evals_index);
      //Print sorted evals
      t2 = MPI_Wtime();
      printfQuda("Sorting time: %f sec\n",t2-t1);
      printfQuda("Sorted eigenvalues based on their absolute values:\n");
      
      // print out the computed ritz values and their error estimates 
      for(j=0; j< nconv; j++){
	printfQuda("RitzValue[%04d]  %+e  %+e  error= %+e \n",j,
		   real(evals_cplx[sorted_evals_index[j]]),
		   imag(evals_cplx[sorted_evals_index[j]]),
		   std::abs(*(workl+ipntr[10]-1+sorted_evals_index[j])) );
      }      
    }

    /*Print additional convergence information.*/
    if( (info)==1){
      printfQuda("Maximum number of iterations reached.\n");
    }
    else{
      if(info==3){
	printfQuda("Error: No shifts could be applied during implicit\n");
	printfQuda("Error: Arnoldi update, try increasing NkV.\n");
      }
    }
  }//- if(info < 0) else part

#ifndef MPI_COMMS
  if (NULL != arpack_logfile)
    _AFT(finilog)(&arpack_log_u);
#else
  if(comm_rank() == 0){
    if (NULL != arpack_logfile){
      _AFT(finilog)(&arpack_log_u);
    }
  }
#endif     

  //- calculate eigenvalues of the actual operator
  printfQuda("Eigenvalues of the %s Dirac operator:\n",
	     isFullOp ? "Full" : "Even-Odd");
  printfQuda("===========\n");

  t1 = MPI_Wtime();
  // !!!!!!! please check that ipntr[0] does not change 
  ColorSpinorParam cpuParam3(helem_cplx,*param,GK_localL,!isFullOp); 
  cpuColorSpinorField *h_v3 = NULL;
  for(int i =0 ; i < NeV ; i++){
    cpuParam3.v = (helem_cplx+i*LDV);
    h_v3 = new cpuColorSpinorField(cpuParam3);
    *d_v = *h_v3;                                    //d_v = v
    diracOp->MdagM(*d_v2,*d_v);                      //d_v2 = M*v
    evals_cplx[i]=blas::cDotProduct(*d_v,*d_v2);     //lambda = v^dag * M*v
    blas::axpby(1.0,*d_v2,-real(evals_cplx[i]),*d_v);//d_v=||M*v-lambda*v||

    //QKXTM: DMH careful here. It might be norm() in a different namespace...
    double norma = blas::norm2(*d_v);
    printfQuda("Eval[%04d] = %+e  %+e    Residual: %+e\n",
	       i,real(evals_cplx[i]),imag(evals_cplx[i]),sqrt(norma));
    delete h_v3;
  }
  t2 = MPI_Wtime();
  printfQuda("\neigenSolver: TIME_REPORT - Eigenvalues of Dirac operator: %f sec\n",t2-t1);

  //free memory
  free(resid);
  free(iparam);
  free(ipntr);
  free(workd);
  free(workl);
  free(rwork);
  free(sorted_evals);
  free(sorted_evals_index);
  free(select);
  free(workev);

  delete h_v;
  delete h_v2;
  delete d_v;
  delete d_v2;

  return;
}

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::rotateFromChiralToUKQCD(){
  if(NeV == 0) return;
  std::complex<Float> transMatrix[4][4];
  for(int mu = 0 ; mu < 4 ; mu++)
    for(int nu = 0 ; nu < 4 ; nu++){
      transMatrix[mu][nu].real(0.0);
      transMatrix[mu][nu].imag(0.0);
    }

  Float value = 1./sqrt(2.);

  transMatrix[0][0].real(-value); // g4*g5*U
  transMatrix[1][1].real(-value);
  transMatrix[2][2].real(value);
  transMatrix[3][3].real(value);

  transMatrix[0][2].real(value);
  transMatrix[1][3].real(value);
  transMatrix[2][0].real(value);
  transMatrix[3][1].real(value);

  std::complex<Float> tmp[4];
  std::complex<Float> *vec_cmlx = NULL;

  for(int i = 0 ; i < NeV ; i++){
    vec_cmlx = (std::complex<Float>*) &(h_elem[i*total_length_per_NeV]);
    for(int iv = 0 ; iv < (GK_localVolume)/fullorHalf ; iv++){
      for(int ic = 0 ; ic < 3 ; ic++){
	memset(tmp,0,4*2*sizeof(Float));
	for(int mu = 0 ; mu < 4 ; mu++)
	  for(int nu = 0 ; nu < 4 ; nu++)
	    tmp[mu] = tmp[mu] + transMatrix[mu][nu] * ( *(vec_cmlx+(iv*4*3+nu*3+ic)) );
	for(int mu = 0 ; mu < 4 ; mu++)
	  *(vec_cmlx+(iv*4*3+mu*3+ic)) = tmp[mu];
      }//-ic
    }//-iv
  }//-i

  printfQuda("Rotation to UKQCD basis completed successfully\n");
}

template<typename Float>
void QKXTM_Deflation_Kepler<Float>::multiply_by_phase(){
  if(NeV == 0)return;
  Float phaseRe, phaseIm;
  Float tmp0,tmp1;

  if(!isFullOp){
    for(int ivec = 0 ; ivec < NeV ; ivec++)
    for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
    for(int y=0; y<GK_localL[1];y++)
    for(int x=0; x<GK_localL[0];x++)
    for(int mu=0; mu<4; mu++)
    for(int c1=0; c1<3; c1++)
      {
	int oddBit     = (x+y+z+t) & 1;
	if(oddBit){
	  continue;
	}
	else{
	  phaseRe = cos(PI*(t+comm_coords(default_topo)[3]*GK_localL[3])/((Float) GK_totalL[3]));
	  phaseIm = sin(PI*(t+comm_coords(default_topo)[3]*GK_localL[3])/((Float) GK_totalL[3]));
	  int pos = ((t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
		      z*GK_localL[1]*GK_localL[0]+
		      y*GK_localL[0]+
		      x)/2)*4*3*2 + mu*3*2 + c1*2 ;
	  tmp0 = (h_elem[ivec*total_length_per_NeV + pos + 0] * phaseRe - 
		  h_elem[ivec*total_length_per_NeV + pos + 1] * phaseIm);
	  tmp1 = (h_elem[ivec*total_length_per_NeV + pos + 0] * phaseIm + 
		  h_elem[ivec*total_length_per_NeV + pos + 1] * phaseRe);
	  h_elem[ivec*total_length_per_NeV + pos + 0] = tmp0;
	  h_elem[ivec*total_length_per_NeV + pos + 1] = tmp1;
	}
      }
  }
  else{
    for(int ivec = 0 ; ivec < NeV ; ivec++)
    for(int t=0; t<GK_localL[3];t++)
    for(int z=0; z<GK_localL[2];z++)
    for(int y=0; y<GK_localL[1];y++)
    for(int x=0; x<GK_localL[0];x++)
    for(int mu=0; mu<4; mu++)
    for(int c1=0; c1<3; c1++){
      phaseRe = cos(PI*(t+comm_coords(default_topo)[3]*GK_localL[3])/((Float) GK_totalL[3]));
      phaseIm = sin(PI*(t+comm_coords(default_topo)[3]*GK_localL[3])/((Float) GK_totalL[3]));
      int pos = (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
		 z*GK_localL[1]*GK_localL[0]+
		 y*GK_localL[0]+
		 x)*4*3*2 + mu*3*2 + c1*2 ;
      tmp0 = (h_elem[ivec*total_length_per_NeV + pos + 0] * phaseRe - 
	      h_elem[ivec*total_length_per_NeV + pos + 1] * phaseIm);
      tmp1 = (h_elem[ivec*total_length_per_NeV + pos + 0] * phaseIm + 
	      h_elem[ivec*total_length_per_NeV + pos + 1] * phaseRe);
      h_elem[ivec*total_length_per_NeV + pos + 0] = tmp0;
      h_elem[ivec*total_length_per_NeV + pos + 1] = tmp1;
    }
  }//-else

  printfQuda("Multiplication by phase completed successfully\n");
}


template<typename Float>
void QKXTM_Deflation_Kepler<Float>::readEigenVectors(char *prefix_path){
  if(NeV == 0)return;
  LimeReader *limereader;
  FILE *fid;
  char *lime_type,*lime_data;
  unsigned long int lime_data_size;
  char dummy;
  MPI_Offset offset;
  MPI_Datatype subblock;  //MPI-type, 5d subarray
  MPI_File mpifid;
  MPI_Status status;
  int sizes[5], lsizes[5], starts[5];
  unsigned int i,j;
  unsigned short int chunksize,mu,c1;
  char *buffer;
  unsigned int x,y,z,t;
  int  isDouble; // default precision
  int error_occured=0;
  int next_rec_is_prop = 0;
  char filename[257];
   
  for(int nev = 0 ; nev < NeV ; nev++){
    sprintf(filename,"%s.%05d",prefix_path,nev);
    if(comm_rank() == 0) {
      /* read lime header */
      fid=fopen(filename,"r");
      if(fid==NULL) {
	fprintf(stderr,"process 0: Error in %s Could not open %s for reading\n",__func__, filename);
	error_occured=1;
      }
      if ((limereader = limeCreateReader(fid))==NULL) {
	fprintf(stderr,"process 0: Error in %s! Could not create limeReader\n", __func__);
	error_occured=1;
      }
      if(!error_occured) {
	while(limeReaderNextRecord(limereader) != LIME_EOF ) {
	  lime_type = limeReaderType(limereader);
	  if(strcmp(lime_type,"propagator-type")==0) {
	    lime_data_size = limeReaderBytes(limereader);
	    lime_data = (char * )malloc(lime_data_size);
	    limeReaderReadData((void *)lime_data,&lime_data_size,limereader);
	    
	    if (strncmp ("DiracFermion_Source_Sink_Pairs", lime_data, 
			 strlen ("DiracFermion_Source_Sink_Pairs"))!=0 &&
		strncmp ("DiracFermion_Sink", lime_data, 
			 strlen ("DiracFermion_Sink"))!=0 ) {
	      fprintf (stderr, " process 0: Error in %s! Got %s for \"propagator-type\", expecting %s or %s\n", __func__, lime_data, 
		       "DiracFermion_Source_Sink_Pairs", 
		       "DiracFermion_Sink");
	      error_occured = 1;
	      break;
	    }
	    free(lime_data);
	  }
	  //lime_type="scidac-binary-data";
	  if((strcmp(lime_type,"etmc-propagator-format")==0) || 
	     (strcmp(lime_type,"etmc-source-format")==0) || 
	     (strcmp(lime_type,"etmc-eigenvectors-format")==0) || 
	     (strcmp(lime_type,"eigenvector-info")==0)) {
	    lime_data_size = limeReaderBytes(limereader);
	    lime_data = (char * )malloc(lime_data_size);
	    limeReaderReadData((void *)lime_data,&lime_data_size,limereader);
	    sscanf(qcd_getParam("<precision>",lime_data, 
				lime_data_size),"%i",&isDouble);    
	    //		     printf("got precision: %i\n",isDouble);
	    free(lime_data);
	    
	    next_rec_is_prop = 1;
	  }
	  if(strcmp(lime_type,"scidac-binary-data")==0 && 
	     next_rec_is_prop) {	      
	    break;
	  }
	}
	/* read 1 byte to set file-pointer to start of binary data */
	lime_data_size=1;
	limeReaderReadData(&dummy,&lime_data_size,limereader);
	offset = ftell(fid)-1;
	limeDestroyReader(limereader);      
	fclose(fid);
      }     
    }//end myid==0 condition
    
    MPI_Bcast(&error_occured,1,MPI_INT,0,MPI_COMM_WORLD);
    if(error_occured) errorQuda("Error with reading eigenVectors\n");
    //     if(isDouble != 32 && isDouble != 64 )isDouble = 32;     
    MPI_Bcast(&isDouble,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&offset,sizeof(MPI_Offset),MPI_BYTE,0,MPI_COMM_WORLD);
    
    //     printfQuda("I have precision %d\n",isDouble);
    
    if( typeid(Float) == typeid(double) ){
      if( isDouble != 64 ) errorQuda("Your precisions does not agree");
    }
    else if(typeid(Float) == typeid(float) ){
      if( isDouble != 32 ) errorQuda("Your precisions does not agree");
    }
    else
      errorQuda("Problem with the precision\n");

    if(isDouble==64)
      isDouble=1;      
    else if(isDouble==32)
      isDouble=0; 
    else
      {
	fprintf(stderr,"process %i: Error in %s! Unsupported precision\n",
		comm_rank(), __func__);
      }  
     
    if(isDouble)
      {

	sizes[0] = GK_totalL[3];
	sizes[1] = GK_totalL[2];
	sizes[2] = GK_totalL[1];
	sizes[3] = GK_totalL[0];
	sizes[4] = (4*3*2);
	 
	lsizes[0] = GK_localL[3];
	lsizes[1] = GK_localL[2];
	lsizes[2] = GK_localL[1];
	lsizes[3] = GK_localL[0];
	lsizes[4] = sizes[4];
	 
	starts[0]      = comm_coords(default_topo)[3]*GK_localL[3];
	starts[1]      = comm_coords(default_topo)[2]*GK_localL[2];
	starts[2]      = comm_coords(default_topo)[1]*GK_localL[1];
	starts[3]      = comm_coords(default_topo)[0]*GK_localL[0];
	starts[4]      = 0;


	 
	MPI_Type_create_subarray(5,sizes,lsizes,starts,MPI_ORDER_C,
				 MPI_DOUBLE,&subblock);
	MPI_Type_commit(&subblock);
      
	MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, 
		      MPI_INFO_NULL, &mpifid);
	MPI_File_set_view(mpifid, offset, MPI_DOUBLE, subblock, 
			  "native", MPI_INFO_NULL);
	 
	//load time-slice by time-slice:
	chunksize=4*3*2*sizeof(double);
	buffer = (char*) malloc(chunksize*GK_localVolume);
	if(buffer==NULL)
	  {
	    fprintf(stderr,"process %i: Error in %s! Out of memory\n",
		    comm_rank(), __func__);
	    return;
	  }
	MPI_File_read_all(mpifid, buffer, 4*3*2*GK_localVolume, 
			  MPI_DOUBLE, &status);
	if(!qcd_isBigEndian())      
	  qcd_swap_8((double*)buffer,(size_t)(2*4*3)*(size_t)GK_localVolume);
	i=0;
	if(!isFullOp){
	  for(t=0; t<GK_localL[3];t++)
	    for(z=0; z<GK_localL[2];z++)
	      for(y=0; y<GK_localL[1];y++)
		for(x=0; x<GK_localL[0];x++)
		  for(mu=0; mu<4; mu++)
		    for(c1=0; c1<3; c1++){
		      int oddBit     = (x+y+z+t) & 1;
		      if(oddBit){
			h_elem[nev*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
				 z*GK_localL[1]*GK_localL[0]+
				 y*GK_localL[0]+
				 x)/2)*4*3*2 + 
			       mu*3*2 + c1*2 + 0 ] = ((double*)buffer)[i];
			h_elem[nev*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
				 z*GK_localL[1]*GK_localL[0]+
				 y*GK_localL[0]+
				 x)/2)*4*3*2 + 
			       mu*3*2 + c1*2 + 1 ] = ((double*)buffer)[i+1];
			i+=2;
		      }
		      else{
			i+=2;
		      }
		    }
	}
	else{
	  for(t=0; t<GK_localL[3];t++)
	    for(z=0; z<GK_localL[2];z++)
	      for(y=0; y<GK_localL[1];y++)
		for(x=0; x<GK_localL[0];x++)
		  for(mu=0; mu<4; mu++)
		    for(c1=0; c1<3; c1++){
		      h_elem[nev*total_length_per_NeV + 
			     (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
			      z*GK_localL[1]*GK_localL[0]+
			      y*GK_localL[0]+
			      x)*4*3*2 + 
			     mu*3*2 + c1*2 + 0 ] = ((double*)buffer)[i];
		      h_elem[nev*total_length_per_NeV + 
			     (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
			      z*GK_localL[1]*GK_localL[0]+
			      y*GK_localL[0]+
			      x)*4*3*2 + 
			     mu*3*2 + c1*2 + 1 ] = ((double*)buffer)[i+1];
		      i+=2;
		    }
	}


	free(buffer);
	MPI_File_close(&mpifid);
	MPI_Type_free(&subblock);
	 
	continue;
      }//end isDouble condition
    else
      {
	sizes[0] = GK_totalL[3];
	sizes[1] = GK_totalL[2];
	sizes[2] = GK_totalL[1];
	sizes[3] = GK_totalL[0];
	sizes[4] = (4*3*2);
	 
	lsizes[0] = GK_localL[3];
	lsizes[1] = GK_localL[2];
	lsizes[2] = GK_localL[1];
	lsizes[3] = GK_localL[0];
	lsizes[4] = sizes[4];
	 
	starts[0]      = comm_coords(default_topo)[3]*GK_localL[3];
	starts[1]      = comm_coords(default_topo)[2]*GK_localL[2];
	starts[2]      = comm_coords(default_topo)[1]*GK_localL[1];
	starts[3]      = comm_coords(default_topo)[0]*GK_localL[0];
	starts[4]      = 0;

	//	 for(int ii = 0 ; ii < 5 ; ii++)
	//  printf("%d %d %d %d\n",comm_rank(),sizes[ii],lsizes[ii],starts[ii]);

	MPI_Type_create_subarray(5,sizes,lsizes,starts,MPI_ORDER_C,
				 MPI_FLOAT,&subblock);
	MPI_Type_commit(&subblock);
      
	MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, 
		      MPI_INFO_NULL, &mpifid);
	MPI_File_set_view(mpifid, offset, MPI_FLOAT, subblock, 
			  "native", MPI_INFO_NULL);
      
	//load time-slice by time-slice:
	chunksize=4*3*2*sizeof(float);
	buffer = (char*) malloc(chunksize*GK_localVolume);
	if(buffer==NULL)
	  {
	    fprintf(stderr,"process %i: Error in %s! Out of memory\n",
		    comm_rank(), __func__);
	    return;
	  }
	MPI_File_read_all(mpifid, buffer, 4*3*2*GK_localVolume, 
			  MPI_FLOAT, &status);

	if(!qcd_isBigEndian())
	  qcd_swap_4((float*) buffer,(size_t)(2*4*3)*(size_t)GK_localVolume);
      
	i=0;
	if(!isFullOp){
	  for(t=0; t<GK_localL[3];t++)
	    for(z=0; z<GK_localL[2];z++)
	      for(y=0; y<GK_localL[1];y++)
		for(x=0; x<GK_localL[0];x++)
		  for(mu=0; mu<4; mu++)
		    for(c1=0; c1<3; c1++){
		      int oddBit     = (x+y+z+t) & 1;
		      if(oddBit){
			h_elem[nev*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
				 z*GK_localL[1]*GK_localL[0]+
				 y*GK_localL[0]+x)/2)*4*3*2 + 
			       mu*3*2 + c1*2 + 0 ] = 
			  *((float*)(buffer + i)); i+=4;
			h_elem[nev*total_length_per_NeV + 
			       ((t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
				 z*GK_localL[1]*GK_localL[0]+
				 y*GK_localL[0]+
				 x)/2)*4*3*2 + 
			       mu*3*2 + c1*2 + 1 ] = 
			  *((float*)(buffer + i)); i+=4;
		      }
		      else{
			i+=8;
		      }
		    }    
	}  
	else{
	  for(t=0; t<GK_localL[3];t++)
	    for(z=0; z<GK_localL[2];z++)
	      for(y=0; y<GK_localL[1];y++)
		for(x=0; x<GK_localL[0];x++)
		  for(mu=0; mu<4; mu++)
		    for(c1=0; c1<3; c1++){
		      h_elem[nev*total_length_per_NeV + 
			     (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
			      z*GK_localL[1]*GK_localL[0]+
			      y*GK_localL[0]+
			      x)*4*3*2 + 
			     mu*3*2 + c1*2 + 0 ] = 
			*((float*)(buffer + i)); i+=4;
		      h_elem[nev*total_length_per_NeV + 
			     (t*GK_localL[2]*GK_localL[1]*GK_localL[0]+
			      z*GK_localL[1]*GK_localL[0]+
			      y*GK_localL[0]+
			      x)*4*3*2 + mu*3*2 + c1*2 + 1 ] = 
			*((float*)(buffer + i)); i+=4;
		    }    
	}
      
	free(buffer);
	MPI_File_close(&mpifid);
	MPI_Type_free(&subblock);            
      
	continue;
      }//end isDouble condition
  }
  printfQuda("Eigenvectors loaded successfully\n");
}//end qcd_getVectorLime 


template<typename Float>
void QKXTM_Deflation_Kepler<Float>::readEigenValues(char *filename){
  if(NeV == 0)return;
  FILE *ptr;
  Float dummy;
  ptr = fopen(filename,"r");
  if(ptr == NULL)errorQuda("Error cannot open file to read eigenvalues\n");
  char stringFormat[257];
  if(typeid(Float) == typeid(double))
    strcpy(stringFormat,"%lf");
  else if(typeid(Float) == typeid(float))
    strcpy(stringFormat,"%f");

  for(int i = 0 ; i < NeV ; i++){
    fscanf(ptr,stringFormat,&(EigenValues()[2*i]),&dummy);
    EigenValues()[2*i+1] = 0.0;
  }

  printfQuda("Eigenvalues loaded successfully\n");
  fclose(ptr);
}

//-C.K: This member function performs the operation 
// vec_defl = vec_in - (U U^dag) vec_in
template <typename Float>
void QKXTM_Deflation_Kepler<Float>::
projectVector(QKXTM_Vector_Kepler<Float> &vec_defl, 
	      QKXTM_Vector_Kepler<Float> &vec_in, 
	      int is){
  
  if(!isFullOp) errorQuda("projectVector: This function only works with the Full Operator\n");
  
  if(NeV == 0){
    printfQuda("NeV = %d. Will not deflate source vector!!!\n",NeV);
    vec_defl.packVector((Float*) vec_in.H_elem());
    vec_defl.loadVector();

    return;
  }

  Float *ptr_elem = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;
  Float *tmp_vec  = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;

  Float *out_vec        = (Float*) calloc(NeV*2,sizeof(Float)) ;
  Float *out_vec_reduce = (Float*) calloc(NeV*2,sizeof(Float)) ;
  
  if(ptr_elem == NULL || 
     tmp_vec == NULL || 
     out_vec == NULL || 
     out_vec_reduce == NULL) 
    errorQuda("projectVector: Error with memory allocation\n");
  
  Float alpha[2] = {1.,0.};
  Float beta[2] = {0.,0.};
  Float al[2] = {-1.0,0.0};
  int incx = 1;
  int incy = 1;
  long int NN = (GK_localVolume/fullorHalf)*4*3;
  
  //-C.K. tmp_vec = vec_in
  memcpy(tmp_vec,vec_in.H_elem(),bytes_total_length_per_NeV); 
  memset(out_vec,0,NeV*2*sizeof(Float));
  memset(out_vec_reduce,0,NeV*2*sizeof(Float));
  memset(ptr_elem,0,NN*2*sizeof(Float));

  if( typeid(Float) == typeid(float) ){
    
    //-C.K: out_vec_reduce = h_elem^dag * tmp_vec -> U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasConjTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, 
		tmp_vec, incx, (void*) beta, out_vec, incy );
    
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_FLOAT,MPI_SUM,
		  MPI_COMM_WORLD); 
    
    //-C.K: ptr_elem = h_elem * out_vec_reduce -> ptr_elem = U*U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasNoTrans, NN, NeV, (void*) alpha, 
		(void*) h_elem, NN, out_vec_reduce, incx, 
		(void*) beta, ptr_elem, incy );
    
    //-C.K. tmp_vec = -1.0*ptr_elem + tmp_vec -> 
    //       tmp_vec = vec_in - U*U^dag * vec_in
    cblas_caxpy (NN, (void*)al, (void*)ptr_elem, incx, (void*)tmp_vec, incy);
  }
  else if( typeid(Float) == typeid(double) ){
    cblas_zgemv(CblasColMajor, CblasConjTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, 
		tmp_vec, incx, (void*) beta, out_vec, incy );
    
    MPI_Allreduce(out_vec,out_vec_reduce,NeV*2,MPI_DOUBLE,MPI_SUM,
		  MPI_COMM_WORLD);
    
    cblas_zgemv(CblasColMajor, CblasNoTrans, NN, NeV, 
		(void*) alpha, (void*) h_elem, NN, 
		out_vec_reduce, incx, (void*) beta, ptr_elem, incy );
    
    cblas_zaxpy (NN, (void*)al, (void*)ptr_elem, incx, (void*)tmp_vec, incy);
  }
  

  //   Float udotb[2];
  //   Float udotb_reduce[2];
  //   Float *tmp_vec2 = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;

  //   char fbase[257];
  
  //   if(tmp_vec2 == NULL)errorQuda("projectVector: Error with memory allocation\n");

  //   memcpy(tmp_vec,vec_in.H_elem(),bytes_total_length_per_NeV); //-C.K. tmp_vec = vec_in
  //   memset(tmp_vec2,0,NN*2*sizeof(Float));

  //   //  dumpVector(tmp_vec,is,"MdagSource");

  //   if( typeid(Float) == typeid(float) ){
  //     for(int iv = 0;iv<NeV;iv++){
  //       memcpy(ptr_elem,&(h_elem[iv*total_length_per_NeV]),bytes_total_length_per_NeV);  //-C.K.: ptr_elem = eVec[iv]

  //       cblas_cdotc_sub(NN, ptr_elem, incx, tmp_vec, incy, udotb); 
  //       MPI_Allreduce(udotb,udotb_reduce,2,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);  //-C.K.: udotb_reduce = evec[iv]^dag * vec_in
  //       printfQuda("evec[%d]^dag * vec_in = %16.15e + i %16.15e\n",iv,udotb_reduce[0],udotb_reduce[1]); 

  //       cblas_caxpy (NN, (void*) udotb_reduce, (void*) ptr_elem, incx, (void*) tmp_vec2, incy);  //-C.K.: tmp_vec2 = (evec[iv]^dag * vec_in)* eVec[iv] + tmp_vec2
  //       //      sprintf(fbase,"scalarDoteVec_%03d",iv);
  //       //      dumpVector(tmp_vec2,is,fbase);
  //     }
  //     cblas_caxpy (NN, (void*) al, (void*) tmp_vec2, incx, (void*) tmp_vec, incy);
  //   }
  //   else if( typeid(Float) == typeid(double) ){
  //     for(int iv = 0;iv<NeV;iv++){
  //       memcpy(ptr_elem,&(h_elem[iv*total_length_per_NeV]),bytes_total_length_per_NeV);  //-C.K.: ptr_elem = eVec[iv]

  //       cblas_zdotc_sub(NN, ptr_elem, incx, tmp_vec, incy, udotb); 
  //       MPI_Allreduce(udotb,udotb_reduce,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  //-C.K.: udotb_reduce = evec[iv]^dag * vec_in
  //       printfQuda("*** projectVector: evec[%d]^dag * vec_in = %16.15e + i %16.15e\n",iv,udotb_reduce[0],udotb_reduce[1]); 

  //       cblas_zaxpy (NN, (void*) udotb_reduce, (void*) ptr_elem, incx, (void*) tmp_vec2, incy);  //-C.K.: tmp_vec2 = (evec[iv]^dag * vec_in)* eVec[iv] + tmp_vec2
  //       //      sprintf(fbase,"scalarDoteVec_%03d",iv);
  //       //      dumpVector(tmp_vec2,is,fbase);
  //     }    
  //     cblas_zaxpy (NN, (void*) al, (void*) tmp_vec2, incx, (void*) tmp_vec, incy);  //-C.K.: tmp_vec = tmp_vec - tmp_vec2 = vec_in  - UU^dag * vec_in
  //   }
  //   free(tmp_vec2);

  //  dumpVector(tmp_vec,is,"deflatedSource");

  vec_defl.packVector((Float*) tmp_vec);
  vec_defl.loadVector();

  free(ptr_elem);
  free(tmp_vec);
  free(out_vec);
  free(out_vec_reduce);

  printfQuda("projectVector: Deflation of the source vector completed succesfully\n");
}

//-C.K: This member function performs the operation 
//vec_defl = vec_in - (U U^dag) vec_in
template <typename Float>
void QKXTM_Deflation_Kepler<Float>::
projectVector(QKXTM_Vector_Kepler<Float> &vec_defl, 
	      QKXTM_Vector_Kepler<Float> &vec_in, 
	      int is, int NeV_defl){
  
  if(!isFullOp) errorQuda("projectVector: This function only works with the Full Operator\n");
  
  if(NeV_defl == 0){
    printfQuda("NeV = %d. Will not deflate source vector!\n",NeV_defl);
    vec_defl.packVector((Float*) vec_in.H_elem());
    vec_defl.loadVector();
    return;
  }

  Float *ptr_elem = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;
  Float *tmp_vec  = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;

  Float *out_vec        = (Float*) calloc(NeV_defl*2,sizeof(Float)) ;
  Float *out_vec_reduce = (Float*) calloc(NeV_defl*2,sizeof(Float)) ;
  
  if(ptr_elem == NULL || 
     tmp_vec == NULL || 
     out_vec == NULL || 
     out_vec_reduce == NULL) 
    errorQuda("projectVector: Error with memory allocation\n");
  
  Float alpha[2] = {1.,0.};
  Float beta[2] = {0.,0.};
  Float al[2] = {-1.0,0.0};
  int incx = 1;
  int incy = 1;
  long int NN = (GK_localVolume/fullorHalf)*4*3;
  
  //-C.K. tmp_vec = vec_in
  memcpy(tmp_vec,vec_in.H_elem(),bytes_total_length_per_NeV);
  memset(out_vec,0,NeV_defl*2*sizeof(Float));
  memset(out_vec_reduce,0,NeV_defl*2*sizeof(Float));
  memset(ptr_elem,0,NN*2*sizeof(Float));

  if( typeid(Float) == typeid(float) ){
    //-C.K: out_vec_reduce = h_elem^dag * tmp_vec -> U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasConjTrans, NN, NeV_defl, 
		(void*) alpha, (void*) h_elem, 
		NN, tmp_vec, incx, (void*) beta, out_vec, incy );  
    MPI_Allreduce(out_vec,out_vec_reduce,NeV_defl*2,MPI_FLOAT,
		  MPI_SUM,MPI_COMM_WORLD);
    
    //-C.K: ptr_elem = h_elem * out_vec_reduce -> ptr_elem = U*U^dag * vec_in
    cblas_cgemv(CblasColMajor, CblasNoTrans, NN, NeV_defl, 
		(void*) alpha, (void*) h_elem, NN,
		out_vec_reduce, incx, (void*) beta, ptr_elem, incy );  
    
    //-C.K. tmp_vec = -1.0*ptr_elem + tmp_vec -> 
    //      tmp_vec = vec_in - U*U^dag * vec_in
    cblas_caxpy (NN, (void*)al, (void*)ptr_elem, incx, (void*)tmp_vec, incy);
  }
  else if( typeid(Float) == typeid(double) ){
    cblas_zgemv(CblasColMajor, CblasConjTrans, NN, NeV_defl, 
		(void*) alpha, (void*) h_elem, 
		NN, tmp_vec, incx, (void*) beta, out_vec, incy );
    MPI_Allreduce(out_vec,out_vec_reduce,NeV_defl*2,MPI_DOUBLE,MPI_SUM,
		  MPI_COMM_WORLD);

    cblas_zgemv(CblasColMajor, CblasNoTrans, NN, NeV_defl, 
		(void*) alpha, (void*) h_elem, NN,
		out_vec_reduce, incx, (void*) beta, ptr_elem, incy );    
    
    cblas_zaxpy (NN, (void*)al, (void*)ptr_elem, incx, (void*)tmp_vec, incy);
  }

  
  //   Float udotb[2];
  //   Float udotb_reduce[2];
  //   Float *tmp_vec2 = (Float*) calloc((GK_localVolume)*4*3*2,sizeof(Float)) ;

  //   char fbase[257];
  
  //   if(tmp_vec2 == NULL)errorQuda("projectVector: Error with memory allocation\n");

  //   memcpy(tmp_vec,vec_in.H_elem(),bytes_total_length_per_NeV); //-C.K. tmp_vec = vec_in
  //   memset(tmp_vec2,0,NN*2*sizeof(Float));

  //   //  dumpVector(tmp_vec,is,"MdagSource");

  //   if( typeid(Float) == typeid(float) ){
  //     for(int iv = 0;iv<NeV_defl;iv++){
  //       memcpy(ptr_elem,&(h_elem[iv*total_length_per_NeV_defl]),bytes_total_length_per_NeV_defl);  //-C.K.: ptr_elem = eVec[iv]

  //       cblas_cdotc_sub(NN, ptr_elem, incx, tmp_vec, incy, udotb); 
  //       MPI_Allreduce(udotb,udotb_reduce,2,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);  //-C.K.: udotb_reduce = evec[iv]^dag * vec_in
  //       printfQuda("evec[%d]^dag * vec_in = %16.15e + i %16.15e\n",iv,udotb_reduce[0],udotb_reduce[1]); 

  //       cblas_caxpy (NN, (void*) udotb_reduce, (void*) ptr_elem, incx, (void*) tmp_vec2, incy);  //-C.K.: tmp_vec2 = (evec[iv]^dag * vec_in)* eVec[iv] + tmp_vec2
  //       //      sprintf(fbase,"scalarDoteVec_%03d",iv);
  //       //      dumpVector(tmp_vec2,is,fbase);
  //     }
  //     cblas_caxpy (NN, (void*) al, (void*) tmp_vec2, incx, (void*) tmp_vec, incy);
  //   }
  //   else if( typeid(Float) == typeid(double) ){
  //     for(int iv = 0;iv<NeV_defl;iv++){
  //       memcpy(ptr_elem,&(h_elem[iv*total_length_per_NeV_defl]),bytes_total_length_per_NeV_defl);  //-C.K.: ptr_elem = eVec[iv]

  //       cblas_zdotc_sub(NN, ptr_elem, incx, tmp_vec, incy, udotb); 
  //       MPI_Allreduce(udotb,udotb_reduce,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  //-C.K.: udotb_reduce = evec[iv]^dag * vec_in
  //       printfQuda("*** projectVector: evec[%d]^dag * vec_in = %16.15e + i %16.15e\n",iv,udotb_reduce[0],udotb_reduce[1]); 

  //       cblas_zaxpy (NN, (void*) udotb_reduce, (void*) ptr_elem, incx, (void*) tmp_vec2, incy);  //-C.K.: tmp_vec2 = (evec[iv]^dag * vec_in)* eVec[iv] + tmp_vec2
  //       //      sprintf(fbase,"scalarDoteVec_%03d",iv);
  //       //      dumpVector(tmp_vec2,is,fbase);
  //     }    
  //     cblas_zaxpy (NN, (void*) al, (void*) tmp_vec2, incx, (void*) tmp_vec, incy);  //-C.K.: tmp_vec = tmp_vec - tmp_vec2 = vec_in  - UU^dag * vec_in
  //   }
  //   free(tmp_vec2);

  //  dumpVector(tmp_vec,is,"deflatedSource");

  vec_defl.packVector((Float*) tmp_vec);
  vec_defl.loadVector();

  free(ptr_elem);
  free(tmp_vec);
  free(out_vec);
  free(out_vec_reduce);

  printfQuda("projectVector: Deflation of the source vector completed succesfully\n");
}

