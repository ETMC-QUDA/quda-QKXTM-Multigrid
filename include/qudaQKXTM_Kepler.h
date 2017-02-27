#include <comm_quda.h>
#include <quda_internal.h>
#include <quda.h>
#include <iostream>
#include <complex>
#include <cuda.h>
#include <color_spinor_field.h>
#include <enum_quda.h>
#include <typeinfo>
//DMH 
#include <qudaQKXTM_Kepler_utils.h>
#include <dirac_quda.h>

#ifndef _QUDAQKXTM_KEPLER_H
#define _QUDAQKXTM_KEPLER_H

// function for writing
extern "C"{
#include <lime.h>
}

static void qcd_swap_4(float *Rd, size_t N)
{
  register char *i,*j,*k;
  char swap;
  char *max;
  char *R =(char*) Rd;

  max = R+(N<<2);
  for(i=R;i<max;i+=4)
    {
      j=i; k=j+3;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
    }
}


static void qcd_swap_8(double *Rd, int N)
{
  register char *i,*j,*k;
  char swap;
  char *max;
  char *R = (char*) Rd;

  max = R+(N<<3);
  for(i=R;i<max;i+=8)
    {
      j=i; k=j+7;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
      j++; k--;
      swap = *j; *j = *k;  *k = swap;
    }
}

static int qcd_isBigEndian()
{
  union{
    char C[4];
    int  R   ;
  }word;
  word.R=1;
  if(word.C[3]==1) return 1;
  if(word.C[0]==1) return 0;

  return -1;
}

static char* qcd_getParam(char token[],char* params,int len)
{
  int i,token_len=strlen(token);

  for(i=0;i<len-token_len;i++)
    {
      if(memcmp(token,params+i,token_len)==0)
	{
	  i+=token_len;
	  *(strchr(params+i,'<'))='\0';
	  break;
	}
    }
  return params+i;
}

namespace quda {
  
  // forward declaration
  template<typename Float>  class QKXTM_Field_Kepler;
  template<typename Float>  class QKXTM_Gauge_Kepler;
  template<typename Float>  class QKXTM_Vector_Kepler;
  template<typename Float>  class QKXTM_Propagator_Kepler;
  template<typename Float>  class QKXTM_Propagator3D_Kepler;
  template<typename Float>  class QKXTM_Vector3D_Kepler;
  
  ////////////////////////
  // CLASS: QKXTM_Field //
  ////////////////////////
  
  template<typename Float>
    class QKXTM_Field_Kepler {
    // base class use only for inheritance not polymorphism
  protected:
    
    int field_length;
    int total_length;        
    int ghost_length;
    int total_plus_ghost_length;

    size_t bytes_total_length;
    size_t bytes_ghost_length;
    size_t bytes_total_plus_ghost_length;

    Float *h_elem;
    Float *h_elem_backup;
    Float *d_elem;
    Float *h_ext_ghost;

    bool isAllocHost;
    bool isAllocDevice;
    bool isAllocHostBackup;

    void create_host();
    void create_host_backup();
    void destroy_host();
    void destroy_host_backup();
    void create_device();
    void destroy_device();

  public:
    QKXTM_Field_Kepler(ALLOCATION_FLAG alloc_flag,CLASS_ENUM classT);
    virtual ~QKXTM_Field_Kepler();
    void zero_host();
    void zero_host_backup();
    void zero_device();
    void createTexObject(cudaTextureObject_t *tex);
    void destroyTexObject(cudaTextureObject_t tex);
    
    Float* H_elem() const { return h_elem; }
    Float* D_elem() const { return d_elem; }

    size_t Bytes_total() const { return bytes_total_length; }
    size_t Bytes_ghost() const { return bytes_ghost_length; }
    size_t Bytes_total_plus_ghost() const { return bytes_total_plus_ghost_length; }
    
    int Precision() const{
      if( typeid(Float) == typeid(float) )
	return 4;
      else if( typeid(Float) == typeid(double) )
	return 8;
      else
	return 0;
    } 
    void printInfo();
  };

  ////////////////////////
  // CLASS: QKXTM_Gauge //
  ////////////////////////
  
  template<typename Float>
    class QKXTM_Gauge_Kepler : public QKXTM_Field_Kepler<Float> {
  public:
    QKXTM_Gauge_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT);
    ~QKXTM_Gauge_Kepler(){;}
    
    void packGauge(void **gauge);
    void packGaugeToBackup(void **gauge);
    void loadGaugeFromBackup();
    void justDownloadGauge();
    void loadGauge();
    
    void ghostToHost();
    void cpuExchangeGhost();
    void ghostToDevice();
    void calculatePlaq();
    
  };
  
  /////////////////////////
  // Class: QKXTM_Vector //
  /////////////////////////
  
  template<typename Float>
    class QKXTM_Vector_Kepler : public QKXTM_Field_Kepler<Float> {
  public:
    QKXTM_Vector_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT);
    ~QKXTM_Vector_Kepler(){;}
    
    void packVector(Float *vector);
    void unpackVector();
    void unpackVector(Float *vector);
    void loadVector();
    void unloadVector();
    void ghostToHost();
    void cpuExchangeGhost();
    void ghostToDevice();
    
    void download(); // take the vector from device to host
    void uploadToCuda( ColorSpinorField *cudaVector, bool isEv = false);
    void downloadFromCuda(ColorSpinorField *cudaVector, bool isEv = false);
    void gaussianSmearing(QKXTM_Vector_Kepler<Float> &vecIn,
			  QKXTM_Gauge_Kepler<Float> &gaugeAPE);
    void scaleVector(double a);
    void castDoubleToFloat(QKXTM_Vector_Kepler<double> &vecIn);
    void castFloatToDouble(QKXTM_Vector_Kepler<float> &vecIn);
    void norm2Host();
    void norm2Device();
    void copyPropagator3D(QKXTM_Propagator3D_Kepler<Float> &prop, 
			  int timeslice, int nu , int c2);
    void copyPropagator(QKXTM_Propagator_Kepler<Float> &prop, 
			int nu , int c2);
    void write(char* filename);
    void conjugate();
    void apply_gamma5();
  };

  ///////////////////////////
  // Class: QKXTM_Vector3D //
  ///////////////////////////
  
  template<typename Float>
    class QKXTM_Vector3D_Kepler : public QKXTM_Field_Kepler<Float> {
  public:
    QKXTM_Vector3D_Kepler();
    ~QKXTM_Vector3D_Kepler(){;}
  };
  
  /////////////////////////////
  // CLASS: QKXTM_Propagator //
  /////////////////////////////
  
  template<typename Float>
    class QKXTM_Propagator_Kepler : public QKXTM_Field_Kepler<Float> {
    
  public:
    QKXTM_Propagator_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT);
    ~QKXTM_Propagator_Kepler(){;}
    
    void ghostToHost();
    void cpuExchangeGhost();
    void ghostToDevice();
    
    void conjugate();
    void apply_gamma5();
    
    void absorbVectorToHost(QKXTM_Vector_Kepler<Float> &vec, 
			    int nu, int c2);
    void absorbVectorToDevice(QKXTM_Vector_Kepler<Float> &vec, 
			      int nu, int c2);
    void rotateToPhysicalBase_host(int sign);
    void rotateToPhysicalBase_device(int sign);
  };
  
  ///////////////////////////////
  // CLASS: QKXTM_Propagator3D //
  /////////////////////////////// 
  
  template<typename Float>
    class QKXTM_Propagator3D_Kepler : public QKXTM_Field_Kepler<Float> {
    
  public:
    QKXTM_Propagator3D_Kepler(ALLOCATION_FLAG alloc_flag, CLASS_ENUM classT);
    ~QKXTM_Propagator3D_Kepler(){;}
    
    void absorbTimeSliceFromHost(QKXTM_Propagator_Kepler<Float> &prop, 
				 int timeslice);
    void absorbTimeSlice(QKXTM_Propagator_Kepler<Float> &prop, 
			 int timeslice);
    void absorbVectorTimeSlice(QKXTM_Vector_Kepler<Float> &vec, 
			       int timeslice, int nu, int c2);
    void broadcast(int tsink);
  };
  
  //////////////////////////////
  // CLASS: QKXTM_Contraction //
  ////////////////////////////// 

  template<typename Float>
    class QKXTM_Contraction_Kepler {
  public:
   QKXTM_Contraction_Kepler(){;}
   ~QKXTM_Contraction_Kepler(){;}
   void contractMesons(QKXTM_Propagator_Kepler<Float> &prop1,
		       QKXTM_Propagator_Kepler<Float> &prop2, 
		       char *filename_out, int isource);
   void contractBaryons(QKXTM_Propagator_Kepler<Float> &prop1,
			QKXTM_Propagator_Kepler<Float> &prop2, 
			char *filename_out, int isource);
   void contractMesons(QKXTM_Propagator_Kepler<Float> &prop1,
		       QKXTM_Propagator_Kepler<Float> &prop2, 
		       void *corrMesons , int isource, CORR_SPACE CorrSpace);
   void contractBaryons(QKXTM_Propagator_Kepler<Float> &prop1,
			QKXTM_Propagator_Kepler<Float> &prop2, 
			void *corrBaryons, int isource,CORR_SPACE CorrSpace);
   
   void writeTwopBaryons_ASCII(void *corrBaryons, char *filename_out, 
			       int isource, CORR_SPACE CorrSpace);
   void writeTwopMesons_ASCII (void *corrMesons , char *filename_out, 
			       int isource, CORR_SPACE CorrSpace);
   
   void copyTwopBaryonsToHDF5_Buf(void *Twop_baryons_HDF5, void *corrBaryons,
				  int isource, CORR_SPACE CorrSpace);
   void copyTwopMesonsToHDF5_Buf (void *Twop_mesons_HDF5 , void *corrMesons, 
				  CORR_SPACE CorrSpace);
   
   void writeTwopBaryonsHDF5(void *twopBaryons, char *filename, 
			     qudaQKXTMinfo_Kepler info, int isource);
   void writeTwopMesonsHDF5 (void *twopMesons , char *filename, 
			     qudaQKXTMinfo_Kepler info, int isource);
   
   void writeTwopBaryonsHDF5_MomSpace(void *twopBaryons, char *filename, 
				      qudaQKXTMinfo_Kepler info,int isource);
   void writeTwopBaryonsHDF5_PosSpace(void *twopBaryons, char *filename, 
				      qudaQKXTMinfo_Kepler info,int isource);
   void writeTwopMesonsHDF5_MomSpace (void *twopMesons , char *filename, 
				      qudaQKXTMinfo_Kepler info,int isource);
   void writeTwopMesonsHDF5_PosSpace (void *twopMesons , char *filename, 
				      qudaQKXTMinfo_Kepler info,int isource);

   void seqSourceFixSinkPart1(QKXTM_Vector_Kepler<Float> &vec, 
			      QKXTM_Propagator3D_Kepler<Float> &prop1, 
			      QKXTM_Propagator3D_Kepler<Float> &prop2, 
			      int timeslice,int nu,int c2, 
			      WHICHPROJECTOR typeProj, 
			      WHICHPARTICLE testParticle);
   void seqSourceFixSinkPart2(QKXTM_Vector_Kepler<Float> &vec, 
			      QKXTM_Propagator3D_Kepler<Float> &prop, 
			      int timeslice,int nu,int c2, 
			      WHICHPROJECTOR typeProj, 
			      WHICHPARTICLE testParticle);
   
   void contractFixSink(QKXTM_Propagator_Kepler<Float> &seqProp, 
			QKXTM_Propagator_Kepler<Float> &prop, 
			QKXTM_Gauge_Kepler<Float> &gauge, 
			WHICHPROJECTOR typeProj,
			WHICHPARTICLE testParticle, 
			int partFlag, char *filename_out, int isource, 
			int tsinkMtsource);
   void contractFixSink(QKXTM_Propagator_Kepler<Float> &seqProp, 
			QKXTM_Propagator_Kepler<Float> &prop,
			QKXTM_Gauge_Kepler<Float> &gauge, 
			void *corrThp_local, void *corrThp_noether, 
			void *corrThp_oneD, 
			WHICHPROJECTOR typeProj, 
			WHICHPARTICLE testParticle, 
			int partFlag, int isource, CORR_SPACE CorrSpace);

   void writeThrp_ASCII(void *corrThp_local, void *corrThp_noether, 
			void *corrThp_oneD, WHICHPARTICLE testParticle, 
			int partflag , char *filename_out, int isource, 
			int tsinkMtsource, CORR_SPACE CorrSpace);
   void copyThrpToHDF5_Buf(void *Thrp_HDF5, void *corrThp,  int mu, int uORd,
			   int its, int Nsink, int pr, int thrp_sign, 
			   THRP_TYPE type, CORR_SPACE CorrSpace);
   void writeThrpHDF5(void *Thrp_local_HDF5, void *Thrp_noether_HDF5, 
		      void **Thrp_oneD_HDF5, char *filename, 
		      qudaQKXTMinfo_Kepler info, int isource, 
		      WHICHPARTICLE NUCLEON);
   void writeThrpHDF5_MomSpace(void *Thrp_local_HDF5, 
			       void *Thrp_noether_HDF5, 
			       void **Thrp_oneD_HDF5, char *filename, 
			       qudaQKXTMinfo_Kepler info, int isource, 
			       WHICHPARTICLE NUCLEON);
   void writeThrpHDF5_PosSpace(void *Thrp_local_HDF5, 
			       void *Thrp_noether_HDF5, 
			       void **Thrp_oneD_HDF5, char *filename, 
			       qudaQKXTMinfo_Kepler info, int isource, 
			       WHICHPARTICLE NUCLEON);
  };

  ////////////////////////////
  // CLASS: QKXTM_Deflation //
  //////////////////////////// 

  template<typename Float>
    class QKXTM_Deflation_Kepler{

  private:
    int field_length;
    long int total_length;
    long int total_length_per_NeV;
    
    
    int PolyDeg;
    int NeV;
    int NkV;
    // Which part of the spectrum we want to solve
    WHICHSPECTRUM spectrumPart; 
    bool isACC;
    double tolArpack;
    int maxIterArpack;
    char arpack_logfile[512];
    double amin;
    double amax;
    bool isEv;
    bool isFullOp;
    QudaTwistFlavorType flavor_sign;
    
    int fullorHalf;
    
    size_t bytes_total_length_per_NeV;
    size_t bytes_total_length;
    Float *h_elem;
    Float *eigenValues;
    void create_host();
    void destroy_host();
    
    //  DiracMdagM *matDiracOp;
    Dirac *diracOp;
    
    QudaInvertParam *invert_param;
    
  public:
    QKXTM_Deflation_Kepler(int,bool);
    QKXTM_Deflation_Kepler(QudaInvertParam*,qudaQKXTM_arpackInfo);
    
    ~QKXTM_Deflation_Kepler();
    
    void zero();
    Float* H_elem() const { return h_elem; }
    Float* EigenValues() const { return eigenValues; }
    size_t Bytes() const { return bytes_total_length; }
    size_t Bytes_Per_NeV() const { return bytes_total_length_per_NeV; }
    long int Length() const { return total_length; }
    long int Length_Per_NeV() const { return total_length_per_NeV; }
    int NeVs() const { return NeV;}
    void printInfo();
    
    // for tmLQCD conventions
    void readEigenVectors(char *filename);
    void writeEigenVectors_ASCII(char *filename);
    void readEigenValues(char *filename);
    void deflateVector(QKXTM_Vector_Kepler<Float> &vec_defl, 
		       QKXTM_Vector_Kepler<Float> &vec_in);
    void ApplyMdagM(Float *vec_out, Float *vec_in, QudaInvertParam *param);
    void MapEvenOddToFull();
    void MapEvenOddToFull(int i);
    void copyEigenVectorToQKXTM_Vector_Kepler(int eigenVector_id, 
					      Float *vec);
    void copyEigenVectorFromQKXTM_Vector_Kepler(int eigenVector_id,
						Float *vec);
    void rotateFromChiralToUKQCD();
    void multiply_by_phase();
    // for QUDA conventions
    void polynomialOperator(cudaColorSpinorField &out, 
			    const cudaColorSpinorField &in);
    void eigenSolver();
    void Loop_w_One_Der_FullOp_Exact(int n, QudaInvertParam *param, 
				     void *gen_uloc, void *std_uloc, 
				     void **gen_oneD, void **std_oneD, 
				     void **gen_csvC, void **std_csvC);
    void projectVector(QKXTM_Vector_Kepler<Float> &vec_defl, 
		       QKXTM_Vector_Kepler<Float> &vec_in, int is);
    void projectVector(QKXTM_Vector_Kepler<Float> &vec_defl, 
		       QKXTM_Vector_Kepler<Float> &vec_in, 
		       int is, int NeV_defl);
    void copyToEigenVector(Float *vec, Float *vals);
  };
  
}
// End quda namespace


//////////////////////////////////
// Multigrid Inversion Routines //
//////////////////////////////////

void calcMG_threepTwop_EvenOdd(void **gaugeSmeared, void **gauge,
			       QudaGaugeParam *gauge_param,
			       QudaInvertParam *param,
			       quda::qudaQKXTMinfo_Kepler info,
			       char *filename_twop, char *filename_threep,
			       quda::WHICHPARTICLE NUCLEON);


void calcMG_loop_wOneD_TSM_EvenOdd(void **gaugeToPlaquette, QudaInvertParam *param,
				   QudaGaugeParam *gauge_param, 
				   quda::qudaQKXTM_loopInfo loopInfo, 
				   quda::qudaQKXTMinfo_Kepler info);

///////////////////////////////////////
// MG with Exact Deflation Routines  //
///////////////////////////////////////

#ifdef HAVE_ARPACK
void calcMG_loop_wOneD_TSM_wExact(void **gaugeToPlaquette, 
				  QudaInvertParam *EVparam,
				  QudaInvertParam *param,
				  QudaGaugeParam *gauge_param, 
				  quda::qudaQKXTM_arpackInfo arpackInfo,
				  quda::qudaQKXTM_arpackInfo arpackInfoEO, 
				  quda::qudaQKXTM_loopInfo loopInfo, 
				  quda::qudaQKXTMinfo_Kepler info);
#endif

#endif
