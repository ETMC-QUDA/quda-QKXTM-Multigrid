
include_directories(.)
include_directories(../lib)
include_directories(../include)
include_directories(../tests)
include_directories(../lib/code_pieces_Kepler)
include_directories(${QUDA_QKXTM_HDF5HOME}/include)
include_directories(${QUDA_QKXTM_LIMEHOME}/include)
include_directories(SYSTEM ${QUDA_QKXTM_GSLHOME}/include)
if(QUDA_QKXTM_ARPACK)
  include_directories(SYSTEM ${QUDA_QKXTM_ARPACKHOME}/PARPACK/SRC/MPI)
  include_directories(SYSTEM ${QUDA_QKXTM_ARPACKHOME}/PARPACK/UTIL/MPI)
  include_directories(SYSTEM ${QUDA_QKXTM_ARPACKHOME}/SRC)
  include_directories(SYSTEM ${QUDA_QKXTM_ARPACKHOME}/UTIL)
endif(QUDA_QKXTM_ARPACK)  

#build a common library for all test utilities
set(QKXTM_COMMON gtest-all.cc QKXTM_util.cpp misc.cpp)
cuda_add_library(qkxtm STATIC ${QKXTM_COMMON})
set(QKXTM_LIBS quda qkxtm ${CMAKE_THREAD_LIBS_INIT} ${QUDA_LIBS} cuda)

if(QUDA_QIO)
  LIST(APPEND QKXTM_LIBS ${QIO_LIB} ${LIME_LIB})
endif()

if(QUDA_QMP)
  LIST(APPEND QKXTM_LIBS ${QMP_LIB} ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MPI)
  LIST(APPEND QKXTM_LIBS ${MPI_CXX_LIBRARIES})
endif()

if(QUDA_MAGMA)
  find_library(MAGMA ${MAGMA_LIBRARIES} PATH ${MAGMA_LIBRARY_DIRS})
  LIST(APPEND QKXTM_LIBS ${MAGMA})
endif()

if(QUDA_NUMA_NVML)
  LIST(APPEND QKXTM_LIBS ${NVML_LIBRARY})
endif()

LIST(APPEND QKXTM_LIBS ${QKXTM_GSL_LIB})
LIST(APPEND QKXTM_LIBS ${QKXTM_HDF5_LIB})
LIST(APPEND QKXTM_LIBS ${QKXTM_LIME_LIB})
if(QUDA_QKXTM_ARPACK)
  LIST(APPEND QKXTM_LIBS ${QKXTM_PARPACK_LIB})
  LIST(APPEND QKXTM_LIBS ${QKXTM_ARPACK_LIB})
endif(QUDA_QKXTM_ARPACK)

#define QKXTM executables

#CalcMG_2pt3pt_EvenOdd
# This function will calculate hadronic correlation functions using a 
# handful of randomly separated, gaussian smeared point sources, constructed 
# with a smeared gauge field. Even/Odd preconditioning is employed, as well
# as QUDA's MG solver routines.
cuda_add_executable(CalcMG_2pt3pt_EvenOdd CalcMG_2pt3pt_EvenOdd.cpp ../tests/wilson_dslash_reference.cpp ../tests/domain_wall_dslash_reference.cpp ../tests/clover_reference.cpp ../tests/blas_reference.cpp)
target_link_libraries(CalcMG_2pt3pt_EvenOdd ${QKXTM_LIBS})

if(QUDA_QKXTM_ARPACK)
#CalcMG_Loops_w_oneD_TSM_EvenOdd
# This function will calculate the All-to-All propagator with a covariant 
# derivate, and correlation functions thereof. It employs undiluted 
# stochastic sources, Even/Odd preconditioning, functionality for using the 
# Trunacted Solver Method, as well as QUDA's MG solver routines.
cuda_add_executable(CalcMG_Loops_w_oneD_TSM_EvenOdd CalcMG_Loops_w_oneD_TSM_EvenOdd.cpp ../tests/wilson_dslash_reference.cpp ../tests/domain_wall_dslash_reference.cpp ../tests/clover_reference.cpp ../tests/blas_reference.cpp)
target_link_libraries(CalcMG_Loops_w_oneD_TSM_EvenOdd ${QKXTM_LIBS})

#CalcMG_Loops_w_oneD_TSM_wExact
# This function will calculate the All-to-All propagator with a covariant 
# derivate, and correlation functions thereof. It calculates the eigenvectors
# of the full operator and the corresponding correlations from that part.
# It then projects out that exact part and employs undiluted 
# stochastic sources, Even/Odd preconditioning, functionality for using the 
# Trunacted Solver Method, as well as QUDA's MG solver routines for the 
# remainder.
  cuda_add_executable(CalcMG_Loops_w_oneD_TSM_wExact CalcMG_Loops_w_oneD_TSM_wExact.cpp ../tests/wilson_dslash_reference.cpp ../tests/domain_wall_dslash_reference.cpp ../tests/clover_reference.cpp ../tests/blas_reference.cpp)
  target_link_libraries(CalcMG_Loops_w_oneD_TSM_wExact ${QKXTM_LIBS})
endif(QUDA_QKXTM_ARPACK)
