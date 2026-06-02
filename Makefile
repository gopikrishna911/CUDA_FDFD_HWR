# ============================================================
# Rhodotron FDFD eigensolver — CUDA-accelerated build
# ============================================================
# Binaries built (in this directory):
#   solve_full_model     production solve (PEC + conformal IBC)
#   manuscript_3stage    three-stage manuscript table generator
#   convergence_study    grid-convergence diagnostic
#
# Quick start:
#   make                       build all three binaries
#   make solve_full_model      production binary only
#   make clean
#
# Override your GPU's compute capability if needed:
#   make CUDA_ARCH=sm_89
# ============================================================

CXX          = g++
NVCC         = nvcc
CXXFLAGS     = -Wall -Wextra -O2 -std=c++11
CUDA_ARCH   ?= sm_86
NVCCFLAGS    = -O3 -arch=$(CUDA_ARCH) --use_fast_math
LDFLAGS      = -lm
CUDA_LDFLAGS = -lcudart -lm
CUDA_PATH   ?= /usr/local/cuda
CUDA_INC     = -I$(CUDA_PATH)/include -Iinclude
CUDA_LIB     = -L$(CUDA_PATH)/lib64

INC_DIR  = include
SRC_DIR  = src
APP_DIR  = apps
TEST_DIR = tests

# Shared library objects (used by every binary)
OBJS_LIB = curl_E.o curl_H.o curlcurl_operator.o \
           pipe_model.o conformal_geometry.o \
           q_factor.o r_over_q.o field_export.o field_map_export.o \
           cuda_fields.o cuda_vector_ops.o cuda_curls.o \
           cuda_operator.o cuda_eigensolver.o cuda_pipe_model.o \
           cuda_conformal_curls.o cuda_conformal_pipe.o

# Source search paths for pattern rules
vpath %.cpp $(SRC_DIR)
vpath %.cu  $(SRC_DIR)

.PHONY: all apps tests clean
all: solve_full_model manuscript_3stage convergence_study
apps: solve_full_model manuscript_3stage
tests: convergence_study

# ----- Binary link rules -----
solve_full_model: $(OBJS_LIB) solve_full_model.o
	$(NVCC) $^ -o $@ $(CUDA_LIB) $(CUDA_LDFLAGS)

manuscript_3stage: $(OBJS_LIB) manuscript_3stage.o
	$(NVCC) $^ -o $@ $(CUDA_LIB) $(CUDA_LDFLAGS)

convergence_study: $(OBJS_LIB) convergence_study.o
	$(NVCC) $^ -o $@ $(CUDA_LIB) $(CUDA_LDFLAGS)

# ----- Library compile rules (vpath finds sources in src/) -----
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CUDA_INC) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c $< -o $@

# ----- App driver compile rules (explicit; -x cu for CUDA API) -----
solve_full_model.o: $(APP_DIR)/solve_full_model.cpp
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -x cu -c $< -o $@

manuscript_3stage.o: $(APP_DIR)/manuscript_3stage.cpp
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -x cu -c $< -o $@

# ----- Convergence test compile rule -----
convergence_study.o: $(TEST_DIR)/convergence_study.cpp
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -x cu -c $< -o $@

# ----- Clean -----
clean:
	rm -f *.o solve_full_model manuscript_3stage convergence_study