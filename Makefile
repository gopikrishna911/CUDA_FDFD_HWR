CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -Wextra -O2 -std=c++11
NVCCFLAGS = -O3 -arch=sm_86 --use_fast_math
LDFLAGS = -lm
CUDA_LDFLAGS = -lcudart -lm

# Detect CUDA install path
CUDA_PATH ?= /usr/local/cuda
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64

# CPU Targets
TARGET_CURL = test_curl
TARGET_CURLCURL = test_curl_curl
TARGET_OPERATOR = test_operator
TARGET_RHODOTRON = test_rhodotron
TARGET_PIPE_MODEL = test_pipe_model
TARGET_CONVERGENCE = test_convergence
TARGET_CONV_FULL = test_convergence_full
TARGET_IBC = test_ibc
TARGET_IBC_PIPE = test_ibc_pipe

# CUDA Targets
TARGET_CUDA_FIELDS = test_cuda_fields
TARGET_CUDA_VECOPS = test_cuda_vector_ops
TARGET_CUDA_CURLS = test_cuda_curls
TARGET_CUDA_OPERATOR = test_cuda_operator
TARGET_CUDA_EIGEN = test_cuda_eigensolver
TARGET_RHODOTRON_GPU = test_rhodotron_gpu

# Conformal Targets
TARGET_CONFORMAL_CUDA = test_conformal_cuda
TARGET_CONFORMAL_IBC = test_conformal_ibc
TARGET_CONFORMAL_IBC_FULL = test_conformal_ibc_full

# CPU Object files
OBJS_COMMON = curl_E.o curl_H.o
OBJS_CURL = $(OBJS_COMMON) test_curl.o
OBJS_CURLCURL = $(OBJS_COMMON) test_curl_curl.o
OBJS_OPERATOR = $(OBJS_COMMON) curlcurl_operator.o test_operator.o
OBJS_RHODOTRON = $(OBJS_COMMON) curlcurl_operator.o test_rhodotron.o


# CUDA Object files
OBJS_CUDA_FIELDS = $(OBJS_COMMON) curlcurl_operator.o cuda_fields.o test_cuda_fields.o
OBJS_CUDA_VECOPS = $(OBJS_COMMON) curlcurl_operator.o cuda_fields.o cuda_vector_ops.o test_cuda_vector_ops.o
OBJS_CUDA_CURLS = $(OBJS_COMMON) curlcurl_operator.o cuda_fields.o cuda_vector_ops.o cuda_curls.o test_cuda_curls.o
OBJS_CUDA_OPERATOR = $(OBJS_COMMON) curlcurl_operator.o cuda_fields.o cuda_vector_ops.o cuda_curls.o cuda_operator.o test_cuda_operator.o
OBJS_CUDA_EIGEN = $(OBJS_COMMON) curlcurl_operator.o cuda_fields.o cuda_vector_ops.o cuda_curls.o cuda_operator.o cuda_eigensolver.o test_cuda_eigensolver.o
OBJS_RHODOTRON_GPU = $(OBJS_COMMON) curlcurl_operator.o cuda_fields.o cuda_vector_ops.o cuda_curls.o cuda_operator.o cuda_eigensolver.o test_rhodotron_gpu.o
OBJS_PIPE_MODEL = $(OBJS_COMMON) curlcurl_operator.o pipe_model.o q_factor.o r_over_q.o field_export.o cuda_fields.o cuda_vector_ops.o cuda_curls.o cuda_operator.o cuda_eigensolver.o cuda_pipe_model.o test_pipe_model.o
OBJS_CONVERGENCE = $(OBJS_COMMON) curlcurl_operator.o pipe_model.o q_factor.o r_over_q.o field_export.o cuda_fields.o cuda_vector_ops.o cuda_curls.o cuda_operator.o cuda_eigensolver.o cuda_pipe_model.o test_convergence.o
OBJS_CONV_FULL = $(OBJS_COMMON) curlcurl_operator.o pipe_model.o q_factor.o r_over_q.o cuda_fields.o cuda_vector_ops.o cuda_curls.o cuda_operator.o cuda_eigensolver.o cuda_pipe_model.o test_convergence_full.o
OBJS_IBC = curl_E.o curl_H.o curlcurl_operator.o pipe_model.o q_factor.o cuda_fields.o cuda_vector_ops.o cuda_curls.o cuda_operator.o cuda_eigensolver.o cuda_pipe_model.o test_ibc.o
OBJS_IBC_PIPE = curl_E.o curl_H.o curlcurl_operator.o pipe_model.o q_factor.o cuda_fields.o cuda_vector_ops.o cuda_curls.o cuda_operator.o cuda_eigensolver.o cuda_pipe_model.o test_ibc_pipe.o

# Conformal object files — shared base
OBJS_CONFORMAL_BASE = curl_E.o curl_H.o curlcurl_operator.o pipe_model.o conformal_geometry.o \
                      cuda_fields.o cuda_vector_ops.o cuda_curls.o cuda_operator.o \
                      cuda_eigensolver.o cuda_pipe_model.o cuda_conformal_curls.o

# test_conformal_cuda: conformal geometry unit tests
OBJS_CONFORMAL_CUDA = $(OBJS_CONFORMAL_BASE) test_conformal_cuda.o

# test_conformal_ibc: conformal + IBC (original 4-phase test)
OBJS_CONFORMAL_IBC = $(OBJS_CONFORMAL_BASE) cuda_conformal_pipe.o q_factor.o r_over_q.o field_export.o test_conformal_ibc.o

# test_conformal_ibc_full: PEC + perturbative IBC with field/R-over-Q export
OBJS_CONFORMAL_IBC_FULL = $(OBJS_CONFORMAL_BASE) cuda_conformal_pipe.o q_factor.o r_over_q.o field_export.o field_map_export.o test_conformal_ibc_full.o


# ============================================================
# Top-level targets
# ============================================================

all: cpu cuda conformal

cpu: $(TARGET_CURL) $(TARGET_CURLCURL) $(TARGET_OPERATOR) $(TARGET_RHODOTRON)

cuda: $(TARGET_CUDA_FIELDS) $(TARGET_CUDA_VECOPS) $(TARGET_CUDA_CURLS) $(TARGET_CUDA_OPERATOR) $(TARGET_CUDA_EIGEN) $(TARGET_RHODOTRON_GPU) $(TARGET_PIPE_MODEL) $(TARGET_CONVERGENCE) $(TARGET_CONV_FULL) $(TARGET_IBC) $(TARGET_IBC_PIPE)

conformal: $(TARGET_CONFORMAL_CUDA) $(TARGET_CONFORMAL_IBC) $(TARGET_CONFORMAL_IBC_FULL)

# ============================================================
# CPU Targets
# ============================================================

$(TARGET_CURL): $(OBJS_CURL)
	$(CXX) $(OBJS_CURL) -o $(TARGET_CURL) $(LDFLAGS)

$(TARGET_CURLCURL): $(OBJS_CURLCURL)
	$(CXX) $(OBJS_CURLCURL) -o $(TARGET_CURLCURL) $(LDFLAGS)

$(TARGET_OPERATOR): $(OBJS_OPERATOR)
	$(CXX) $(OBJS_OPERATOR) -o $(TARGET_OPERATOR) $(LDFLAGS)

$(TARGET_RHODOTRON): $(OBJS_RHODOTRON)
	$(CXX) $(OBJS_RHODOTRON) -o $(TARGET_RHODOTRON) $(LDFLAGS)

# ============================================================
# CUDA Targets
# ============================================================

$(TARGET_CUDA_FIELDS): $(OBJS_CUDA_FIELDS)
	$(NVCC) $(OBJS_CUDA_FIELDS) -o $(TARGET_CUDA_FIELDS) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_CUDA_VECOPS): $(OBJS_CUDA_VECOPS)
	$(NVCC) $(OBJS_CUDA_VECOPS) -o $(TARGET_CUDA_VECOPS) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_CUDA_CURLS): $(OBJS_CUDA_CURLS)
	$(NVCC) $(OBJS_CUDA_CURLS) -o $(TARGET_CUDA_CURLS) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_CUDA_OPERATOR): $(OBJS_CUDA_OPERATOR)
	$(NVCC) $(OBJS_CUDA_OPERATOR) -o $(TARGET_CUDA_OPERATOR) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_CUDA_EIGEN): $(OBJS_CUDA_EIGEN)
	$(NVCC) $(OBJS_CUDA_EIGEN) -o $(TARGET_CUDA_EIGEN) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_RHODOTRON_GPU): $(OBJS_RHODOTRON_GPU)
	$(NVCC) $(OBJS_RHODOTRON_GPU) -o $(TARGET_RHODOTRON_GPU) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_PIPE_MODEL): $(OBJS_PIPE_MODEL)
	$(NVCC) $(OBJS_PIPE_MODEL) -o $(TARGET_PIPE_MODEL) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_CONVERGENCE): $(OBJS_CONVERGENCE)
	$(NVCC) $(OBJS_CONVERGENCE) -o $(TARGET_CONVERGENCE) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_CONV_FULL): $(OBJS_CONV_FULL)
	$(NVCC) $(OBJS_CONV_FULL) -o $(TARGET_CONV_FULL) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_IBC): $(OBJS_IBC)
	$(NVCC) $(OBJS_IBC) -o $(TARGET_IBC) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_IBC_PIPE): $(OBJS_IBC_PIPE)
	$(NVCC) $(OBJS_IBC_PIPE) -o $(TARGET_IBC_PIPE) $(CUDA_LIB) $(CUDA_LDFLAGS)

# ============================================================
# Conformal Targets
# ============================================================

$(TARGET_CONFORMAL_CUDA): $(OBJS_CONFORMAL_CUDA)
	$(NVCC) $(OBJS_CONFORMAL_CUDA) -o $(TARGET_CONFORMAL_CUDA) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_CONFORMAL_IBC): $(OBJS_CONFORMAL_IBC)
	$(NVCC) $(OBJS_CONFORMAL_IBC) -o $(TARGET_CONFORMAL_IBC) $(CUDA_LIB) $(CUDA_LDFLAGS)

$(TARGET_CONFORMAL_IBC_FULL): $(OBJS_CONFORMAL_IBC_FULL)
	$(NVCC) $(OBJS_CONFORMAL_IBC_FULL) -o $(TARGET_CONFORMAL_IBC_FULL) $(CUDA_LIB) $(CUDA_LDFLAGS)

# ============================================================
# CPU Compilation Rules
# ============================================================

curl_E.o: curl_E.cpp curl_E.h
	$(CXX) $(CXXFLAGS) -c curl_E.cpp

curl_H.o: curl_H.cpp curl_H.h curl_E.h
	$(CXX) $(CXXFLAGS) -c curl_H.cpp

curlcurl_operator.o: curlcurl_operator.cpp curlcurl_operator.h curl_E.h curl_H.h
	$(CXX) $(CXXFLAGS) -c curlcurl_operator.cpp

test_curl.o: test_curl.cpp curl_E.h
	$(CXX) $(CXXFLAGS) -c test_curl.cpp

test_curl_curl.o: test_curl_curl.cpp curl_E.h curl_H.h
	$(CXX) $(CXXFLAGS) -c test_curl_curl.cpp

test_operator.o: test_operator.cpp curlcurl_operator.h
	$(CXX) $(CXXFLAGS) -c test_operator.cpp

test_rhodotron.o: test_rhodotron.cpp curlcurl_operator.h
	$(CXX) $(CXXFLAGS) -c test_rhodotron.cpp

pipe_model.o: pipe_model.cpp pipe_model.h curl_E.h
	$(CXX) $(CXXFLAGS) -c pipe_model.cpp

conformal_geometry.o: conformal_geometry.cpp conformal_geometry.h pipe_model.h curl_E.h
	$(CXX) $(CXXFLAGS) -c conformal_geometry.cpp

q_factor.o: q_factor.cpp q_factor.h curlcurl_operator.h curl_E.h curl_H.h
	$(CXX) $(CXXFLAGS) -c q_factor.cpp

r_over_q.o: r_over_q.cpp r_over_q.h curlcurl_operator.h curl_E.h
	$(CXX) $(CXXFLAGS) -c r_over_q.cpp

field_export.o: field_export.cpp field_export.h curlcurl_operator.h curl_E.h
	$(CXX) $(CXXFLAGS) -c field_export.cpp

field_map_export.o: field_map_export.cpp field_map_export.h curlcurl_operator.h curl_E.h
	$(CXX) $(CXXFLAGS) -c field_map_export.cpp

# ============================================================
# CUDA Compilation Rules
# ============================================================

cuda_fields.o: cuda_fields.cu cuda_fields.h curl_E.h curl_H.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c cuda_fields.cu

test_cuda_fields.o: test_cuda_fields.cpp cuda_fields.h curlcurl_operator.h
	$(CXX) $(CXXFLAGS) $(CUDA_INC) -c test_cuda_fields.cpp

cuda_vector_ops.o: cuda_vector_ops.cu cuda_vector_ops.h cuda_fields.h curl_E.h curl_H.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c cuda_vector_ops.cu

test_cuda_vector_ops.o: test_cuda_vector_ops.cpp cuda_vector_ops.h curlcurl_operator.h
	$(CXX) $(CXXFLAGS) $(CUDA_INC) -c test_cuda_vector_ops.cpp

cuda_curls.o: cuda_curls.cu cuda_curls.h cuda_fields.h curl_E.h curl_H.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c cuda_curls.cu

test_cuda_curls.o: test_cuda_curls.cpp cuda_curls.h cuda_fields.h curlcurl_operator.h
	$(CXX) $(CXXFLAGS) $(CUDA_INC) -c test_cuda_curls.cpp

cuda_operator.o: cuda_operator.cu cuda_operator.h cuda_curls.h cuda_fields.h cuda_vector_ops.h curlcurl_operator.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c cuda_operator.cu

test_cuda_operator.o: test_cuda_operator.cpp cuda_operator.h curlcurl_operator.h
	$(CXX) $(CXXFLAGS) $(CUDA_INC) -c test_cuda_operator.cpp

cuda_eigensolver.o: cuda_eigensolver.cu cuda_eigensolver.h cuda_operator.h cuda_curls.h cuda_fields.h cuda_vector_ops.h curlcurl_operator.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c cuda_eigensolver.cu

test_cuda_eigensolver.o: test_cuda_eigensolver.cpp cuda_eigensolver.h cuda_operator.h curlcurl_operator.h
	$(CXX) $(CXXFLAGS) $(CUDA_INC) -c test_cuda_eigensolver.cpp

test_rhodotron_gpu.o: test_rhodotron_gpu.cpp cuda_eigensolver.h cuda_operator.h curlcurl_operator.h
	$(CXX) $(CXXFLAGS) $(CUDA_INC) -c test_rhodotron_gpu.cpp

cuda_pipe_model.o: cuda_pipe_model.cu cuda_pipe_model.h cuda_eigensolver.h pipe_model.h cuda_operator.h cuda_fields.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c cuda_pipe_model.cu

test_pipe_model.o: test_pipe_model.cpp cuda_pipe_model.h cuda_eigensolver.h pipe_model.h curlcurl_operator.h q_factor.h r_over_q.h field_export.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -x cu -c test_pipe_model.cpp -o test_pipe_model.o

test_convergence.o: test_convergence.cpp cuda_pipe_model.h cuda_eigensolver.h pipe_model.h curlcurl_operator.h q_factor.h r_over_q.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -x cu -c test_convergence.cpp -o test_convergence.o

test_convergence_full.o: test_convergence_full.cpp cuda_pipe_model.h cuda_eigensolver.h pipe_model.h curlcurl_operator.h q_factor.h r_over_q.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -x cu -c test_convergence_full.cpp -o test_convergence_full.o

test_ibc.o: test_ibc.cpp cuda_pipe_model.h cuda_eigensolver.h cuda_operator.h pipe_model.h curlcurl_operator.h q_factor.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -x cu -c test_ibc.cpp -o test_ibc.o

test_ibc_pipe.o: test_ibc_pipe.cpp cuda_pipe_model.h cuda_eigensolver.h cuda_operator.h pipe_model.h curlcurl_operator.h q_factor.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -x cu -c test_ibc_pipe.cpp -o test_ibc_pipe.o

# ============================================================
# Conformal Compilation Rules
# ============================================================

cuda_conformal_curls.o: cuda_conformal_curls.cu cuda_conformal.h conformal_geometry.h cuda_fields.h cuda_curls.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c cuda_conformal_curls.cu

cuda_conformal_pipe.o: cuda_conformal_pipe.cu cuda_conformal_pipe.h cuda_conformal.h cuda_pipe_model.h conformal_geometry.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c cuda_conformal_pipe.cu

test_conformal_cuda.o: test_conformal_cuda.cpp cuda_conformal.h cuda_pipe_model.h conformal_geometry.h pipe_model.h curlcurl_operator.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -x cu -c test_conformal_cuda.cpp -o test_conformal_cuda.o

test_conformal_ibc.o: test_conformal_ibc.cpp cuda_conformal_pipe.h cuda_pipe_model.h cuda_eigensolver.h conformal_geometry.h pipe_model.h curlcurl_operator.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -x cu -c test_conformal_ibc.cpp -o test_conformal_ibc.o

test_conformal_ibc_full.o: test_conformal_ibc_full.cpp cuda_conformal_pipe.h cuda_pipe_model.h cuda_eigensolver.h conformal_geometry.h pipe_model.h curlcurl_operator.h q_factor.h r_over_q.h field_export.h field_map_export.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -x cu -c test_conformal_ibc_full.cpp -o test_conformal_ibc_full.o

# ============================================================
# Clean & Run
# ============================================================

clean:
	rm -f *.o $(TARGET_CURL) $(TARGET_CURLCURL) $(TARGET_OPERATOR) \
	       $(TARGET_RHODOTRON) $(TARGET_CUDA_FIELDS) $(TARGET_CUDA_VECOPS) $(TARGET_CUDA_CURLS) \
		   $(TARGET_CUDA_OPERATOR) $(TARGET_CUDA_EIGEN) $(TARGET_RHODOTRON_GPU) $(TARGET_PIPE_MODEL) \
		   $(TARGET_CONVERGENCE) $(TARGET_CONV_FULL) $(TARGET_IBC) $(TARGET_IBC_PIPE) \
		   $(TARGET_CONFORMAL_CUDA) $(TARGET_CONFORMAL_IBC) $(TARGET_CONFORMAL_IBC_FULL)

run_curl: $(TARGET_CURL)
	./$(TARGET_CURL)

run_curlcurl: $(TARGET_CURLCURL)
	./$(TARGET_CURLCURL)

run_operator: $(TARGET_OPERATOR)
	./$(TARGET_OPERATOR)

run_rhodotron: $(TARGET_RHODOTRON)
	./$(TARGET_RHODOTRON)

run_cuda_fields: $(TARGET_CUDA_FIELDS)
	./$(TARGET_CUDA_FIELDS)

run_cuda_vecops: $(TARGET_CUDA_VECOPS)
	./$(TARGET_CUDA_VECOPS)

run_cuda_curls: $(TARGET_CUDA_CURLS)
	./$(TARGET_CUDA_CURLS)

run_cuda_operator: $(TARGET_CUDA_OPERATOR)
	./$(TARGET_CUDA_OPERATOR)

run_cuda_eigen: $(TARGET_CUDA_EIGEN)
	./$(TARGET_CUDA_EIGEN)

run_rhodotron_gpu: $(TARGET_RHODOTRON_GPU)
	./$(TARGET_RHODOTRON_GPU)

run_pipe_model: $(TARGET_PIPE_MODEL)
	./$(TARGET_PIPE_MODEL)

run_convergence: $(TARGET_CONVERGENCE)
	./$(TARGET_CONVERGENCE)

run_conv_full: $(TARGET_CONV_FULL)
	./$(TARGET_CONV_FULL)

run_ibc: $(TARGET_IBC)
	./$(TARGET_IBC)

run_ibc_pipe: $(TARGET_IBC_PIPE)
	./$(TARGET_IBC_PIPE)

run_conformal_cuda: $(TARGET_CONFORMAL_CUDA)
	./$(TARGET_CONFORMAL_CUDA)

run_conformal_ibc: $(TARGET_CONFORMAL_IBC)
	./$(TARGET_CONFORMAL_IBC)

run_conformal_ibc_full: $(TARGET_CONFORMAL_IBC_FULL)
	./$(TARGET_CONFORMAL_IBC_FULL)

run_all_cpu: cpu
	./$(TARGET_CURL)
	./$(TARGET_CURLCURL)
	./$(TARGET_OPERATOR)
	./$(TARGET_RHODOTRON)

run_all: all
	./$(TARGET_CURL)
	./$(TARGET_CURLCURL)
	./$(TARGET_OPERATOR)
	./$(TARGET_RHODOTRON)
	./$(TARGET_CUDA_FIELDS)

.PHONY: all cpu cuda conformal clean \
        run_curl run_curlcurl run_operator run_rhodotron \
        run_cuda_fields run_cuda_vecops run_cuda_curls \
		run_cuda_operator run_cuda_eigen run_rhodotron_gpu run_pipe_model \
		run_convergence run_conv_full run_ibc run_ibc_pipe \
		run_conformal_cuda run_conformal_ibc run_conformal_ibc_full \
		run_all_cpu run_all
