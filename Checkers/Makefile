# =========================
#  Makefile: C++ + CUDA
# =========================

# Kompilatory
CXX  := g++
NVCC := nvcc

# Target i katalog na obiekty
TARGET := checkers
BUILD  := build

# Ustal CUDA_HOME automatycznie z nvcc (jeśli nie wyjdzie, fallback)
CUDA_HOME ?= $(shell dirname $$(dirname $$(command -v $(NVCC) 2>/dev/null)) 2>/dev/null)
ifeq ($(strip $(CUDA_HOME)),)
  CUDA_HOME := /usr/local/cuda
endif

# Architektura GPU: podmień przy wywołaniu np. make CUDA_ARCH=80
# A100=80, RTX30/A6000=86, RTX40=89, RTX20=75
CUDA_ARCH ?= 35

# Flagi kompilacji
INCLUDES := -I. -I$(CUDA_HOME)/include

CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra $(INCLUDES)
NVCCFLAGS := -O3 -std=c++17 $(INCLUDES) -arch=sm_$(CUDA_ARCH) -rdc=true

# Linkowanie (nvcc jako linker najprościej)
LDFLAGS := -L$(CUDA_HOME)/lib64 -lcudart

# Źródła
CPP_SRCS := main.cpp common.cpp GPU.cpp human_player.cpp mcts_player.cpp
CU_SRCS  := mcts_kernel.cu

CPP_OBJS := $(patsubst %.cpp,$(BUILD)/%.o,$(CPP_SRCS))
CU_OBJS  := $(patsubst %.cu,$(BUILD)/%.o,$(CU_SRCS))
OBJS     := $(CPP_OBJS) $(CU_OBJS)

# =========================
#  Cele
# =========================
.PHONY: all clean info run

all: info $(TARGET)

info:
	@echo "CUDA_HOME = $(CUDA_HOME)"
	@echo "CUDA_ARCH = sm_$(CUDA_ARCH)"
	@echo "NVCC      = $$(command -v $(NVCC) || echo 'nvcc-not-found')"
	@echo ""

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD):
	mkdir -p $(BUILD)

# C++ obiekty
$(BUILD)/%.o: %.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA obiekty
$(BUILD)/%.o: %.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD) $(TARGET)

# Przykład: make run ARGS="game.txt 5 1 0"
run: $(TARGET)
	./$(TARGET) $(ARGS)
