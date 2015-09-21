SHARK_ROOT_DEBUG := /home/hnr137/SharkLibDebug
SHARK_ROOT := /home/hnr137/SharkLib
SHARK_HEADERS := /home/hnr137/SharkLib
CUDA_ROOT := /usr/local/cuda
CUDNN_ROOT := /home/hnr137/cudnn-6.5-linux-x64-v2
SOURCES := $(wildcard src/*.h)
SOURCES += $(wildcard src/layers/*.h)
OBJS := $(addprefix obj/, $(notdir $(SOURCES:.h=.o)))
CU_DEPS := $(addprefix obj/, $(notdir $(src/layers/*.cu)))
DEPS := $(wildcard src/*.h)
DEPS += $(wildcard src/layers/*.h)
DEPS += $(CU_DEPS)
TEST_SOURCES := $(wildcard tests/*.cpp)
TEST_OBJS := $(addprefix obj/, $(notdir $(TEST_SOURCES:.cpp=.o)))
INCLUDE := -I$(CUDA_ROOT)/include/ -I$(CUDNN_ROOT) -I$(SHARK_HEADERS)/include/ 
INCLUDE_TEST := -I$(SHARK_HEADERS)/Test/Models/
LIBS_DEBUG := -lm -lstdc++ -L$(CUDNN_ROOT) -lcudnn -L$(SHARK_ROOT_DEBUG)/lib/ -lshark_debug -lboost_serialization -lboost_unit_test_framework -lboost_timer -lboost_system -L$(CUDA_ROOT)/lib64/ -lcublas -lnppc -lnppi -lcurand
LIBS := -lm -lstdc++ -L$(CUDNN_ROOT) -lcudnn -L$(SHARK_ROOT)/lib/ -lshark -lboost_serialization -lboost_unit_test_framework -lboost_timer -lboost_system -L$(CUDA_ROOT)/lib64/ -lcublas -lnppc -lnppi -lcurand
CXX := $(CUDA_ROOT)/bin/nvcc
CFLAGS_DEBUG := -std=c++11 -g -G -DBOOST_TEST_DYN_LINK=1 -gencode arch=compute_35,code=sm_35
CFLAGS := -std=c++11 -g -G -gencode arch=compute_35,code=sm_35 -DBOOST_TEST_DYN_LINK=1 -O3
EXE := bin/convnet
TEST_EXE := bin/test
BASE_DEPS := ./src/layers/shark_cuda_helpers.cpp ./src/layers/AbstractLayer.cpp ./src/layers/AbstractWeightedLayer.cu ./src/layers/DistortLayer.cu

obj: $(OBJS)
	echo "done."

./obj/%.o: ./tests/%.cpp $(DEPS)
	$(CXX) $(CFLAGS) $(INCLUDE) -I$(SHARK_HEADERS)/Test/Models/ -o $@ $< $(LIBS)

tests: $(DEPS) $(TEST_SOURCES)
	$(CXX) $(CFLAGS) $(INCLUDE) $(INCLUDE_TEST) -o ./obj/LayersGradientUnitTest.o ./tests/LayersGradientUnitTest.cpp $(BASE_DEPS) $(LIBS)
	$(CXX) $(CFLAGS) $(INCLUDE) $(INCLUDE_TEST) -o ./obj/ConvNetTests.o ./tests/ConvNetTests.cpp $(BASE_DEPS) $(LIBS)

mnist: $(DEPS) ./tests/MnistTest.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) $(BASE_DEPS) -o ./obj/MnistTest.o ./tests/MnistTest.cpp $(LIBS)

cifar: $(DEPS) ./tests/CifarTest.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) $(BASE_DEPS) -o ./obj/CifarTest.o ./tests/CifarTest.cpp $(LIBS)

distort_to_disk: $(DEPS) ./tests/DistortLayerToDiskTest.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) -o ./obj/DistortLayerToDiskTest.o ./tests/DistortLayerToDiskTest.cpp $(BASE_DEPS) $(LIBS)

clean:
	rm -f $(OBJS) $(EXE) $(TEST_OBJS) $(TEST_EXE) ./obj/DistortLayer.o
