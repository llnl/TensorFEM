# Compiler
CXX = c++

ifndef TENSORFEM_TPLS_FARM
TENSORFEM_TPLS_FARM=.
endif

ifndef TENSORFEM_TPL_INSTALL_DIR
TENSORFEM_TPL_INSTALL_DIR=${TENSORFEM_TPLS_FARM}
endif

ifndef TENSORFEM_TOP_DIR
TENSORFEM_TOP_DIR=.
endif

MFEM_PATH=${TENSORFEM_TPL_INSTALL_DIR}/tpl/mfem
METIS_PATH=${TENSORFEM_TPL_INSTALL_DIR}/tpl/mfem_tpls/metis-5.1.0
BOBA_TOP_DIR=${TENSORFEM_TPL_INSTALL_DIR}/tpl/BoBa

TENSORFEM_SRC_DIR = ${TENSORFEM_TOP_DIR}/source
TENSORFEM_INC_DIR = ${TENSORFEM_TOP_DIR}/include/tensorfem

include ${BOBA_TOP_DIR}/Makefile_boba

ifeq ($(IS_MATRIX), matrix)
CXXFLAGS := ${CXXFLAGS} -Xcompiler "-Wno-sign-conversion"
else
CXXFLAGS := ${CXXFLAGS} -Wno-sign-conversion
endif

ifdef TENSORFEM_CI
CXX_DEFINES := ${CXX_DEFINES} -DTENSORFEM_CI
endif

# Compiler flags
INCLUDES := ${INCLUDES} -isystem ${TENSORFEM_INC_DIR} -isystem ${MFEM_PATH} -isystem ${BOBA_TOP_DIR}/include/BOBA

# Linker flags
LIBS := ${LIBS} -L${MFEM_PATH} -L${METIS_PATH}/lib -lmfem -lmetis

LINK = ${CXX} ${LINKFLAGS} ${CXX_DEFINES}
COMPILE=${CXX} ${CXXFLAGS} ${CXX_DEFINES} ${INCLUDES}
DEPFLAGS = -MD -MP -MF $(@:.o=.d) -MT $@

EXAMPLES_DIR=./examples

# Build rules
all: \
	tensor_fem${NAME_FLAG}.o \
	tutorial_qpt_extraction \
	integration \
	example_assembly \

#
# TensorFEM
#
tensor_fem${NAME_FLAG}.o: ${TENSORFEM_SRC_DIR}/tensor_fem.cpp ${TENSORFEM_INC_DIR}/tensor_fem.hpp
	${COMPILE} ${OPTS} ${DEPFLAGS} -c $< -o tensor_fem${NAME_FLAG}.o

##
##  integration
##

integration: integration${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

integration${NAME_FLAG}.out: integration${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

integration${NAME_FLAG}.o: ${EXAMPLES_DIR}/tutorial_integration/integration.cpp tensor_fem${NAME_FLAG}.o boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} ${DEPFLAGS} -c ${EXAMPLES_DIR}/tutorial_integration/integration.cpp -o integration${NAME_FLAG}.o

##
## example_assembly
##

example_assembly: example_assembly${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

example_assembly${NAME_FLAG}.out: example_assembly${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

example_assembly${NAME_FLAG}.o: ${EXAMPLES_DIR}/example_assembly/example_assembly.cpp tensor_fem${NAME_FLAG}.o boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} ${DEPFLAGS} -c ${EXAMPLES_DIR}/example_assembly/example_assembly.cpp -o example_assembly${NAME_FLAG}.o

##
## tutorial_qpt_extraction
##

tutorial_qpt_extraction: tutorial_qpt_extraction${NAME_FLAG}.out
	echo "Done making $@${NAME_FLAG}.out"

tutorial_qpt_extraction${NAME_FLAG}.out: tutorial_qpt_extraction${NAME_FLAG}.o boba${NAME_FLAG}.o
	${LINK} -o $@ $^ ${LIBS}

tutorial_qpt_extraction${NAME_FLAG}.o: ${EXAMPLES_DIR}/tutorial_qpt_extraction/tutorial_qpt_extraction.cpp tensor_fem${NAME_FLAG}.o boba${NAME_FLAG}.o
	${COMPILE} ${OPTS} ${DEPFLAGS} -c ${EXAMPLES_DIR}/tutorial_qpt_extraction/tutorial_qpt_extraction.cpp -o tutorial_qpt_extraction${NAME_FLAG}.o

-include $(wildcard *.d)

# Clean rule
clean:
	rm -f *.out *.o *.d

cleanruns: clean
	rm -f *.mesh *.gf *.fes

# Phony targets
.PHONY: all clean
