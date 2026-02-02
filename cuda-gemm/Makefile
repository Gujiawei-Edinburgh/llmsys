NVCC ?= nvcc
CUDA_ARCH ?= sm_89
NVCCFLAGS ?= -O3 --std=c++14 -arch=$(CUDA_ARCH)
PTXASFLAGS ?=
LDFLAGS ?= -lcublas
INCLUDES := -Iinclude

SRCDIR := src
TESTDIR := test
BUILDDIR := build
BINDIR := bin

SRC_CU := $(wildcard $(SRCDIR)/*.cu)
OBJ := $(patsubst $(SRCDIR)/%.cu,$(BUILDDIR)/%.o,$(SRC_CU))

TEST_CU := $(wildcard $(TESTDIR)/*.cu)
TESTS := $(basename $(notdir $(TEST_CU)))
TEST_BINS := $(addprefix $(BINDIR)/,$(TESTS))

.PHONY: all tests clean $(TESTS)

all: tests

tests: $(TEST_BINS)

$(TESTS): %: $(BINDIR)/%

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $(PTXASFLAGS) $(INCLUDES) -c $< -o $@

$(BINDIR)/%: $(TESTDIR)/%.cu $(OBJ) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(PTXASFLAGS) $(INCLUDES) $< $(OBJ) -o $@ $(LDFLAGS)

$(BUILDDIR):
	mkdir -p $@

$(BINDIR):
	mkdir -p $@

clean:
	rm -rf $(BUILDDIR) $(BINDIR)

.DEFAULT:
	@if [ "$@" = "Makefile" ]; then exit 0; fi
	@echo "Unknown target '$@'. Valid targets: all tests clean $(TESTS)"
	@exit 1
