NVCC ?= nvcc
NVCCFLAGS ?= -O3 --std=c++14
LDFLAGS ?= -lcublas
INCLUDES := -Iinclude

SRCDIR := src
TESTDIR := test
BUILDDIR := build
BINDIR := bin

SRC_CU := $(wildcard $(SRCDIR)/*.cu)
OBJ := $(patsubst $(SRCDIR)/%.cu,$(BUILDDIR)/%.o,$(SRC_CU))

TEST_CU := $(wildcard $(TESTDIR)/*.cu)
TEST_BINS := $(patsubst $(TESTDIR)/%.cu,$(BINDIR)/%,$(TEST_CU))

.PHONY: all tests clean

all: tests

tests: $(TEST_BINS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BINDIR)/%: $(TESTDIR)/%.cu $(OBJ) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< $(OBJ) -o $@ $(LDFLAGS)

$(BUILDDIR):
	mkdir -p $@

$(BINDIR):
	mkdir -p $@

clean:
	rm -rf $(BUILDDIR) $(BINDIR)
