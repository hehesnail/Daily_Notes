LLVM_CONFIG?=llvm-config

ifndef VERBOSE
QUIET:=@
endif

SRC_DIR?=$(PWD)
LDFLAGS+=$(shell $(LLVM_CONFIG) --ldflags)
COMMON_FLAGS=-Wall -Wextra
CXXFLAGS+=$(COMMON_FLAGS) $(shell $(LLVM_CONFIG) --cxxflags)
CPPFLAGS+=$(shell $(LLVM_CONFIG) --cppflags) -I$(SRC_DIR)

SUM=sum
SUM_OBJECTS=sum.o
default: $(SUM)

%.o : $(SRC_DIR)/%.cc
	@echo Compling $*.cc
	$(QUIET)$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $<

$(SUM): $(SUM_OBJECTS)
	@echo Linking $@
	$(QUIET)$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ `$(LLVM_CONFIG) --libs bitwriter core support