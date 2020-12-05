LLVM_CONFIG?=llvm-config

ifndef VERBOSE
QUIET:=@
endif

SRC_DIR?=$(PWD)
LDFLAGS+=$(shell $(LLVM_CONFIG) --ldflags) 
COMMON_FLAGS=-Wall -Wextra
CXXFLAGS+=$(COMMON_FLAGS) $(shell $(LLVM_CONFIG) --cxxflags)
CPPFLAGS+=$(shell $(LLVM_CONFIG) --cppflags) -I$(SRC_DIR)

ADDIR=add_ir
ADDIR_OBJECTS=add_ir.o
default: $(ADDIR)

%.o : $(SRC_DIR)/%.cc
	@echo Compling $*.cc
	$(QUIET)$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) -pthread -ltinfo $<

$(ADDIR): $(ADDIR_OBJECTS)
	@echo Linking $@
	$(QUIET)$(CXX) -o $@ $(CXXFLAGS) -pthread  $(LDFLAGS) $^ `$(LLVM_CONFIG) --libs bitwriter core support` -ltinfo

clean:
	$(QUIET) rm -f $(ADDIR) $(ADDIR_OBJECTS)
