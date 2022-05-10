# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/briansena/Desktop/P2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/briansena/Desktop/P2

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/briansena/Desktop/P2/CMakeFiles /home/briansena/Desktop/P2//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/briansena/Desktop/P2/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named AGGEN

# Build rule for target.
AGGEN: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 AGGEN
.PHONY : AGGEN

# fast build rule for target.
AGGEN/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/build
.PHONY : AGGEN/fast

#=============================================================================
# Target rules for targets named AGEST

# Build rule for target.
AGEST: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 AGEST
.PHONY : AGEST

# fast build rule for target.
AGEST/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/build
.PHONY : AGEST/fast

src/AGEST.o: src/AGEST.cpp.o
.PHONY : src/AGEST.o

# target to build an object file
src/AGEST.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/AGEST.cpp.o
.PHONY : src/AGEST.cpp.o

src/AGEST.i: src/AGEST.cpp.i
.PHONY : src/AGEST.i

# target to preprocess a source file
src/AGEST.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/AGEST.cpp.i
.PHONY : src/AGEST.cpp.i

src/AGEST.s: src/AGEST.cpp.s
.PHONY : src/AGEST.s

# target to generate assembly for a file
src/AGEST.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/AGEST.cpp.s
.PHONY : src/AGEST.cpp.s

src/AGGEN.o: src/AGGEN.cpp.o
.PHONY : src/AGGEN.o

# target to build an object file
src/AGGEN.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/AGGEN.cpp.o
.PHONY : src/AGGEN.cpp.o

src/AGGEN.i: src/AGGEN.cpp.i
.PHONY : src/AGGEN.i

# target to preprocess a source file
src/AGGEN.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/AGGEN.cpp.i
.PHONY : src/AGGEN.cpp.i

src/AGGEN.s: src/AGGEN.cpp.s
.PHONY : src/AGGEN.s

# target to generate assembly for a file
src/AGGEN.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/AGGEN.cpp.s
.PHONY : src/AGGEN.cpp.s

src/Util_Euclidean.o: src/Util_Euclidean.cpp.o
.PHONY : src/Util_Euclidean.o

# target to build an object file
src/Util_Euclidean.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_Euclidean.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_Euclidean.cpp.o
.PHONY : src/Util_Euclidean.cpp.o

src/Util_Euclidean.i: src/Util_Euclidean.cpp.i
.PHONY : src/Util_Euclidean.i

# target to preprocess a source file
src/Util_Euclidean.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_Euclidean.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_Euclidean.cpp.i
.PHONY : src/Util_Euclidean.cpp.i

src/Util_Euclidean.s: src/Util_Euclidean.cpp.s
.PHONY : src/Util_Euclidean.s

# target to generate assembly for a file
src/Util_Euclidean.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_Euclidean.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_Euclidean.cpp.s
.PHONY : src/Util_Euclidean.cpp.s

src/Util_Genetics.o: src/Util_Genetics.cpp.o
.PHONY : src/Util_Genetics.o

# target to build an object file
src/Util_Genetics.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_Genetics.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_Genetics.cpp.o
.PHONY : src/Util_Genetics.cpp.o

src/Util_Genetics.i: src/Util_Genetics.cpp.i
.PHONY : src/Util_Genetics.i

# target to preprocess a source file
src/Util_Genetics.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_Genetics.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_Genetics.cpp.i
.PHONY : src/Util_Genetics.cpp.i

src/Util_Genetics.s: src/Util_Genetics.cpp.s
.PHONY : src/Util_Genetics.s

# target to generate assembly for a file
src/Util_Genetics.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_Genetics.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_Genetics.cpp.s
.PHONY : src/Util_Genetics.cpp.s

src/Util_ReadData.o: src/Util_ReadData.cpp.o
.PHONY : src/Util_ReadData.o

# target to build an object file
src/Util_ReadData.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_ReadData.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_ReadData.cpp.o
.PHONY : src/Util_ReadData.cpp.o

src/Util_ReadData.i: src/Util_ReadData.cpp.i
.PHONY : src/Util_ReadData.i

# target to preprocess a source file
src/Util_ReadData.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_ReadData.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_ReadData.cpp.i
.PHONY : src/Util_ReadData.cpp.i

src/Util_ReadData.s: src/Util_ReadData.cpp.s
.PHONY : src/Util_ReadData.s

# target to generate assembly for a file
src/Util_ReadData.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_ReadData.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_ReadData.cpp.s
.PHONY : src/Util_ReadData.cpp.s

src/Util_mytools.o: src/Util_mytools.cpp.o
.PHONY : src/Util_mytools.o

# target to build an object file
src/Util_mytools.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_mytools.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_mytools.cpp.o
.PHONY : src/Util_mytools.cpp.o

src/Util_mytools.i: src/Util_mytools.cpp.i
.PHONY : src/Util_mytools.i

# target to preprocess a source file
src/Util_mytools.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_mytools.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_mytools.cpp.i
.PHONY : src/Util_mytools.cpp.i

src/Util_mytools.s: src/Util_mytools.cpp.s
.PHONY : src/Util_mytools.s

# target to generate assembly for a file
src/Util_mytools.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Util_mytools.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Util_mytools.cpp.s
.PHONY : src/Util_mytools.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... AGEST"
	@echo "... AGGEN"
	@echo "... src/AGEST.o"
	@echo "... src/AGEST.i"
	@echo "... src/AGEST.s"
	@echo "... src/AGGEN.o"
	@echo "... src/AGGEN.i"
	@echo "... src/AGGEN.s"
	@echo "... src/Util_Euclidean.o"
	@echo "... src/Util_Euclidean.i"
	@echo "... src/Util_Euclidean.s"
	@echo "... src/Util_Genetics.o"
	@echo "... src/Util_Genetics.i"
	@echo "... src/Util_Genetics.s"
	@echo "... src/Util_ReadData.o"
	@echo "... src/Util_ReadData.i"
	@echo "... src/Util_ReadData.s"
	@echo "... src/Util_mytools.o"
	@echo "... src/Util_mytools.i"
	@echo "... src/Util_mytools.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

