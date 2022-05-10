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

src/Euclidean.o: src/Euclidean.cpp.o
.PHONY : src/Euclidean.o

# target to build an object file
src/Euclidean.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Euclidean.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Euclidean.cpp.o
.PHONY : src/Euclidean.cpp.o

src/Euclidean.i: src/Euclidean.cpp.i
.PHONY : src/Euclidean.i

# target to preprocess a source file
src/Euclidean.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Euclidean.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Euclidean.cpp.i
.PHONY : src/Euclidean.cpp.i

src/Euclidean.s: src/Euclidean.cpp.s
.PHONY : src/Euclidean.s

# target to generate assembly for a file
src/Euclidean.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Euclidean.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Euclidean.cpp.s
.PHONY : src/Euclidean.cpp.s

src/Genetics.o: src/Genetics.cpp.o
.PHONY : src/Genetics.o

# target to build an object file
src/Genetics.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Genetics.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Genetics.cpp.o
.PHONY : src/Genetics.cpp.o

src/Genetics.i: src/Genetics.cpp.i
.PHONY : src/Genetics.i

# target to preprocess a source file
src/Genetics.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Genetics.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Genetics.cpp.i
.PHONY : src/Genetics.cpp.i

src/Genetics.s: src/Genetics.cpp.s
.PHONY : src/Genetics.s

# target to generate assembly for a file
src/Genetics.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/Genetics.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/Genetics.cpp.s
.PHONY : src/Genetics.cpp.s

src/ReadData.o: src/ReadData.cpp.o
.PHONY : src/ReadData.o

# target to build an object file
src/ReadData.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/ReadData.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/ReadData.cpp.o
.PHONY : src/ReadData.cpp.o

src/ReadData.i: src/ReadData.cpp.i
.PHONY : src/ReadData.i

# target to preprocess a source file
src/ReadData.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/ReadData.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/ReadData.cpp.i
.PHONY : src/ReadData.cpp.i

src/ReadData.s: src/ReadData.cpp.s
.PHONY : src/ReadData.s

# target to generate assembly for a file
src/ReadData.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/ReadData.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/ReadData.cpp.s
.PHONY : src/ReadData.cpp.s

src/mytools.o: src/mytools.cpp.o
.PHONY : src/mytools.o

# target to build an object file
src/mytools.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/mytools.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/mytools.cpp.o
.PHONY : src/mytools.cpp.o

src/mytools.i: src/mytools.cpp.i
.PHONY : src/mytools.i

# target to preprocess a source file
src/mytools.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/mytools.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/mytools.cpp.i
.PHONY : src/mytools.cpp.i

src/mytools.s: src/mytools.cpp.s
.PHONY : src/mytools.s

# target to generate assembly for a file
src/mytools.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGGEN.dir/build.make CMakeFiles/AGGEN.dir/src/mytools.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/AGEST.dir/build.make CMakeFiles/AGEST.dir/src/mytools.cpp.s
.PHONY : src/mytools.cpp.s

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
	@echo "... src/Euclidean.o"
	@echo "... src/Euclidean.i"
	@echo "... src/Euclidean.s"
	@echo "... src/Genetics.o"
	@echo "... src/Genetics.i"
	@echo "... src/Genetics.s"
	@echo "... src/ReadData.o"
	@echo "... src/ReadData.i"
	@echo "... src/ReadData.s"
	@echo "... src/mytools.o"
	@echo "... src/mytools.i"
	@echo "... src/mytools.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

