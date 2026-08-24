#####################################
# src/unit/shmemg/rma/CMakeLists.txt
#####################################

# --- Add the common source files
set(SOURCES
    ../../../shmemvv.c
    ../../../log.c
)

# --- Add the executables
add_executable(shmemg_put put.c ${SOURCES})

# --- Set the output directory for the executables
set_target_properties(
  shmemg_put
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/rma
)

# --- Include directories for each target
target_include_directories(shmemg_put PRIVATE src/include)

# --- Link any necessary libraries to each target
target_link_libraries(shmemg_put)
