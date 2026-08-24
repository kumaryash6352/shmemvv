#####################################
# src/unit/shmemg/CMakeLists.txt
#####################################

# --- Define options to disable specific SHMEMG test categories
# All test categories are built by default unless explicitly disabled
option(DISABLE_SHMEMG_RMA "Disable SHMEMG remote memory access tests" OFF)

# --- Skip all SHMEMG tests if DISABLE_ALL_SHMEMG_TESTS is ON
if(NOT DISABLE_ALL_SHMEMG_TESTS)

  # --- Conditionally add subdirectories based on disable options
  if(NOT DISABLE_SHMEMG_RMA)
    add_subdirectory(rma)
  endif()

endif()
