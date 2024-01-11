# === Create Plugin ===


set(plugin_generator_script "${CMAKE_SOURCE_DIR}/python_module/plugin_generator.py")
set(plugin_name "py_swe_plugin")
set(plugin_build_dir ${CMAKE_CURRENT_BINARY_DIR})
set(plugin_header_file "${CMAKE_SOURCE_DIR}/python_module/py_swe_plugin_header.h")
set(plugin_module_file "${CMAKE_SOURCE_DIR}/python_module/py_swe_plugin_module.py")

add_custom_command(
    OUTPUT "lib${plugin_name}.so"
    COMMAND python ${plugin_generator_script} ${plugin_name} ${plugin_build_dir} ${plugin_header_file} ${plugin_module_file}
    COMMENT "Generating python plugin"
)

# Create a custom target that depends on the custom command
add_custom_target(plugin_generator DEPENDS "lib${plugin_name}.so")


# === Create Fortan App ===


set(SRC_FILES
    fortran_functions.f90
    fortran_program.f90
    fortran_types.f90
)

find_package(MPI REQUIRED COMPONENTS Fortran)

add_executable(fort_python ${SRC_FILES})
target_link_directories(fort_python PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(fort_python PRIVATE MPI::MPI_Fortran ${plugin_name})
add_dependencies(fort_python plugin_generator)

