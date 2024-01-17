# === Create Plugin ===


set(plugin_generator_script "${CMAKE_SOURCE_DIR}/python_module/plugin_generator.py")
set(plugin_name "py_swe_plugin")
set(plugin_build_dir ${CMAKE_CURRENT_BINARY_DIR})
set(plugin_header_file "${CMAKE_SOURCE_DIR}/python_module/py_swe_plugin_header.h")
set(plugin_module_file "${CMAKE_SOURCE_DIR}/python_module/py_swe_plugin_module.py")

add_custom_command(
    OUTPUT "lib${plugin_name}.so" "${plugin_name}.c"
    COMMAND python ${plugin_generator_script} ${plugin_name} ${plugin_build_dir} ${plugin_header_file} ${plugin_module_file}
    COMMENT "Generating python plugin"
    DEPENDS ${plugin_header_file} ${plugin_module_file}
)

# Create a custom target that depends on the custom command
add_custom_target(plugin_generator DEPENDS "lib${plugin_name}.so" "${plugin_name}.c")


# === Create Fortan App ===


set(SRC_FILES
    fortran_functions.f90
    fortran_program.f90
    fortran_types.f90
)

if(NOT LINK_PYTHON_DYNAMICALLY)
    # If we want to embedded the Python interpreter statically
    # then instead of linking against the shared library, we
    # can build against the source code of the shared library 
    # that is synthesised by the plugin generation target.
    set(SRC_FILES 
        ${SRC_FILES}
        ${plugin_build_dir}/${plugin_name}.c
    )
endif()

find_package(MPI REQUIRED COMPONENTS Fortran)
find_package(Python REQUIRED Development)

add_executable(fort_python ${SRC_FILES})
add_dependencies(fort_python plugin_generator)
target_link_directories(fort_python PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(fort_python PRIVATE MPI::MPI_Fortran)

if(LINK_PYTHON_DYNAMICALLY)
    target_link_libraries(fort_python PRIVATE ${plugin_name})
else()
    target_link_libraries(fort_python PRIVATE Python::Python)
endif()
