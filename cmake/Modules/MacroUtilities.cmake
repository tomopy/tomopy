# MacroUtilities - useful macros and functions for generic tasks
#
# CMake Extensions
# ----------------
# macro set_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided
#
# function enum_option(<option>
#                      VALUES <value1> ... <valueN>
#                      TYPE   <valuetype>
#                      DOC    <docstring>
#                      [DEFAULT <elem>]
#                      [CASE_INSENSITIVE])
#          Declare a cache variable <option> that can only take values
#          listed in VALUES. TYPE may be FILEPATH, PATH or STRING.
#          <docstring> should describe that option, and will appear in
#          the interactive CMake interfaces. If DEFAULT is provided,
#          <elem> will be taken as the zero-indexed element in VALUES
#          to which the value of <option> should default to if not
#          provided. Otherwise, the default is taken as the first
#          entry in VALUES. If CASE_INSENSITIVE is present, then
#          checks of the value of <option> against the allowed values
#          will ignore the case when performing string comparison.
#
#
# General
# --------------
# function add_feature(<NAME> <DOCSTRING>)
#          Add a  feature, whose activation is specified by the
#          existence of the variable <NAME>, to the list of enabled/disabled
#          features, plus a docstring describing the feature
#
# function print_enabled_features()
#          Print enabled  features plus their docstrings.
#
#

# - Include guard
if(__${PROJECT_NAME}_macroutilities_isloaded)
  return()
endif()
set(__${PROJECT_NAME}_macroutilities_isloaded YES)

cmake_policy(PUSH)
if(NOT CMAKE_VERSION VERSION_LESS 3.1)
    cmake_policy(SET CMP0054 NEW)
endif()

include(CMakeDependentOption)
include(CMakeParseArguments)

#-----------------------------------------------------------------------
# CMAKE EXTENSIONS
#-----------------------------------------------------------------------
# macro set_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided
#
MACRO(set_ifnot _var _value)
    if(NOT DEFINED ${_var})
        set(${_var} ${_value} ${ARGN})
    endif()
ENDMACRO()

#-----------------------------------------------------------------------
# macro safe_remove_duplicates(<list>)
#       ensures remove_duplicates is only called if list has values
#
MACRO(safe_remove_duplicates _list)
    if(NOT "${${_list}}" STREQUAL "")
        list(REMOVE_DUPLICATES ${_list})
    endif(NOT "${${_list}}" STREQUAL "")
ENDMACRO()

#-----------------------------------------------------------------------
# function - capitalize - make a string capitalized (first letter is capital)
#   usage:
#       capitalize("SHARED" CShared)
#   message(STATUS "-- CShared is \"${CShared}\"")
#   $ -- CShared is "Shared"
FUNCTION(capitalize str var)
    # make string lower
    string(TOLOWER "${str}" str)
    string(SUBSTRING "${str}" 0 1 _first)
    string(TOUPPER "${_first}" _first)
    string(SUBSTRING "${str}" 1 -1 _remainder)
    if(GOOD_CMAKE)
        string(CONCAT str "${_first}" "${_remainder}")
    else(GOOD_CMAKE)
        set(str "${_first}${_remainder}")
    endif(GOOD_CMAKE)
    set(${var} "${str}" PARENT_SCOPE)
ENDFUNCTION()

#-----------------------------------------------------------------------
# macro set_ifnot_match(<var> <value>)
#       If variable var is not set, set its value to that provided
#
MACRO(SET_IFNOT_MATCH VAR APPEND)
    if(NOT "${APPEND}" STREQUAL "")
        STRING(REGEX MATCH "${APPEND}" _MATCH "${${VAR}}")
        if(NOT "${_MATCH}" STREQUAL "")
            SET(${VAR} "${${VAR}} ${APPEND}")
        endif()
    endif()
ENDMACRO()

#-----------------------------------------------------------------------
# macro cache_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided and cache it
#
MACRO(cache_ifnot _var _value _type _doc)
  if(NOT ${_var} OR NOT ${CACHE_VARIABLES} MATCHES ${_var})
    set(${_var} ${_value} CACHE ${_type} "${_doc}")
  endif()
ENDMACRO()

#-----------------------------------------------------------------------
# function enum_option(<option>
#                      VALUES <value1> ... <valueN>
#                      TYPE   <valuetype>
#                      DOC    <docstring>
#                      [DEFAULT <elem>]
#                      [CASE_INSENSITIVE])
#          Declare a cache variable <option> that can only take values
#          listed in VALUES. TYPE may be FILEPATH, PATH or STRING.
#          <docstring> should describe that option, and will appear in
#          the interactive CMake interfaces. If DEFAULT is provided,
#          <elem> will be taken as the zero-indexed element in VALUES
#          to which the value of <option> should default to if not
#          provided. Otherwise, the default is taken as the first
#          entry in VALUES. If CASE_INSENSITIVE is present, then
#          checks of the value of <option> against the allowed values
#          will ignore the case when performing string comparison.
#
FUNCTION(enum_option _var)
    set(options CASE_INSENSITIVE)
    set(oneValueArgs DOC TYPE DEFAULT)
    set(multiValueArgs VALUES)
    cmake_parse_arguments(_ENUMOP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # - Validation as needed arguments
    if(NOT _ENUMOP_VALUES)
        message(FATAL_ERROR "enum_option must be called with non-empty VALUES\n(Called for enum_option '${_var}')")
    endif()

    # - Set argument defaults as needed
    if(_ENUMOP_CASE_INSENSITIVE)
        set(_ci_values )
        foreach(_elem ${_ENUMOP_VALUES})
        string(TOLOWER "${_elem}" _ci_elem)
        list(APPEND _ci_values "${_ci_elem}")
        endforeach()
        set(_ENUMOP_VALUES ${_ci_values})
    endif()

    set_ifnot(_ENUMOP_TYPE STRING)
    set_ifnot(_ENUMOP_DEFAULT 0)
    list(GET _ENUMOP_VALUES ${_ENUMOP_DEFAULT} _default)

    if(NOT DEFINED ${_var})
        set(${_var} ${_default} CACHE ${_ENUMOP_TYPE} "${_ENUMOP_DOC} (${_ENUMOP_VALUES})")
    else()
        set(_var_tmp ${${_var}})
        if(_ENUMOP_CASE_INSENSITIVE)
        string(TOLOWER ${_var_tmp} _var_tmp)
        endif()

        list(FIND _ENUMOP_VALUES ${_var_tmp} _elem)
        if(_elem LESS 0)
        message(FATAL_ERROR "Value '${${_var}}' for variable ${_var} is not allowed\nIt must be selected from the set: ${_ENUMOP_VALUES} (DEFAULT: ${_default})\n")
        else()
        # - convert to lowercase
        if(_ENUMOP_CASE_INSENSITIVE)
            set(${_var} ${_var_tmp} CACHE ${_ENUMOP_TYPE} "${_ENUMOP_DOC} (${_ENUMOP_VALUES})" FORCE)
        endif()
        endif()
    endif()
ENDFUNCTION()


#-----------------------------------------------------------------------
# GENERAL
#-----------------------------------------------------------------------
# function add_feature(<NAME> <DOCSTRING>)
#          Add a project feature, whose activation is specified by the
#          existence of the variable <NAME>, to the list of enabled/disabled
#          features, plus a docstring describing the feature
#
FUNCTION(ADD_FEATURE _var _description)
  set(EXTRA_DESC "")
  foreach(currentArg ${ARGN})
      if(NOT "${currentArg}" STREQUAL "${_var}" AND
         NOT "${currentArg}" STREQUAL "${_description}")
          set(EXTRA_DESC "${EXTA_DESC}${currentArg}")
      endif()
  endforeach()

  set_property(GLOBAL APPEND PROPERTY PROJECT_FEATURES ${_var})
  #set(${_var} ${${_var}} CACHE INTERNAL "${_description}${EXTRA_DESC}")

  set_property(GLOBAL PROPERTY ${_var}_DESCRIPTION "${_description}${EXTRA_DESC}")
ENDFUNCTION()


#------------------------------------------------------------------------------#
# function add_subfeature(<ROOT_OPTION> <NAME> <DOCSTRING>)
#          Add a subfeature, whose activation is specified by the
#          existence of the variable <NAME>, to the list of enabled/disabled
#          features, plus a docstring describing the feature
#
FUNCTION(ADD_SUBFEATURE _root _var _description)
    set(EXTRA_DESC "")
    foreach(currentArg ${ARGN})
        if(NOT "${currentArg}" STREQUAL "${_var}" AND
           NOT "${currentArg}" STREQUAL "${_description}")
            set(EXTRA_DESC "${EXTA_DESC}${currentArg}")
        endif()
    endforeach()

    set_property(GLOBAL APPEND PROPERTY ${_root}_FEATURES ${_var})
    set_property(GLOBAL PROPERTY ${_root}_${_var}_DESCRIPTION "${_description}${EXTRA_DESC}")
ENDFUNCTION()


#------------------------------------------------------------------------------#
# function add_option(<OPTION_NAME> <DOCSRING> <DEFAULT_SETTING> [NO_FEATURE])
#          Add an option and add as a feature if NO_FEATURE is not provided
#
FUNCTION(ADD_OPTION _NAME _MESSAGE _DEFAULT)
    SET(_FEATURE ${ARGN})
    OPTION(${_NAME} "${_MESSAGE}" ${_DEFAULT})
    IF(NOT "${_FEATURE}" STREQUAL "NO_FEATURE")
        ADD_FEATURE(${_NAME} "${_MESSAGE}")
    ELSE()
        MARK_AS_ADVANCED(${_NAME})
    ENDIF()
ENDFUNCTION(ADD_OPTION _NAME _MESSAGE _DEFAULT)


#------------------------------------------------------------------------------#
# function add_dependent_option(<OPTION_NAME> <DOCSRING>
#                               <CONDITION_TRUE_SETTING> <CONDITION>
#                               <DEFAULT_SETTING> [NO_FEATURE])
#          Add an option and add as a feature if NO_FEATURE is not provided
#
FUNCTION(ADD_DEPENDENT_OPTION _NAME _MESSAGE _COND_SETTING _COND _DEFAULT)
    SET(_FEATURE ${ARGN})
    IF(DEFINED ${_NAME} AND NOT ${_COND})
        OPTION(${_NAME} "${_MESSAGE}" ${_DEFAULT})
    ELSE(DEFINED ${_NAME} AND NOT ${_COND})
        CMAKE_DEPENDENT_OPTION(${_NAME} "${_MESSAGE}" ${_COND_SETTING}
            "${_COND}" ${_DEFAULT})
    ENDIF(DEFINED ${_NAME} AND NOT ${_COND})

    IF(NOT "${_FEATURE}" STREQUAL "NO_FEATURE")
        ADD_FEATURE(${_NAME} "${_MESSAGE}")
    ELSE()
        MARK_AS_ADVANCED(${_NAME})
    ENDIF()
ENDFUNCTION(ADD_DEPENDENT_OPTION _NAME _MESSAGE _DEFAULT _COND _COND_SETTING)


#------------------------------------------------------------------------------#
# macro CHECKOUT_GIT_SUBMODULE()
#
#   Run "git submodule update" if a file in a submodule does not exist
#
#   ARGS:
#       RECURSIVE (option) -- add "--recursive" flag
#       RELATIVE_PATH (one value) -- typically the relative path to submodule
#                                    from PROJECT_SOURCE_DIR
#       WORKING_DIRECTORY (one value) -- (default: PROJECT_SOURCE_DIR)
#       TEST_FILE (one value) -- file to check for (default: CMakeLists.txt)
#       ADDITIONAL_CMDS (many value) -- any addition commands to pass
#
MACRO(CHECKOUT_GIT_SUBMODULE)
    # parse args
    cmake_parse_arguments(
        CHECKOUT
        "RECURSIVE"
        "RELATIVE_PATH;WORKING_DIRECTORY;TEST_FILE"
        "ADDITIONAL_CMDS"
        ${ARGN})

    if(NOT CHECKOUT_WORKING_DIRECTORY)
        set(CHECKOUT_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif(NOT CHECKOUT_WORKING_DIRECTORY)

    if(NOT CHECKOUT_TEST_FILE)
        set(CHECKOUT_TEST_FILE "CMakeLists.txt")
    endif(NOT CHECKOUT_TEST_FILE)

    set(_DIR "${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}")
    # ensure the (possibly empty) directory exists
    if(NOT EXISTS "${_DIR}")
        message(FATAL_ERROR "submodule directory does not exist")
    endif(NOT EXISTS "${_DIR}")

    # if this file exists --> project has been checked out
    # if not exists --> not been checked out
    set(_TEST_FILE "${_DIR}/${CHECKOUT_TEST_FILE}")

    # if the module has not been checked out
    if(NOT EXISTS "${_TEST_FILE}")
        find_package(Git REQUIRED)

        set(_RECURSE )
        if(CHECKOUT_RECURSIVE)
            set(_RECURSE --recursive)
        endif(CHECKOUT_RECURSIVE)

        # perform the checkout
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                    ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY
                ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET)

        # check the return code
        if(RET GREATER 0)
            set(_CMD "${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}")
            message(STATUS "macro(CHECKOUT_SUBMODULE) failed.")
            message(FATAL_ERROR "Command: \"${_CMD}\"")
        endif()

    endif()

ENDMACRO()


#------------------------------------------------------------------------------#
# function print_enabled_features()
#          Print enabled  features plus their docstrings.
#
FUNCTION(print_enabled_features)
    set(_basemsg "The following features are defined/enabled (+):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY PROJECT_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(${_feature})
            # add feature to text
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")
            # get description
            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)
            # print description, if not standard ON/OFF, print what is set to
            if(_desc)
                if(NOT "${${_feature}}" STREQUAL "ON" AND
                   NOT "${${_feature}}" STREQUAL "TRUE")
                    set(_currentFeatureText "${_currentFeatureText}: ${_desc} -- [\"${${_feature}}\"]")
                else()
                    string(REGEX REPLACE "^USE_" "" _feature_tmp "${_feature}")
                    string(TOLOWER "${_feature_tmp}" _feature_tmp_l)
                    capitalize("${_feature_tmp}" _feature_tmp_c)
                    foreach(_var _feature_tmp _feature_tmp_l _feature_tmp_c)
                        set(_ver "${${${_var}}_VERSION}")
                        if(NOT "${_ver}" STREQUAL "")
                            set(_desc "${_desc} -- [found version ${_ver}]")
                            break()
                        endif()
                        unset(_ver)
                    endforeach(_var _feature_tmp _feature_tmp_l _feature_tmp_c)
                    set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
                endif()
                set(_desc NOTFOUND)
            endif(_desc)
            # check for subfeatures
            get_property(_subfeatures GLOBAL PROPERTY ${_feature}_FEATURES)
            # remove duplicates and sort if subfeatures exist
            if(NOT "${_subfeatures}" STREQUAL "")
                list(REMOVE_DUPLICATES _subfeatures)
                list(SORT _subfeatures)
            endif()

            # sort enabled and disabled features into lists
            set(_enabled_subfeatures )
            set(_disabled_subfeatures )
            foreach(_subfeature ${_subfeatures})
                if(${_subfeature})
                    list(APPEND _enabled_subfeatures ${_subfeature})
                else()
                    list(APPEND _disabled_subfeatures ${_subfeature})
                endif()
            endforeach()

            # loop over enabled subfeatures
            foreach(_subfeature ${_enabled_subfeatures})
                # add subfeature to text
                set(_currentFeatureText "${_currentFeatureText}\n       + ${_subfeature}")
                # get subfeature description
                get_property(_subdesc GLOBAL PROPERTY ${_feature}_${_subfeature}_DESCRIPTION)
                # print subfeature description. If not standard ON/OFF, print
                # what is set to
                if(_subdesc)
                    if(NOT "${${_subfeature}}" STREQUAL "ON" AND
                       NOT "${${_subfeature}}" STREQUAL "TRUE")
                        set(_currentFeatureText "${_currentFeatureText}: ${_subdesc} -- [\"${${_subfeature}}\"]")
                    else()
                        set(_currentFeatureText "${_currentFeatureText}: ${_subdesc}")
                    endif()
                    set(_subdesc NOTFOUND)
                endif(_subdesc)
            endforeach(_subfeature)

            # loop over disabled subfeatures
            foreach(_subfeature ${_disabled_subfeatures})
                # add subfeature to text
                set(_currentFeatureText "${_currentFeatureText}\n       - ${_subfeature}")
                # get subfeature description
                get_property(_subdesc GLOBAL PROPERTY ${_feature}_${_subfeature}_DESCRIPTION)
                # print subfeature description.
                if(_subdesc)
                    set(_currentFeatureText "${_currentFeatureText}: ${_subdesc}")
                    set(_subdesc NOTFOUND)
                endif(_subdesc)
            endforeach(_subfeature)

        endif(${_feature})
    endforeach(_feature)

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
ENDFUNCTION()


#------------------------------------------------------------------------------#
# function print_disabled_features()
#          Print disabled features plus their docstrings.
#
FUNCTION(print_disabled_features)
    set(_basemsg "The following features are NOT defined/enabled (-):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY PROJECT_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(NOT ${_feature})
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")

            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)

            if(_desc)
              set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
              set(_desc NOTFOUND)
            endif(_desc)
        endif()
    endforeach(_feature)

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
ENDFUNCTION()

#------------------------------------------------------------------------------#
# function print_features()
#          Print all features plus their docstrings.
#
FUNCTION(print_features)
    message(STATUS "")
    print_enabled_features()
    print_disabled_features()
ENDFUNCTION()


#------------------------------------------------------------------------------#
MACRO(DETERMINE_LIBDIR_DEFAULT VAR)
    set(_LIBDIR_DEFAULT "lib")
    # Override this default 'lib' with 'lib64' iff:
    #  - we are on Linux system but NOT cross-compiling
    #  - we are NOT on debian
    #  - we are on a 64 bits system
    # reason is: amd64 ABI: https://github.com/hjl-tools/x86-psABI/wiki/X86-psABI
    # For Debian with multiarch, use 'lib/${CMAKE_LIBRARY_ARCHITECTURE}' if
    # CMAKE_LIBRARY_ARCHITECTURE is set (which contains e.g. "i386-linux-gnu"
    # and CMAKE_INSTALL_PREFIX is "/usr"
    # See http://wiki.debian.org/Multiarch
    if(DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX)
        set(__LAST_LIBDIR_DEFAULT "lib")
        # __LAST_LIBDIR_DEFAULT is the default value that we compute from
        # _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX, not a cache entry for
        # the value that was last used as the default.
        # This value is used to figure out whether the user changed the
        # LIBDIR_DEFAULT value manually, or if the value was the
        # default one. When CMAKE_INSTALL_PREFIX changes, the value is
        # updated to the new default, unless the user explicitly changed it.
    endif()
    if(CMAKE_SYSTEM_NAME MATCHES "^(Linux|kFreeBSD|GNU)$"
            AND NOT CMAKE_CROSSCOMPILING)
        if (EXISTS "/etc/debian_version") # is this a debian system ?
            if(CMAKE_LIBRARY_ARCHITECTURE)
                if("${CMAKE_INSTALL_PREFIX}" MATCHES "^/usr/?$")
                    set(_LIBDIR_DEFAULT "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
                endif()
                if(DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX
                        AND "${_GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX}" MATCHES "^/usr/?$")
                    set(__LAST_LIBDIR_DEFAULT "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
                endif()
            endif()
        else() # not debian, rely on CMAKE_SIZEOF_VOID_P:
            if(NOT DEFINED CMAKE_SIZEOF_VOID_P)
                message(AUTHOR_WARNING
                    "Unable to determine default LIBDIR_DEFAULT directory "
                    "because no target architecture is known. "
                    "Please enable at least one language before including GNUInstallDirs.")
            else()
                if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
                    set(_LIBDIR_DEFAULT "lib64")
                    if(DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX)
                        set(__LAST_LIBDIR_DEFAULT "lib64")
                    endif()
                endif()
            endif()
        endif()
    endif()

    # if assign to another variable
    if(NOT "${VAR}" STREQUAL "LIBDIR_DEFAULT")
        set(${VAR} "${_LIBDIR_DEFAULT}")
    endif(NOT "${VAR}" STREQUAL "LIBDIR_DEFAULT")

    # cache the value
    if(NOT DEFINED LIBDIR_DEFAULT)
        set(LIBDIR_DEFAULT "${_LIBDIR_DEFAULT}" CACHE PATH "Object code libraries (${_LIBDIR_DEFAULT})" FORCE)
    elseif(DEFINED __LAST_LIBDIR_DEFAULT
            AND "${__LAST_LIBDIR_DEFAULT}" STREQUAL "${LIBDIR_DEFAULT}")
        set_property(CACHE LIBDIR_DEFAULT PROPERTY VALUE "${_LIBDIR_DEFAULT}")
    endif()
ENDMACRO()


#----------------------------------------------------------------------------
# Macro for building library
#
MACRO(BUILD_LIBRARY)
    cmake_parse_arguments(LIB
    "VERSION" "TYPE;TARGET_NAME;OUTPUT_NAME;EXTENSION" "LINK_LIBRARIES;SOURCES;EXTRA_ARGS"
    ${ARGN})

    MACRO(setifnot VAR)
        if(NOT ${VAR} OR "${${VAR}}" STREQUAL "")
            set(${VAR} ${ARGN})
        endif()
    ENDMACRO()

    setifnot(LIB_EXTENSION  ${PYTHON_MODULE_EXTENSION})
    setifnot(LIB_OUTPUT_NAME ${LIB_TARGET_NAME})

    add_library(${LIB_TARGET_NAME} ${LIB_TYPE} ${LIB_SOURCES})

    target_link_libraries(${LIB_TARGET_NAME}
    PUBLIC ${EXTERNAL_LIBRARIES} ${LIB_LINK_LIBRARIES})

    set_target_properties(${LIB_TARGET_NAME}
        PROPERTIES
            OUTPUT_NAME                 ${LIB_OUTPUT_NAME}
            LANGUAGE                    CXX
            LINKER_LANGUAGE             CXX
            POSITION_INDEPENDENT_CODE   ON
            ${LIB_EXTRA_ARGS})

    list(APPEND INSTALL_LIBRARIES ${TARGET_NAME})
ENDMACRO(BUILD_LIBRARY)


#------------------------------------------------------------------------------#
# always determine the default lib directory
DETERMINE_LIBDIR_DEFAULT(LIBDIR_DEFAULT)

cmake_policy(POP)
