# MacroUtilities - useful macros and functions for generic tasks
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
    string(CONCAT str "${_first}" "${_remainder}")
    set(${var} "${str}" PARENT_SCOPE)
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
FUNCTION(CHECKOUT_GIT_SUBMODULE)
    # parse args
    cmake_parse_arguments(
        CHECKOUT
        "RECURSIVE"
        "RELATIVE_PATH;WORKING_DIRECTORY;TEST_FILE"
        "ADDITIONAL_CMDS"
        ${ARGN})
    find_package(Git)
    if(NOT Git_FOUND)
        message(WARNING "Git not found. submodule ${CHECKOUT_RELATIVE_PATH} not checked out")
        return()
    endif()

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

    set(_RECURSE )
    if(CHECKOUT_RECURSIVE)
        set(_RECURSE --recursive)
    endif(CHECKOUT_RECURSIVE)

    # if the module has not been checked out
    if(NOT EXISTS "${_TEST_FILE}")
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
            message(WARNING "Command: \"${_CMD}\"")
            return()
        endif()
    elseif(NOT SKIP_GIT_UPDATE)
        message(STATUS "Executing '${GIT_EXECUTABLE} submodule update ${_RECURSE} ${CHECKOUT_RELATIVE_PATH}'... Disable with SKIP_GIT_UPDATE...")
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} submodule update ${_RECURSE} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY
                ${CHECKOUT_WORKING_DIRECTORY})
    endif()

ENDFUNCTION()


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

cmake_policy(POP)
