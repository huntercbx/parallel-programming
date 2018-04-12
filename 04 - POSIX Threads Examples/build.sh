#!/bin/bash

# Looks for installed GCC or CLang compilers and selects one of them
function find_cxx {
    CXX=`which g++` && return
    CXX=`which clang++` && return
    echo 'Neither GCC nor Clang found on this computer, exitting'
    exit 1
}

# If the compiler was not specified as an environment variable — trying to find it
[ -z "$CXX" ] && find_cxx

# Invoking the compiler
$CXX -std=c++11 "$1" -lpthread -o "${1%.*}"
