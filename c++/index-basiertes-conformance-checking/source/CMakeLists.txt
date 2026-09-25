cmake_minimum_required (VERSION 3.16)
project (seqan3_tutorial CXX)

# add seqan3 to search path
list (APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../seqan3/build_system")
 
# require seqan3 with a version between >=3.0.0 and <4.0.0
find_package (seqan3 3.0 REQUIRED)
 
# build app with seqan3
add_executable (alignment_txt alignment_txt.cpp)
target_link_libraries (alignment_txt seqan3::seqan3)

add_executable (xes_to_txt xes_to_txt.cpp)
target_link_libraries (xes_to_txt seqan3::seqan3)

add_executable (txt_to_xes txt_to_xes.cpp)
target_link_libraries (txt_to_xes seqan3::seqan3)

add_executable (remove_double_traces remove_double_traces.cpp)
target_link_libraries (remove_double_traces seqan3::seqan3)


add_library(PugiXML pugiconfig.hpp pugixml.cpp pugixml.hpp)
add_library(MurmurHash3 MurmurHash3.h MurmurHash3.cpp)


target_link_libraries (alignment_txt PugiXML)
target_link_libraries (alignment_txt MurmurHash3)

target_link_libraries (xes_to_txt PugiXML)
target_link_libraries (txt_to_xes PugiXML)