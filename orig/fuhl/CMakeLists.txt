cmake_minimum_required(VERSION 3.11.0)
project(SpeechClassifier)

add_subdirectory(dlib-19.16)

add_executable(speech_classifier classifier.cpp)
target_link_libraries(speech_classifier dlib::dlib)