cmake_minimum_required(VERSION 3.20)
project(VocalIsolator VERSION 1.0.0)

add_subdirectory(JUCE)

include(FetchContent)
FetchContent_Declare(
        onnx_runtime
        URL https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-win-x64-1.17.1.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnx_runtime)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${onnx_runtime_SOURCE_DIR}/lib/onnxruntime.dll"
        IMPORTED_IMPLIB "${onnx_runtime_SOURCE_DIR}/lib/onnxruntime.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${onnx_runtime_SOURCE_DIR}/include"
)

juce_add_plugin(VocalIsolator
        FORMATS VST3 Standalone
        PRODUCT_NAME "Vocal Isolator"
        PLUGIN_MANUFACTURER_CODE "VOCA"
        PLUGIN_CODE "Isol"
        IS_SYNTH FALSE
        NEEDS_MIDI_INPUT FALSE
        NEEDS_MIDI_OUTPUT FALSE
        COPY_PLUGIN_AFTER_BUILD TRUE
)

target_sources(VocalIsolator PRIVATE
        src/PluginProcessor.cpp
        src/PluginEditor.cpp
)

target_link_libraries(VocalIsolator PRIVATE
        juce::juce_audio_plugin_client
        juce::juce_audio_utils
        juce::juce_audio_devices
        juce::juce_dsp
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
        onnxruntime
)

add_custom_command(TARGET VocalIsolator POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${onnx_runtime_SOURCE_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:VocalIsolator>
)