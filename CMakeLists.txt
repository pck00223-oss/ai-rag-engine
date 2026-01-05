cmake_minimum_required(VERSION 3.26)
project(ai_rag_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(hello apps/hello.cpp)

# llama.cpp options (must be set BEFORE add_subdirectory)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp)

# your apps
add_executable(llm_cli apps/llm_cli.cpp)
target_link_libraries(llm_cli PRIVATE llama)
target_include_directories(llm_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)
if (MSVC)
  target_compile_options(llm_cli PRIVATE /EHsc)
endif()

add_executable(infer_demo apps/infer_demo.cpp)
target_link_libraries(infer_demo PRIVATE llama)
target_include_directories(infer_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)
if (MSVC)
  target_compile_options(infer_demo PRIVATE /EHsc)
endif()
if (MSVC)
  target_compile_options(llm_cli PRIVATE /utf-8 /EHsc)
endif()
cmake_minimum_required(VERSION 3.10)
project(AI_RAG_Engine)

set(CMAKE_CXX_STANDARD 17)

# 添加 SQLiteCpp 作为子目录
add_subdirectory(libs/SQLiteCpp)

# 你的源文件
add_executable(ai_rag_engine main.cpp)

# 链接 SQLiteCpp 库
target_link_libraries(ai_rag_engine SQLiteCpp)
