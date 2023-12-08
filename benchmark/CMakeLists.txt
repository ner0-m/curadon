cpmaddpackage("gh:martinus/nanobench@4.3.11")

add_executable(bench_intersection bench_intersection.cu)
target_link_libraries(bench_intersection PRIVATE nanobench doctest curadon)
target_compile_options(
  bench_intersection PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math>)

add_custom_target(bench_all)
add_dependencies(bench_all bench_intersection)
