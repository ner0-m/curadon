build_type := "Debug"
compiler := "g++"
generator := "Ninja"

build_dir := "build" / build_type/ compiler

build: configure
    cmake --build {{build_dir}} --target curadon

build-test: configure
    cmake --build {{build_dir}} --target tests

test filter='': build-test
    ./{{build_dir}}/tests -tc={{filter}}

configure:
    CXX={{compiler}} cmake -S . -B {{build_dir}} -G{{generator}} -DCMAKE_BUILD_TYPE={{build_type}} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    ln -sf {{build_dir}}/compile_commands.json .

clean:
    rm -rf {{build_dir}}

clean-all:
    rm -rf build

profile filter="kernel*": build-test
    nsys profile ./{{build_dir}}/tests -tc={{filter}}

benchmark target="all": configure
    cmake --build {{build_dir}} --target "bench_{{target}}"
    ./{{build_dir}}/benchmark/bench_{{target}}
