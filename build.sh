source liuchao/build.sh env
bazel build --spawn_strategy=standalone --verbose_failures --local_resources 11048,2.0,2.0 -c dbg --copt -g //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tools/pip_package/build_pip_package ./
pip install tensorflow*.whl --force-reinstall --no-deps
