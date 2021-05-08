# bazel build --spawn_strategy=standalone --verbose_failures --local_resources 11048,2.0,2.0 -c dbg --copt -g //tensorflow/tools/pip_package:build_pip_package
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

