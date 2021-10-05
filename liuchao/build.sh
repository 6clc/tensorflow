set -x

function env() {
  source ~/.bashrc
  source /home/liuchao/env/tf2_src/bin/activate
  export PATH=/home/liuchao/opt/bazel-0.19.2/bin:$PATH
}
function build_cc_test(){
  env
  bazel build $* -c dbg --copt -g
}


function main(){
  if [[ "build_cc_test" == $1 ]]; then
    build_cc_test ${@:2}
  else 
    echo "erro parameter" $*
  fi
}


main $*