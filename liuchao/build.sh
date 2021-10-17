set -x

function init() {
  echo "run before build first..."
  # 1. bazel 0.19.2
  # 2. tf_env  /home/$USER/env/tf2_src, py3.6
  # 3. gcc 4.8
}

function env() {
  source ~/.bashrc
  source /home/$USER/env/tf2_src/bin/activate
  export PATH=/home/$USER/opt/bazel-0.19.2/bin:$PATH
}
function build_cc_test(){
  env
  bazel build $* -c dbg --copt="-g"
}


function main(){
  if [[ "build_cc_test" == $1 ]]; then
    build_cc_test ${@:2}
  elif [[ "env" == $1 ]]; then
    env ${@:2}
  else 
    echo "erro parameter" $*
  fi
}


main $*


function python_env() {
cd /tmp/ && \
wget -O py3.6.12_pkg.tgz  ${PYTHON36_URL} && \
tar -zxvf py3.6.12_pkg.tgz && \
cd Python-3.6.12 && \
./configure --prefix=/usr/local/python36 && \
make -j80 && make install && \
cd ../ && rm -rf Python-3.6.12

ENV_DIR=/home/$USER/env/tf2_src
mkdir -p $ENV_DIR
virtualenv -p /usr/local/python36/bin/python3  $ENV_DIR

pip install -r requirements.txt

}


