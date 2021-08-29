set -x
opt_path="/home/liuchao/opt"
version=4.9.0

# pushd $opt_path
# if [[ ! -f gcc-$version ]];then
#     wget -P "$opt_path" http://ftp.gnu.org/gnu/gcc/gcc-$version/gcc-$version.tar.gz
#     tar -xvf gcc-$version.tar.gz
# fi
# popd

#作为安装路径
mkdir -p $opt_path/gcc$version

# 下面的命令不好下载，建议vim看到下载链接后，手动下载到gcc-9.1.0目录下，再执行下面的命令
pushd gcc-$version
    # ./contrib/download_prerequisites
    # 作为编译路径 
    mkdir -p gcc-tmp && pushd gcc-tmp
    ../configure  --enable-languages=c,c++ --disable-multilib --prefix=$opt_path/gcc$version
    make -j32 && make install
    popd
popd