#include <fstream>
#include <iostream>
#include <string>

int main() {
    // /proc/version: カーネルが読み出し時に生成する仮想ファイル。
    // ディスク上には存在せず、open/read するとカーネルが内容をその場で作る。
    std::ifstream file("/proc/version");
    if(!file){
        std::cerr<<"failed to open /proc/version\n";
        return 1;
    }

    std::string line;
    std::getline(file,line);
    std::cout<<line<<'\n';
}
