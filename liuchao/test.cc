#include <iostream>
#include <cstdio>
// Make function factory and use it
#define FUNCTION(name, a) int fun_##name() { return a;}
 
FUNCTION(abcd, 12)
FUNCTION(fff, 2)
FUNCTION(qqq, 23)
 
#undef FUNCTION
#define FUNCTION 34
#define OUTPUT(a) std::cout << "output: " #a << '\n' 
// # 拼接字符串
// # 替换字符串
 
// Using a macro in the definition of a later macro
#define WORD "Hello "
#define OUTER(...) WORD #__VA_ARGS__

#define MODULE_NAME   "MY_LIB"
#define def_error_print(fmt, ...)  printf("[ERROR]["MODULE_NAME"](%s|%d)" fmt, __func__, __LINE__, ##__VA_ARGS__)

int main()
{
    // std::cout << "abcd: " << fun_abcd() << '\n';
    // std::cout << "fff: " << fun_fff() << '\n';
    // std::cout << "qqq: " << fun_qqq() << '\n';
 
    // std::cout << FUNCTION << '\n';
    // OUTPUT(million);               //note the lack of quotes
 
    // std::cout << OUTER(World) << '\n';
    // std::cout << OUTER(WORD World) << '\n';

    // def_error_print("i=%d,j=%d\n",0,0);// 正确打印输出
    printf("%s %d" "%d", __func__, __LINE__, 0);
 
}
