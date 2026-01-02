#include <iostream>
#include "llama.h"

int main()
{
    std::cout << "infer_demo: llama backend init...\n";
    llama_backend_init();
    llama_backend_free();
    std::cout << "infer_demo: OK\n";
    return 0;
}
