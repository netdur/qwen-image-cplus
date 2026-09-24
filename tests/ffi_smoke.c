#include "qwen_image.h"

#include <stdint.h>

int main(void) {
    if (qi_abi_version() != UINT32_C(1)) {
        return 1;
    }
    if (qi_generate_request_size() != sizeof(QiGenerateRequest)) {
        return 2;
    }
    if (qi_generate_to_png(NULL) != QiStatus_InvalidArgument) {
        return 3;
    }
    return 0;
}
