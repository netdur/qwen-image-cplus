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
    if (qi_multi_image_generate_request_size()
            != sizeof(QiMultiImageGenerateRequest)) {
        return 4;
    }
    if (qi_generate_multi_image_to_png(NULL) != QiStatus_InvalidArgument) {
        return 5;
    }
    QiMultiImageGenerateRequest multi = {0};
    multi.abi_version = qi_abi_version();
    multi.struct_size = qi_multi_image_generate_request_size();
    if (qi_generate_multi_image_to_png(&multi) != QiStatus_InvalidArgument) {
        return 6;
    }
    return 0;
}
