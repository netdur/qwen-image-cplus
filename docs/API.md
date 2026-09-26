# Library API

The Homebrew package includes `qwen_image.h`, `libqwen_image.dylib`, and
`libqwen_image.a`. The C ABI is synchronous and file-oriented: a call reads the
model and input paths, writes a PNG, and returns when generation finishes.
Model weights are [downloaded separately](../README.md#model-files).

## C: text to image

Save this as `example.c`:

```c
#include "qwen_image.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s PACK.qipack MODEL_DIR OUTPUT.png\n", argv[0]);
        return 2;
    }
    if (qi_abi_version() != 1) {
        fprintf(stderr, "unsupported qwen_image ABI\n");
        return 2;
    }

    const char *prompt = "A red balloon against a blue sky";
    QiGenerateRequest request = {0};
    request.abi_version = qi_abi_version();
    request.struct_size = qi_generate_request_size();
    request.packed_path = (uint8_t *)argv[1];
    request.packed_path_length = strlen(argv[1]);
    request.model_root = (uint8_t *)argv[2];
    request.model_root_length = strlen(argv[2]);
    request.output_path = (uint8_t *)argv[3];
    request.output_path_length = strlen(argv[3]);
    request.prompt = (uint8_t *)prompt;
    request.prompt_length = strlen(prompt);
    request.width = 512;
    request.height = 512;
    request.steps = 6;
    request.seed = 1301;
    request.cache_mode = QiCacheMode_None;

    QiStatus status = qi_generate_to_png(&request);
    if (status != QiStatus_Ok) {
        fprintf(stderr, "generation failed (status %d)\n", (int)status);
        return 1;
    }
    return 0;
}
```

Compile and run it against the six-step Viggle pack:

```sh
prefix="$(brew --prefix qwen-image-cplus)"
clang -std=c11 -Wall -Wextra -Werror example.c \
  -I"$prefix/include" -L"$prefix/lib" -lqwen_image \
  -Wl,-rpath,"$prefix/lib" \
  -framework Foundation -framework Metal -lobjc -o example
./example models/qwen-image-2.1-viggle-v0.2.1-lora-fp16-v4.qipack models output.png
```

Use a step count appropriate to the selected pack: six for Viggle v0.2.1,
four for Viggle v0.1, or the base pack's supported 40-step path. The C API
does not infer steps from pack metadata; unlike the CLI's `pack` option, you
set `request.steps` explicitly.

## Request contract

- Set `abi_version` and `struct_size` using the exported functions before
  calling the library. The generated header is the source of truth for the
  request layout and enum values.
- Strings are pointer-and-byte-length pairs, not required to be NUL-terminated.
  Keep their storage alive until the synchronous call returns.
- Set both `width` and `height` for a rectangular or square output. They must
  be multiples of 32 within the model's output-token limit. `pixels` remains
  the older square-only field and is used when both dimensions are zero.
- `steps` accepts 3, 4, 6, 8, 25, or 40. The 256x256 text path requires 25 or
  40. For a first run, use `QiCacheMode_None`. Cache modes are calibrated only
  for 256x256 or 1024x1024, with thresholds 0.12, 0.14, 0.16, or 0.24.
- `QiStatus_Ok` means the PNG was written; `QiStatus_InvalidArgument` means the
  request was rejected; `QiStatus_GenerationFailed` covers runtime failures.

## C: image-conditioned generation

`qi_generate_multi_image_to_png` accepts one to ten images and a 512x512 or
1024x1024 output. It is square-only today; the CLI and native C+ API also
support rectangular edits. Image paths and their byte lengths are parallel
borrowed arrays:

```c
uint8_t *image_paths[] = {(uint8_t *)"input.jpg"};
size_t image_lengths[] = {strlen("input.jpg")};

QiMultiImageGenerateRequest edit = {0};
edit.abi_version = qi_abi_version();
edit.struct_size = qi_multi_image_generate_request_size();
edit.packed_path = (uint8_t *)pack;
edit.packed_path_length = strlen(pack);
edit.model_root = (uint8_t *)model_root;
edit.model_root_length = strlen(model_root);
edit.output_path = (uint8_t *)output_path;
edit.output_path_length = strlen(output_path);
edit.prompt = (uint8_t *)prompt;
edit.prompt_length = strlen(prompt);
edit.image_paths = image_paths;
edit.image_path_lengths = image_lengths;
edit.image_count = 1;
edit.pixels = 512;
edit.steps = 6;
edit.seed = 1301;

QiStatus status = qi_generate_multi_image_to_png(&edit);
```

Here `pack`, `model_root`, `output_path`, and `prompt` are caller-owned C
strings. Keep them and both arrays alive until the function returns. The
image-conditioned C ABI has no cache setting.

## Native C+ clients

Source projects can import `qwen_image/api` directly. Its
`GenerateRequest`/`generate_to_png` and
`MultiImageSizedGenerateRequest`/`generate_multi_image_sized_to_png` calls
are the native equivalents; the sized edit API accepts rectangular output.
The native package is a source-level C+ dependency, not part of the installed
C ABI archive. See [`qwen_image/src/api.cplus`](../qwen_image/src/api.cplus)
for the exact types and [`generation_worker.cplus`](../qwen_image/src/generation_worker.cplus)
for a client that sends progress and cancellation events without blocking the
GUI main thread. The plain C ABI does not expose those worker events.
