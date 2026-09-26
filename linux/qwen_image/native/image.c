// Image input for the Linux engine: decoding, EXIF orientation, and
// Pillow-compatible resampling. The macOS engine uses ImageIO for this.
//
// JPEGs decode through the system libjpeg-turbo with Pillow's settings (ISLOW
// IDCT, fancy upsampling), which reproduces Pillow's pixels exactly; the VAE
// encoder visibly amplifies the 1-3 LSB differences of other JPEG decoders.
// Other formats (and CMYK JPEGs) use stb_image.
//
// qi_image_resize_lanczos reproduces Pillow's Resample.c for 8-bit images:
// Lanczos-3 coefficients whose support widens with the downscale factor,
// normalized per output pixel, converted to 22-bit fixed point, accumulated
// from a rounding offset and clipped; horizontal pass first over only the rows
// the vertical pass reads, then the vertical pass. Like Pillow, RGBA is resized
// premultiplied (RGBa) and unpremultiplied afterwards with its integer rules.

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#include <math.h>
#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <jpeglib.h>

#define PRECISION_BITS (32 - 8 - 2)

// MARK: - EXIF orientation

static unsigned read16(const unsigned char *p, int little) {
    return little ? (unsigned)(p[0] | (p[1] << 8)) : (unsigned)((p[0] << 8) | p[1]);
}

static unsigned read32(const unsigned char *p, int little) {
    return little ? (unsigned)(p[0] | (p[1] << 8) | (p[2] << 16) | ((unsigned)p[3] << 24))
                  : (unsigned)(((unsigned)p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
}

// The JPEG EXIF orientation tag (0x0112) from IFD0, or 1 when absent.
static int jpeg_orientation(const unsigned char *data, size_t size) {
    if (size < 4 || data[0] != 0xFF || data[1] != 0xD8) return 1;
    size_t at = 2;
    while (at + 4 <= size && data[at] == 0xFF) {
        const unsigned marker = data[at + 1];
        const size_t length = (size_t)((data[at + 2] << 8) | data[at + 3]);
        if (marker == 0xDA || length < 2) break;
        if (marker == 0xE1 && at + 4 + length - 2 <= size && length >= 14 && memcmp(data + at + 4, "Exif\0\0", 6) == 0) {
            const unsigned char *tiff = data + at + 10;
            const size_t tiff_size = length - 8;
            if (tiff_size < 8) return 1;
            const int little = tiff[0] == 'I';
            const unsigned ifd = read32(tiff + 4, little);
            if (ifd + 2 > tiff_size) return 1;
            const unsigned entries = read16(tiff + ifd, little);
            for (unsigned i = 0; i < entries; ++i) {
                const size_t entry = ifd + 2 + (size_t)i * 12;
                if (entry + 12 > tiff_size) break;
                if (read16(tiff + entry, little) == 0x0112) {
                    const unsigned value = read16(tiff + entry + 8, little);
                    return value >= 1 && value <= 8 ? (int)value : 1;
                }
            }
            return 1;
        }
        at += 2 + length;
    }
    return 1;
}

// Applies an EXIF orientation to RGBA pixels (Pillow's ImageOps.exif_transpose).
static unsigned char *orient(unsigned char *pixels, int *width, int *height, int orientation) {
    if (orientation <= 1) return pixels;
    const int w = *width, h = *height;
    const int swap = orientation >= 5;
    const int ow = swap ? h : w, oh = swap ? w : h;
    unsigned char *out = (unsigned char *)malloc((size_t)ow * oh * 4);
    if (!out) return pixels;
    for (int y = 0; y < oh; ++y) {
        for (int x = 0; x < ow; ++x) {
            int sx = x, sy = y;
            switch (orientation) {
                case 2: sx = w - 1 - x; sy = y; break;
                case 3: sx = w - 1 - x; sy = h - 1 - y; break;
                case 4: sx = x; sy = h - 1 - y; break;
                case 5: sx = y; sy = x; break;
                case 6: sx = y; sy = h - 1 - x; break;
                case 7: sx = w - 1 - y; sy = h - 1 - x; break;
                case 8: sx = w - 1 - y; sy = x; break;
            }
            memcpy(out + ((size_t)y * ow + x) * 4, pixels + ((size_t)sy * w + sx) * 4, 4);
        }
    }
    free(pixels);
    *width = ow;
    *height = oh;
    return out;
}

// MARK: - JPEG

struct jpeg_failure {
    struct jpeg_error_mgr manager;
    jmp_buf jump;
};

static void jpeg_fail(j_common_ptr info) {
    longjmp(((struct jpeg_failure *)info->err)->jump, 1);
}

// Decodes a JPEG into RGBA8 with libjpeg, or returns null (not a JPEG, CMYK,
// or corrupt) so the caller can fall back to stb_image.
static unsigned char *decode_jpeg(const unsigned char *data, size_t size, int *width, int *height) {
    if (size < 3 || data[0] != 0xFF || data[1] != 0xD8) return NULL;
    struct jpeg_decompress_struct info;
    struct jpeg_failure failure;
    unsigned char *volatile rgba = NULL;
    unsigned char *volatile row = NULL;
    info.err = jpeg_std_error(&failure.manager);
    failure.manager.error_exit = jpeg_fail;
    if (setjmp(failure.jump)) {
        jpeg_destroy_decompress(&info);
        free(rgba);
        free(row);
        return NULL;
    }
    jpeg_create_decompress(&info);
    jpeg_mem_src(&info, data, (unsigned long)size);
    jpeg_read_header(&info, TRUE);
    if (info.jpeg_color_space == JCS_CMYK || info.jpeg_color_space == JCS_YCCK) {
        jpeg_destroy_decompress(&info);
        return NULL;
    }
    info.out_color_space = JCS_RGB;
    info.dct_method = JDCT_ISLOW;
    info.do_fancy_upsampling = TRUE;
    jpeg_start_decompress(&info);
    const int w = (int)info.output_width, h = (int)info.output_height;
    rgba = (unsigned char *)malloc((size_t)w * h * 4);
    row = (unsigned char *)malloc((size_t)w * 3);
    if (!rgba || !row) longjmp(failure.jump, 1);
    while (info.output_scanline < info.output_height) {
        const int y = (int)info.output_scanline;
        JSAMPROW rows[1] = {row};
        jpeg_read_scanlines(&info, rows, 1);
        for (int x = 0; x < w; ++x) {
            unsigned char *out = rgba + ((size_t)y * w + x) * 4;
            out[0] = row[x * 3];
            out[1] = row[x * 3 + 1];
            out[2] = row[x * 3 + 2];
            out[3] = 255;
        }
    }
    jpeg_finish_decompress(&info);
    jpeg_destroy_decompress(&info);
    free(row);
    *width = w;
    *height = h;
    return rgba;
}

// Decodes a JPEG/PNG/... into RGBA8 with EXIF orientation applied. Returns a
// malloc'd buffer (free with qi_image_free) or null.
unsigned char *qi_image_load(const char *path, int *width, int *height) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (size <= 0) { fclose(file); return NULL; }
    unsigned char *data = (unsigned char *)malloc((size_t)size);
    if (!data || fread(data, 1, (size_t)size, file) != (size_t)size) { fclose(file); free(data); return NULL; }
    fclose(file);
    unsigned char *pixels = decode_jpeg(data, (size_t)size, width, height);
    if (!pixels) {
        int channels = 0;
        pixels = stbi_load_from_memory(data, (int)size, width, height, &channels, 4);
    }
    const int orientation = jpeg_orientation(data, (size_t)size);
    free(data);
    if (!pixels) return NULL;
    return orient(pixels, width, height, orientation);
}

void qi_image_free(unsigned char *pixels) {
    free(pixels);
}

// MARK: - Pillow resampling

static double sinc_filter(double x) {
    if (x == 0.0) return 1.0;
    x = x * M_PI;
    return sin(x) / x;
}

static double lanczos_filter(double x) {
    if (-3.0 <= x && x < 3.0) return sinc_filter(x) * sinc_filter(x / 3);
    return 0.0;
}

static int precompute_coeffs(int in_size, double in0, double in1, int out_size, int **bounds_out, int32_t **kk_out) {
    double scale = (in1 - in0) / out_size;
    double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = 3.0 * filterscale;
    const int ksize = (int)ceil(support) * 2 + 1;
    double *kk = (double *)malloc((size_t)out_size * ksize * sizeof(double));
    int *bounds = (int *)malloc((size_t)out_size * 2 * sizeof(int));
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = in0 + (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double *k = &kk[(size_t)xx * ksize];
        double ww = 0.0;
        int x = 0;
        for (; x < xmax; ++x) {
            const double w = lanczos_filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; ++x) if (ww != 0.0) k[x] /= ww;
        for (; x < ksize; ++x) k[x] = 0;
        bounds[xx * 2] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
    int32_t *fixed = (int32_t *)malloc((size_t)out_size * ksize * sizeof(int32_t));
    for (size_t i = 0; i < (size_t)out_size * ksize; ++i)
        fixed[i] = kk[i] < 0 ? (int32_t)(-0.5 + kk[i] * (1 << PRECISION_BITS)) : (int32_t)(0.5 + kk[i] * (1 << PRECISION_BITS));
    free(kk);
    *bounds_out = bounds;
    *kk_out = fixed;
    return ksize;
}

static unsigned char clip8(int64_t in) {
    if (in >= ((int64_t)1 << PRECISION_BITS << 8)) return 255;
    if (in <= 0) return 0;
    return (unsigned char)(in >> PRECISION_BITS);
}

// Pillow's MULDIV255 and DIV255 roundings.
static unsigned div255(unsigned v) {
    const unsigned tmp = v + 128;
    return ((tmp >> 8) + tmp) >> 8;
}

static unsigned char *premultiplied(const unsigned char *in, size_t pixels) {
    unsigned char *out = (unsigned char *)malloc(pixels * 4);
    if (!out) return NULL;
    for (size_t i = 0; i < pixels; ++i) {
        const unsigned alpha = in[i * 4 + 3];
        for (int c = 0; c < 3; ++c) out[i * 4 + c] = (unsigned char)div255(in[i * 4 + c] * alpha);
        out[i * 4 + 3] = (unsigned char)alpha;
    }
    return out;
}

static void unpremultiply(unsigned char *pixels, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const unsigned alpha = pixels[i * 4 + 3];
        if (alpha == 255 || alpha == 0) continue;
        for (int c = 0; c < 3; ++c) {
            const unsigned value = (255 * pixels[i * 4 + c]) / alpha;
            pixels[i * 4 + c] = (unsigned char)(value > 255 ? 255 : value);
        }
    }
}

// Composites RGBA8 over white into RGB8, as `white.paste(image, mask=alpha)`.
void qi_image_composite_white(const unsigned char *rgba, size_t pixels, unsigned char *rgb) {
    for (size_t i = 0; i < pixels; ++i) {
        const unsigned alpha = rgba[i * 4 + 3];
        for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = (unsigned char)div255(255 * (255 - alpha) + rgba[i * 4 + c] * alpha);
    }
}

static int resize_premultiplied(const unsigned char *in, int in_w, int in_h, unsigned char *out, int out_w, int out_h);

// Resizes RGBA8 `in` (in_w x in_h) to `out` (out_w x out_h) as Pillow's
// `Image.resize(..., LANCZOS)` does for an RGBA image. Returns 0 on success.
int qi_image_resize_lanczos(const unsigned char *in, int in_w, int in_h, unsigned char *out, int out_w, int out_h) {
    unsigned char *source = premultiplied(in, (size_t)in_w * in_h);
    if (!source) return 1;
    const int result = resize_premultiplied(source, in_w, in_h, out, out_w, out_h);
    free(source);
    if (result == 0) unpremultiply(out, (size_t)out_w * out_h);
    return result;
}

static int resize_premultiplied(const unsigned char *in, int in_w, int in_h, unsigned char *out, int out_w, int out_h) {
    int *bounds_h, *bounds_v;
    int32_t *kk_h, *kk_v;
    const int ksize_h = precompute_coeffs(in_w, 0, in_w, out_w, &bounds_h, &kk_h);
    const int ksize_v = precompute_coeffs(in_h, 0, in_h, out_h, &bounds_v, &kk_v);
    const int need_horizontal = out_w != in_w, need_vertical = out_h != in_h;
    const int ybox_first = bounds_v[0];
    const int ybox_last = bounds_v[out_h * 2 - 2] + bounds_v[out_h * 2 - 1];
    const unsigned char *source = in;
    int source_w = in_w;
    unsigned char *temp = NULL;
    int temp_h = in_h;
    if (need_horizontal) {
        for (int i = 0; i < out_h; ++i) bounds_v[i * 2] -= ybox_first;
        temp_h = ybox_last - ybox_first;
        temp = (unsigned char *)malloc((size_t)out_w * temp_h * 4);
        for (int yy = 0; yy < temp_h; ++yy) {
            const unsigned char *row = in + (size_t)(yy + ybox_first) * in_w * 4;
            for (int xx = 0; xx < out_w; ++xx) {
                const int xmin = bounds_h[xx * 2], xmax = bounds_h[xx * 2 + 1];
                const int32_t *k = &kk_h[(size_t)xx * ksize_h];
                int64_t ss[4] = {1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1)};
                for (int x = 0; x < xmax; ++x)
                    for (int c = 0; c < 4; ++c) ss[c] += (int64_t)row[(x + xmin) * 4 + c] * k[x];
                for (int c = 0; c < 4; ++c) temp[((size_t)yy * out_w + xx) * 4 + c] = clip8(ss[c]);
            }
        }
        source = temp;
        source_w = out_w;
    }
    if (need_vertical) {
        for (int yy = 0; yy < out_h; ++yy) {
            const int ymin = bounds_v[yy * 2], ymax = bounds_v[yy * 2 + 1];
            const int32_t *k = &kk_v[(size_t)yy * ksize_v];
            for (int xx = 0; xx < out_w; ++xx) {
                int64_t ss[4] = {1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1), 1 << (PRECISION_BITS - 1)};
                for (int y = 0; y < ymax; ++y)
                    for (int c = 0; c < 4; ++c) ss[c] += (int64_t)source[((size_t)(y + ymin) * source_w + xx) * 4 + c] * k[y];
                for (int c = 0; c < 4; ++c) out[((size_t)yy * out_w + xx) * 4 + c] = clip8(ss[c]);
            }
        }
    } else {
        memcpy(out, source, (size_t)out_w * out_h * 4);
    }
    free(temp);
    free(bounds_h); free(bounds_v); free(kk_h); free(kk_v);
    return 0;
}
