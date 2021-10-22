#include "fmt/core.h"

#include <filesystem>
#include <functional>
#include <limits>
#include <type_traits>
#include <vips/vips8>

namespace {

using Image = vips::VImage;

using BayerKernel = std::array<uint8_t, 64>;

template<typename T>
[[nodiscard]] constexpr auto
simple_log2(T x) -> int
{
    for (int i = 0; i < std::numeric_limits<T>::digits; ++i) {
        if (((x >> i) & 0x1) != 0) {
            return i;
        }
    }

    return -1;
}

template<typename T, typename U = std::remove_cv_t<T>>
[[nodiscard]] constexpr auto
modulo_mask_pow2(int power)
{
    U result = 0;
    for (int i = 0; i < power; ++i) {
        result |= (1 << i);
    }

    return result;
}

template<int SIZE>
[[nodiscard]] constexpr auto
make_bayer_kernel()
{
    constexpr std::array<BayerKernel, 3> bayer_kernels = {
        BayerKernel{0, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        BayerKernel{0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5,
                    0, 0, 0, 0,  0,  0, 0,  0, 0, 0,  0, 0, 0,  0, 0,  0,
                    0, 0, 0, 0,  0,  0, 0,  0, 0, 0,  0, 0, 0,  0, 0,  0,
                    0, 0, 0, 0,  0,  0, 0,  0, 0, 0,  0, 0, 0,  0, 0,  0},
        BayerKernel{0,  32, 8,  40, 2,  34, 10, 42, 48, 16, 56, 24, 50,
                    18, 58, 26, 12, 44, 4,  36, 14, 46, 6,  38, 60, 28,
                    52, 20, 62, 30, 54, 22, 3,  35, 11, 43, 1,  33, 9,
                    41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7,  39,
                    13, 45, 5,  37, 63, 31, 55, 23, 61, 29, 53, 21}};

    constexpr auto log2_size = simple_log2(SIZE);
    constexpr auto index = log2_size - 1;
    BayerKernel    kernel = bayer_kernels[index];

    constexpr auto max = std::numeric_limits<uint8_t>::max();
    for (int i = 0; i < SIZE * SIZE; ++i) {
        // Table values are scaled by the table size, i.e. by
        // the square of the table width, and need to be normalized
        // before use. Since the table widths are powers of two it is
        // possible to optimize dividing by their squares into shifting
        // right by twice their log_2 value.
        uint_fast32_t x = kernel[i];
        kernel[i] = static_cast<uint8_t>((x * max) >> (log2_size * 2));
    }

    return kernel;
}

struct EnhancementParams {
    int   brightness;
    float contrast;
};

void
enhance(Image & image, EnhancementParams const & p)
{
    image += p.brightness;
    image = image * p.contrast + 255.0 * (1.0 - p.contrast);

    constexpr float sharp_gauss_radius = 1.0F;
    constexpr float sharp_laplacian_dev = 1.0F;
    image.sharpen(Image::option()
                      ->set("radius", sharp_gauss_radius)
                      ->set("sigma", sharp_laplacian_dev));
}

template<typename Format>
void
dither(Image & image, int kernel_size)
{
    struct PixelCoord {
        int x;
        int y;
    };

    // Get image dimensions
    auto const width = image.width();
    auto const height = image.height();

    vips_image_inplace(image.get_image());
    VipsRegion * region = vips_region_new(image.get_image());
    VipsRect     full_image{0, 0, width, height};
    if (vips_region_prepare(region, &full_image) < 0) {
        throw std::runtime_error("Unable to prepare vips region");
    }

    // Pipeline step 1
    auto to_luminance = [](Format const * p) -> Format {
        // Assume RGB with saturated alpha, and approximate luminance as
        // 0.375*R + 0.5*G + 0.125*B
        return (3 * p[0] + p[2] + 4 * p[1]) / 8;
    };

    // Select Bayer kernel
    auto const kernel = std::invoke([kernel_size]() {
        switch (kernel_size) {
            case 2: return make_bayer_kernel<2>();
            case 4: return make_bayer_kernel<4>();
            case 8: return make_bayer_kernel<8>();
            default: throw std::runtime_error("Invalid kernel size");
        }
    });
    auto const kernel_mod_mask =
        modulo_mask_pow2<decltype(width)>(simple_log2(kernel_size));

    auto bayer_function =
        [&kernel, kernel_size, kernel_mod_mask](
            PixelCoord coord, Format luminance, Format * out_p) {
            auto const x = coord.x & kernel_mod_mask;
            auto const y = coord.y & kernel_mod_mask;

            auto const mask_value = kernel[y * kernel_size + x];
            auto const result = (luminance > mask_value)
                                    ? std::numeric_limits<Format>::max()
                                    : static_cast<Format>(0);

            // Only write to a single channel
            out_p[0] = result;
        };

    // Compose pipeline
    auto pipeline = [&](PixelCoord coord, auto * pixel) {
        auto const lum = to_luminance(pixel);
        bayer_function(coord, lum, pixel);
    };

    // Run pipeline
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Casting from Format* to VipsPel* to void* to Format* is probably UB
            void * data = VIPS_REGION_ADDR(region, x, y);
            pipeline(PixelCoord{x, y}, static_cast<Format *>(data));
        }
    }
}

void
write_output(Image & image, std::filesystem::path const & out_file)
{
    std::string const extension = out_file.extension();
    auto const        without_dot = std::string_view{extension}.substr(1);

    fmt::print("Saving as [{}] at {}\n", without_dot, out_file.string());

    // Extract relevant channel
    image = image[0];

    if (without_dot == "png" || without_dot == "PNG") {
        image.pngsave(
            out_file.c_str(),
            Image::option()->set("bitdepth", 1)->set("compression", 9));
    } else {
        image.write_to_file(out_file.c_str());
    }
}

} // namespace

auto
main(int argc, char ** argv) -> int
{
    if (argc != 6) {
        fmt::print(stderr,
                   "Usage: {} "
                   "<input_image> <output_image> <kernel_size (i)> "
                   "<brightness (i)> <contrast (f)>\n",
                   argv[0]);
        return -1;
    }

    try {
        // Parse arguments
        std::filesystem::path in_file{argv[1]};
        std::filesystem::path out_file{argv[2]};

        auto const kernel_size =
            static_cast<int>(std::strtol(argv[3], nullptr, 10));
        auto const brightness =
            static_cast<int>(std::strtol(argv[4], nullptr, 10));
        auto const contrast = std::strtof(argv[5], nullptr);

        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
        if (VIPS_INIT(argv[0]) < 0) {
            fmt::print(stderr, "Unable to initialize libvips\n");
            return -1;
        }

        // Load image
        auto image = Image::new_from_file(in_file.c_str());
        image.colourspace(VipsInterpretation::VIPS_INTERPRETATION_sRGB);

        // Process image
        enhance(image, {brightness, contrast});

        auto error_func = [] {
            throw std::runtime_error("Unsupported image format");
        };

        auto const format = image.format();
        switch (format) {
            case VIPS_FORMAT_NOTSET: error_func(); break;
            case VIPS_FORMAT_UCHAR:
                dither<unsigned char>(image, kernel_size);
                break;
            case VIPS_FORMAT_CHAR: error_func(); break;
            case VIPS_FORMAT_USHORT:
                // NOLINTNEXTLINE(google-runtime-int)
                dither<unsigned short>(image, kernel_size);
                break;
            case VIPS_FORMAT_SHORT:
                // NOLINTNEXTLINE(google-runtime-int)
                dither<short>(image, kernel_size);
                break;
            case VIPS_FORMAT_UINT: dither<unsigned>(image, kernel_size); break;
            case VIPS_FORMAT_INT: dither<int>(image, kernel_size); break;
            case VIPS_FORMAT_FLOAT: dither<float>(image, kernel_size); break;
            case VIPS_FORMAT_COMPLEX: error_func(); break;
            case VIPS_FORMAT_DOUBLE: dither<double>(image, kernel_size); break;
            case VIPS_FORMAT_DPCOMPLEX:
            case VIPS_FORMAT_LAST: error_func(); break;
        }

        // Done
        write_output(image, out_file);

        return 0;
    } catch (std::exception const & e) {
        fmt::print(stderr, "{}\n", e.what());

        return -1;
    }
}
