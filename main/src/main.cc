#include "SG14/inplace_function.h"
#include "fmt/core.h"

#include <Magick++.h>
#include <cassert>
#include <filesystem>
#include <limits>
#include <unordered_map>
#include <variant>

namespace {

using Magick::Quantum;

// B&W dithering needs the corresponding value for white
constexpr auto G_QUANTUM_WHITE_VALUE = std::numeric_limits<Quantum>::max();

using BayesKernel = std::array<Quantum, 64>;

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

template<typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
[[nodiscard]] constexpr auto
modulo_mask_pow2(int power)
{
    T result = 0;
    for (int i = 0; i < power; ++i) {
        result |= (1 << i);
    }

    return result;
}

template<int SIZE>
[[nodiscard]] constexpr auto
make_bayes_kernel()
{
    constexpr std::array<BayesKernel, 3> bayes_kernels = {
        BayesKernel{0, 2, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        BayesKernel{0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5,
                    0, 0, 0, 0,  0,  0, 0,  0, 0, 0,  0, 0, 0,  0, 0,  0,
                    0, 0, 0, 0,  0,  0, 0,  0, 0, 0,  0, 0, 0,  0, 0,  0,
                    0, 0, 0, 0,  0,  0, 0,  0, 0, 0,  0, 0, 0,  0, 0,  0},
        BayesKernel{0,  32, 8,  40, 2,  34, 10, 42, 48, 16, 56, 24, 50,
                    18, 58, 26, 12, 44, 4,  36, 14, 46, 6,  38, 60, 28,
                    52, 20, 62, 30, 54, 22, 3,  35, 11, 43, 1,  33, 9,
                    41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7,  39,
                    13, 45, 5,  37, 63, 31, 55, 23, 61, 29, 53, 21}};

    constexpr auto log2_size = simple_log2(SIZE);
    constexpr auto index = log2_size - 1;
    BayesKernel    kernel = bayes_kernels[index];
    for (int i = 0; i < SIZE * SIZE; ++i) {
        // Table values are scaled by the table size, i.e. by
        // the square of the table width, and need to be normalized
        // before use. Since the table widths are powers of two it is
        // possible to optimize dividing by their squares into shifting
        // right by twice their log_2 value.
        uint_fast32_t x = kernel[i];
        kernel[i] = static_cast<Quantum>((x * G_QUANTUM_WHITE_VALUE) >>
                                         (log2_size * 2));
    }

    return kernel;
}

void
enhance(Magick::Image & image, float brightness, float contrast)
{
    image.brightnessContrast(brightness, contrast);

    constexpr float sharp_gauss_radius = 1.0F;
    constexpr float sharp_laplacian_dev = 1.0F;
    image.sharpen(sharp_gauss_radius, sharp_laplacian_dev);
}

void
dither(Magick::Image & image, int kernel_size)
{
    struct PixelCoord {
        std::size_t x;
        std::size_t y;
    };

    // Get image dimensions
    auto const width = image.columns();
    auto const height = image.rows();
    auto const n_channels = image.channels();

    fmt::print("Image dimensions: {}x{}@{}bpp\n",
               width,
               height,
               n_channels * std::numeric_limits<Quantum>::digits);

    // Prepare image for writing
    image.modifyImage();
    auto * const data = image.getPixels(0, 0, width, height);

    // Pipeline step 1
    auto to_luminance = [](Quantum const * channels) -> Quantum {
        // Assume RGB(A)
        auto const red = channels[0];
        auto const green = channels[1];
        auto const blue = channels[2];

        // Assume saturated alpha, and approximate luminance as
        // 0.375*R + 0.5*G + 0.125*B
        Quantum lum = (3 * red + blue + 4 * green) >> 3;

        return lum;
    };

    // Select Bayes kernel
    auto const kernel = std::invoke([kernel_size]() {
        switch (kernel_size) {
            case 2: return make_bayes_kernel<2>();
            case 4: return make_bayes_kernel<4>();
            case 8: return make_bayes_kernel<8>();
            default: throw std::runtime_error("Invalid kernel size");
        }
    });
    auto const kernel_mod_mask =
        modulo_mask_pow2<std::size_t>(simple_log2(kernel_size));

    auto bayes_function = [&kernel, kernel_size, kernel_mod_mask](
                              PixelCoord coord,
                              Quantum *  pixels,
                              Quantum    luminance) {
        auto const x = coord.x & kernel_mod_mask;
        auto const y = coord.y & kernel_mod_mask;

        pixels[0] = static_cast<int>(luminance > kernel[y * kernel_size + x]) *
                    G_QUANTUM_WHITE_VALUE;
    };

    if (n_channels > 1) {
        // Compose pipeline
        auto pipeline = [&](PixelCoord coord, Quantum * pixels) {
            auto const lum = to_luminance(pixels);
            bayes_function(coord, pixels, lum);
        };

        // Run pipeline
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                std::size_t const index = n_channels * (y * width + x);
                pipeline(PixelCoord{x, y}, &data[index]);
            }
        }
    } else {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                std::size_t const index = n_channels * (y * width + x);
                bayes_function(PixelCoord{x, y}, &data[index], data[index]);
            }
        }
    }

    // Collect data
    image.syncPixels();
}

void
write_output(std::filesystem::path const & out_file, Magick::Image & image)
{
    std::string const extension = out_file.extension();
    auto const        without_dot = std::string_view{extension}.substr(1);

    fmt::print("Saving as [{}] at {}\n", without_dot, out_file.string());

    // Extract relevant channel
    image.channel(MagickCore::ChannelType::RedChannel);

    // Remove any incompatible color profiles
    image.strip();

    image.write(out_file.string());
}

} // namespace

auto
main(int argc, char ** argv) -> int
{
    if (argc != 6) {
        fmt::print(stderr,
                   "Usage: {} "
                   "<input_image> <output_image> <kernel_size (i)> "
                   "<brightness (f)> <contrast (f)>\n",
                   argv[0]);
        return -1;
    }

    try {
        // Parse parameters
        std::filesystem::path in_file{argv[1]};
        std::filesystem::path out_file{argv[2]};

        auto const kernel_size =
            static_cast<int>(std::strtol(argv[3], nullptr, 10));
        float const brightness = std::strtof(argv[4], nullptr);
        float const contrast = std::strtof(argv[5], nullptr);

        Magick::InitializeMagick(argv[0]);

        // Load image
        Magick::Image image{in_file.c_str()};

        // Process image
        enhance(image, brightness, contrast);
        dither(image, kernel_size);

        // Done
        write_output(out_file, image);

        return 0;
    } catch (std::exception const & e) {
        fmt::print(stderr, "{}\n", e.what());

        return -1;
    }
}
