#include <iostream>
#include <omp.h>
#include <vector>
#include <fstream>
#include <string>

using namespace std;

int sub(int x, int y) {
    return min(255, (max(0, x - y)));
}

inline void
collect(vector<unsigned char> &data, vector<unsigned char> &processed, ifstream& in, int channels, unsigned int width,
        unsigned int height,
        double threshold, int num_threads) {
    omp_set_num_threads(num_threads);
    int mn = 256, mx = -1;
    int red[256] = { 0 }, green[256] = { 0 }, blue[256] = { 0 };
    int size = width * height * channels;

    int step = channels * 16 * 1024;
    int chunk_size = 16384;
    // count the amount of pixels of each value from 0 to 255
#pragma omp parallel for shared(channels, data, step, red, green, blue, height, width, size, chunk_size) default(none) schedule(static, chunk_size)
    for (int i = 0; i < size; i += step) {
        int temp_red[256] = { 0 };
        int temp_green[256] = { 0 };
        int temp_blue[256] = { 0 };
        for (int k = i; k < min((int)size, i + step); k += channels) {
            temp_red[data[k]] += 1;
            if (channels == 3) {
                temp_green[data[k + 1]] += 1;
                temp_blue[data[k + 2]] += 1;
            }
        }
#pragma omp critical
        for (int k = 0; k < 256; k++) {
            red[k] += temp_red[k];
            green[k] += temp_green[k];
            blue[k] += temp_blue[k];
        }
    }
    // Find segment boundaries depending on a given threshold
    vector<int> sum_min(channels);
    vector<int> sum_max(channels);
    int n = threshold * width * height;
    for (int i = 0; i < 256; i++) {
        sum_min[0] += red[i];
        sum_max[0] += red[255 - i];
        if (channels == 3) {
            sum_min[1] += green[i];
            sum_min[2] += blue[i];
            sum_max[1] += green[255 - i];
            sum_max[2] += blue[255 - i];
        }
        else {
            sum_min[0] += green[i];
            sum_min[0] += blue[i];
            sum_max[0] += green[255 - i];
            sum_max[0] += blue[255 - i];
        }
        if (mn == 256 && (sum_min[0] > n || (channels == 3 && (sum_min[1] > n || sum_min[2] > n)))) {
            mn = i;
            if (mx != -1) break;
        }

        if (mx == -1 && (sum_max[0] > n || (channels == 3 && (sum_max[1] > n || sum_max[2] > n)))) {
            mx = 255 - i;
            if (mn != 256) break;
        }
    }
    // Precalculate new pixel values
    int colormap[256];
    for (int i = 0; i < 256; i++) {
        if (mn == mx) colormap[i] = mx;
        else if (i <= mn) colormap[i] = 0;
        else if (i >= mx) colormap[i] = 255;
        else colormap[i] = 255.0 * sub(i, mn) / sub(mx, mn);
    }

#pragma omp parallel for shared(processed, data, mx, mn, colormap, size, step, chunk_size) default(none) schedule(static, chunk_size)
    for (int i = 0; i < size; i++) {
        processed[i] = colormap[data[i]];
    }
}


int main(int argc, char* argv[]) {
    if (argc < 5) {
        cout << "Too few arrguments" << endl;
        return 1;
    }
    int number_of_threads;
    double threshold;
    try {
        number_of_threads = stoi(argv[1]);
        threshold = stof(argv[4]);
    }
    catch (invalid_argument e) {
        cout << "Invalid value" << endl;
        return 1;
    }
    number_of_threads = number_of_threads ? number_of_threads : omp_get_max_threads();
    string inputname = argv[2];
    string outputname = argv[3];

    ifstream in(inputname, ios::binary);
    if (in.fail()) {
        cout << "File not found";
        return 1;
    }
    ofstream out(outputname, ios::binary);
    if (out.fail()) {
        cout << "Error" << endl;
        return 1;
    }

    // Read header
    unsigned char format[2];
    unsigned int width, height, max_value;
    in >> format[0] >> format[1];
    if (format[0] != 'P' || (format[1] != '5' && format[1] != '6')) {
        cout << "Invalid file" << endl;
        return 1;
    }
    int channels = 3;
    if (format[1] == '5') channels = 1;

    in >> width >> height >> max_value;
    in.get();
    int size = width * height * channels;

    // Read data
    vector<unsigned char> arr(size);
    in.read(reinterpret_cast<char*>(&arr[0]), size);

    vector<unsigned char> processed(size);
    auto start = omp_get_wtime();

    // Process data
    collect(arr, processed, in, channels, width, height, threshold, number_of_threads);
    auto stop = omp_get_wtime();
    auto duration = stop - start;
    cout << "Time (" << number_of_threads << " thread(s)): " << duration * 1000 << " ms" << endl;

    // Write
    out << format[0] << format[1] << '\n' << width << ' ' << height << '\n' << max_value << '\n';
    out.write(reinterpret_cast<char*>(&processed[0]), size);

    in.close();
    out.close();
    return 0;

}
