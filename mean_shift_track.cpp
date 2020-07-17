// Mean shift for video tracking
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <vector>
#include <omp.h>

#define PI 3.1415926
#define COLOR_BIN_WIDTH 16 // the with of RGB color bin
#define MEAN_SHIFT_MAX_ITER 10
#define EPSILON 0.000001
#define CONVERT_B_BIN_INDEX(x) (x / COLOR_BIN_WIDTH)
#define CONVERT_G_BIN_INDEX(x) (256 / COLOR_BIN_WIDTH + x / COLOR_BIN_WIDTH)
#define CONVERT_R_BIN_INDEX(x) (256 / COLOR_BIN_WIDTH * 2 + x / COLOR_BIN_WIDTH)
#define TOTAL_COLOR_BIN_NUM (256 / COLOR_BIN_WIDTH * 3)
using namespace cv;
using namespace std;

Mat one_pos_colr_hist(Mat img, int center_pos_row, int center_pos_col, int row_window_size, int col_window_size)
{
    int row_low = MAX(0, center_pos_row - row_window_size);
    int row_high = MIN(center_pos_row + row_window_size, img.rows);
    int col_low = MAX(0, center_pos_col - col_window_size);
    int col_high = MIN(center_pos_col + col_window_size, img.cols);

    Mat res(1, TOTAL_COLOR_BIN_NUM, CV_64F);
    res.convertTo(res, CV_64F, 0);
    double norm = 0;

    Mat gaussian_kernel_row = getGaussianKernel(row_window_size, 1.0, CV_64F);
    Mat gaussian_kernel_col = getGaussianKernel(col_window_size, 1.0, CV_64F);
    Mat gkernel = gaussian_kernel_row * gaussian_kernel_col.t();

    for (int i = row_low; i < row_high; ++i)
    {
        for (int j = col_low; j < col_high; ++j)
        {
            int b = (int)img.at<Vec3b>(i, j)[0], g = (int)img.at<Vec3b>(i, j)[1], r = (int)img.at<Vec3b>(i, j)[2];
            res.at<double>(CONVERT_B_BIN_INDEX(b)) += gkernel.at<double>(i - center_pos_row + row_window_size, j - center_pos_col + col_window_size);
            res.at<double>(CONVERT_G_BIN_INDEX(g)) += gkernel.at<double>(i - center_pos_row + row_window_size, j - center_pos_col + col_window_size);
            res.at<double>(CONVERT_R_BIN_INDEX(r)) += gkernel.at<double>(i - center_pos_row + row_window_size, j - center_pos_col + col_window_size);
            norm += gkernel.at<double>(i - center_pos_row + row_window_size, j - center_pos_col + col_window_size) * 3;
        }
    }

    res.convertTo(res, CV_64F, 1.0 / norm);
    return res;
}

/**
 * Calculate color p centered at y
 * The result contains result along row and along col
 */
vector<vector<vector<double>>> calc_p(Mat img, int center_pos_row, int center_pos_col, int row_window_size, int col_window_size)
{
    int row_low = MAX(0, center_pos_row - row_window_size);
    int row_high = MIN(center_pos_row + row_window_size, img.rows);
    int col_low = MAX(0, center_pos_col - col_window_size);
    int col_high = MIN(center_pos_col + col_window_size, img.cols);

    vector<vector<vector<double>>> window_res(row_window_size * 2 + 1, vector<vector<double>>(col_window_size * 2 + 1, vector<double>(TOTAL_COLOR_BIN_NUM, 0)));
#pragma omp parallel for
    for (int i = row_low; i < row_high; ++i)
    {
        for (int j = col_low; j < col_high; ++j)
        {
            // copy
            Mat one_pos_colr = one_pos_colr_hist(img, center_pos_row, center_pos_col, row_window_size, col_window_size);
            for (int k = 0; k < TOTAL_COLOR_BIN_NUM; ++k)
                window_res[i - center_pos_row + row_window_size][j - center_pos_col + col_window_size][k] = one_pos_colr.at<double>(0, k);
        }
    }
    return window_res;
}

vector<vector<vector<double>>> calc_q(Mat img, int center_pos_row, int center_pos_col, int row_window_size, int col_window_size)
{
    return calc_p(img, center_pos_row, center_pos_col, row_window_size, col_window_size);
}

/**
 * Calculating the row and col bhat coefficient
 */
double bhatt_coefficient(vector<double> colr_p, Mat model_q)
{
    double coef = 0;
    for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
    {
        coef += sqrt(colr_p[i] * model_q.at<double>(0, i));
    }
    return coef;
}

/**
 * The weight for one pixel
 */
double weight(vector<double> colr_p, Mat mode_q)
{
    double w = 0;
    for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
    {
        w += sqrt(mode_q.at<double>(0, i) / (colr_p[i] + EPSILON));
    }
    return w;
}

/**
 * calculate new location z
 */
Vec2i new_loc_z(Mat img, int center_pos_row, int center_pos_col, int row_window_size, int col_window_size, vector<vector<vector<double>>> window_colr_p, Mat model_q)
{
    int row_low = center_pos_row - row_window_size;
    int row_high = center_pos_row + row_window_size;
    int col_low = center_pos_col - col_window_size;
    int col_high = center_pos_col + col_window_size;

    row_low = MAX(0, row_low);
    row_high = MIN(row_high, img.rows);
    col_low = MAX(0, col_low);
    col_high = MIN(col_high, img.cols);

    Vec2i new_pos;
    double den = 0, ne1 = 0, ne2 = 0;

    for (int i = row_low; i < row_high; ++i)
    {
        for (int j = col_low; j < col_high; ++j)
        {
            double w = weight(window_colr_p[i - center_pos_row + row_window_size][j - center_pos_col + col_window_size], model_q);
            den += w;
            ne1 += i * w;
            ne2 += j * w;
        }
    }
    new_pos[0] = (int)(ne1 / den);
    new_pos[1] = (int)(ne2 / den);
    return new_pos;
}

/**
 * Update and track objects in video
 * @param img opencv mat format RGB image
 * @param model_q the color mode of target
 * @param prev_pos_row the position of target in previous frame
 * @param prev_pos_col the position of target in previous frame
 * @param window_size the calculated scale
 */
Vec2i mean_shift_track_update(Mat img, Mat mode_q, int prev_pos_row, int prev_pos_col, int row_window_size, int col_window_size)
{
    for (int epoch = 0; epoch < MEAN_SHIFT_MAX_ITER; ++epoch)
    {
        vector<vector<vector<double>>> window_colr_p = calc_p(img, prev_pos_row, prev_pos_col, row_window_size, col_window_size);
        Vec2i new_pos = new_loc_z(img, prev_pos_row, prev_pos_col, row_window_size, col_window_size, window_colr_p, mode_q);
        vector<double> new_loc_colr_p = window_colr_p[new_pos[0]-prev_pos_row+row_window_size][new_pos[1]-prev_pos_col+col_window_size];
        printf("\t in update, first calc new pos=(%d,%d)\n", new_pos[0], new_pos[1]);
        double prev_bhat = bhatt_coefficient(window_colr_p[row_window_size][col_window_size], mode_q),
               new_bhat = bhatt_coefficient(new_loc_colr_p, mode_q);
        printf("\t in update, first calc new pos=(%d,%d), prev_coef=%lf, new coef=%lf\n", new_pos[0], new_pos[1], prev_bhat, new_bhat);
        printf("center_pos color p  = \n");
        for(int i=0;i<TOTAL_COLOR_BIN_NUM;++i)
        {
            printf("%lf\t",window_colr_p[row_window_size][col_window_size][i]);
        }
        printf("\n");
        exit(1);


        while (new_bhat < prev_bhat)
        {
            new_pos[0] = (int)round((new_pos[0] + prev_pos_row) / 2.0);
            new_pos[1] = (int)round((new_pos[1] + prev_pos_col) / 2.0);
            if (abs(new_pos[0] - prev_pos_row) <= 1 && abs(new_pos[1] - prev_pos_col) <= 1)
                break;
            Mat new_colr_p = one_pos_colr_hist(img, new_pos[0], new_pos[1], row_window_size, col_window_size);
            new_bhat = bhatt_coefficient(new_colr_p, mode_q);
        }

        if (abs(new_pos[0] - prev_pos_row) <= 1 && abs(new_pos[1] - prev_pos_col) <= 1)
            return new_pos;
        prev_pos_row = new_pos[0];
        prev_pos_col = new_pos[1];
    }
}

/**
 * Normalize BGR format color image
 * @c (r',g',b')=(r,g,b)/(r+g+b)
 */
Mat bgr_img_norm_color(Mat img)
{
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            int r = (int)img.at<Vec3b>(i, j)[2], g = (int)img.at<Vec3b>(i, j)[1], b = (int)img.at<Vec3b>(i, j)[0];
            int tmp_sum = r + g + b + EPSILON;
            r = (int)(r * 1.0 / tmp_sum * 255);
            g = (int)(g * 1.0 / tmp_sum * 255);
            b = (int)(b * 1.0 / tmp_sum * 255);
            img.at<Vec3b>(i, j) = Vec3b((uchar)b, (uchar)g, (uchar)r);
        }
    }
    return img;
}

void generate_jpgs(char *video_name = (char *)"mot.avi", int max_count = 40, int step = 1)
{
    VideoCapture capture(video_name);
    Mat frame;
    capture >> frame;
    int index = 0, count = 0;
    while (!frame.empty())
    {
        if (count % step == 0)
        {
            char *name = new char[20];
            sprintf(name, "imgs/f%02d.jpg", index++);
            imwrite(name, frame);
            delete[] name;
        }
        capture >> frame;
        count++;
        if (count > max_count)
            break;
    }
}

int main(int argc, char const *argv[])
{
    const int num_frames = 10;
    char *frame_names[num_frames];
    for (int i = 0; i < num_frames; ++i)
    {
        frame_names[i] = new char[10];
        sprintf(frame_names[i], "./imgs/f%02d.jpg", i);
    }

    Mat frames[num_frames];
    for (int i = 0; i < num_frames; ++i)
    {
        frames[i] = imread(frame_names[i]);
        resize(frames[i], frames[i], Size(512, 348));
    }

    int center_row = 180, center_col = 380, row_width = 50, col_width = 10;
    Mat target_mode_q = one_pos_colr_hist(frames[0], center_row, center_col, row_width, col_width);
    Vec2i center_loc(center_row, center_col);

    for (int i = 0; i < num_frames; ++i)
    {
        center_loc = mean_shift_track_update(frames[i], target_mode_q, center_loc[0], center_loc[1], row_width, col_width);
        printf("new obj center_loc = %d, %d\n", center_loc[0], center_loc[1]);
        char *nm = new char[10];
        sprintf(nm, "img %d", i);
        Mat newframe = frames[i];
        for (int row = center_loc[0] - row_width; row < center_loc[0] + row_width; ++row)
        {
            for (int col = center_loc[1] - col_width; col < center_loc[1] + col_width; ++col)
            {
                newframe.at<Vec3b>(row, col)[0] = 180;
            }
        }
        if (i > 7)
            imshow(nm, newframe);
        delete[] nm;
    }
    for (int i = 0; i < num_frames; ++i)
    {
        delete frame_names[i];
    }
    waitKey();
    return 0;
}