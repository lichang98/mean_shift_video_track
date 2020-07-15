// Mean shift for video tracking
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.1415926
#define COLOR_BIN_WIDTH 30 // the with of RGB color bin
#define MEAN_SHIFT_MAX_ITER 10
#define EPSILON 0.000001
#define TOTAL_COLOR_BIN_NUM ((255 / COLOR_BIN_WIDTH + 1) * 3) // assume RGB
#define CONVERT_RED_TO_BIN_INDEX(x) (x / COLOR_BIN_WIDTH + 2 * (255 / COLOR_BIN_WIDTH + 1))
#define CONVERT_GREEN_TO_BIN_INDEX(x) (x / COLOR_BIN_WIDTH + (255 / COLOR_BIN_WIDTH + 1))
#define CONVERT_BLUE_TO_BIN_INDEX(x) (x / COLOR_BIN_WIDTH)

using namespace cv;

double epan_kernel_profile(double x)
{
    if (abs(x) <= 1.0)
        return 1 - x;
    else
        return 0;
}

double norm_distance(int x1, int x2, int y1, int y2, int h1, int h2)
{
    return sqrt(pow((x1 - y1) * 2.0 / h1, 2) + pow((x2 - y2) * 2.0 / h2, 2));
}

/**
 * Calculate color p centered at y
 * The result contains result along row and along col
 */
double *calc_p(Mat img, int center_pos_row, int center_pos_col, int row_window_size, int col_window_size)
{
    // assume image is RGB 3 color
    double *res = new double[TOTAL_COLOR_BIN_NUM];
    for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
        res[i] = 0;
    int row_low = MAX(0, center_pos_row - row_window_size);
    int row_high = MIN(center_pos_row + row_window_size, img.rows);
    int col_low = MAX(0, center_pos_col - col_window_size);
    int col_high = MIN(center_pos_col + col_window_size, img.cols);

    double norm = 0;

    for (int i = row_low; i < row_high; ++i)
    {
        for (int j = col_low; j < col_high; ++j)
        {
            int b = (int)img.at<Vec3b>(i, j)[0], g = (int)img.at<Vec3b>(i, j)[1], r = (int)img.at<Vec3b>(i, j)[2];
            res[CONVERT_BLUE_TO_BIN_INDEX(b)] += epan_kernel_profile(norm_distance(i, j, center_pos_row, center_pos_col, row_window_size, col_window_size));
            res[CONVERT_GREEN_TO_BIN_INDEX(g)] += epan_kernel_profile(norm_distance(i, j, center_pos_row, center_pos_col, row_window_size, col_window_size));
            res[CONVERT_RED_TO_BIN_INDEX(r)] += epan_kernel_profile(norm_distance(i, j, center_pos_row, center_pos_col, row_window_size, col_window_size));
            norm += epan_kernel_profile(norm_distance(i, j, center_pos_row, center_pos_col, row_window_size, col_window_size)) * 3;
        }
    }
    for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
        res[i] /= norm;
    return res;
}

double *calc_q(Mat img, int center_pos_row, int center_pos_col, int row_window_size, int col_window_size)
{
    return calc_p(img, center_pos_row, center_pos_col, row_window_size, col_window_size);
}

/**
 * Calculating the row and col bhat coefficient
 */
double bhatt_coefficient(double *colr_p, double *model_q)
{
    double coef = 0;
    for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
        coef += sqrt(colr_p[i] * model_q[i]);
    return coef;
}

/**
 * The weight for one pixel
 */
double weight(Vec3b pixel, double *colr_p, double *mode_q)
{
    int b = (int)pixel[0], g = (int)pixel[1], r = (int)pixel[2];
    double w = 0;
    int b_bin = CONVERT_BLUE_TO_BIN_INDEX(b), g_bin = CONVERT_GREEN_TO_BIN_INDEX(g), r_bin = CONVERT_RED_TO_BIN_INDEX(r);
    if (colr_p[b_bin] > 0)
        w += sqrt(mode_q[b_bin] / colr_p[b_bin]);
    else
        w += sqrt(mode_q[b_bin] / EPSILON);
    if (colr_p[g_bin] > 0)
        w += sqrt(mode_q[g_bin] / colr_p[g_bin]);
    else
        w += sqrt(mode_q[g_bin] / EPSILON);
    if (colr_p[r_bin] > 0)
        w += sqrt(mode_q[r_bin] / colr_p[r_bin]);
    else
        w += sqrt(mode_q[r_bin] / EPSILON);

    return w;
}

/**
 * calculate new location z
 */
Vec2i new_loc_z(Mat img, int center_pos_row, int center_pos_col, int row_window_size, int col_window_size, double *colr_p, double *model_q)
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
    double den1 = 0, den2 = 0, ne1 = 0, ne2 = 0;

    for (int i = row_low; i < row_high; ++i)
    {
        for (int j = col_low; j < col_high; ++j)
        {
            den1 += weight(img.at<Vec3b>(i, j), colr_p, model_q);
            den2 += weight(img.at<Vec3b>(i, j), colr_p, model_q);
            ne1 += i * weight(img.at<Vec3b>(i, j), colr_p, model_q);
            ne2 += j * weight(img.at<Vec3b>(i, j), colr_p, model_q);
        }
    }
    new_pos[0] = (int)(ne1 / den1);
    new_pos[1] = (int)(ne2 / den2);
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
Vec2i mean_shift_track_update(Mat img, double *mode_q, int prev_pos_row, int prev_pos_col, int row_window_size, int col_window_size)
{
    for (int epoch = 0; epoch < MEAN_SHIFT_MAX_ITER; ++epoch)
    {
        double *prev_colr_p = calc_p(img, prev_pos_row, prev_pos_col, row_window_size, col_window_size);
        Vec2i new_pos = new_loc_z(img, prev_pos_row, prev_pos_col, row_window_size, col_window_size, prev_colr_p, mode_q);
        double *new_colr_p = calc_p(img, new_pos[0], new_pos[1], row_window_size, col_window_size);

        double prev_bhat = bhatt_coefficient(prev_colr_p, mode_q), new_bhat = bhatt_coefficient(new_colr_p, mode_q);

        printf("\t prev pos=(%d, %d), new_pos=(%d,%d), prev coef=%lf, new coef=%lf\n", prev_pos_row, prev_pos_col, new_pos[0], new_pos[1], prev_bhat, new_bhat);
        printf("--prev colr_p=\n");
        for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
            printf("%lf\t", prev_colr_p[i]);
        printf("\n\n\t new color p=\n");
        for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
            printf("%lf\t", new_colr_p[i]);
        printf("\n\n\t mode q=\n");
        for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
            printf("%lf\t", mode_q[i]);
        printf("\n\n");
        delete[] prev_colr_p;
        delete[] new_colr_p;
        while (new_bhat < prev_bhat)
        {
            new_pos[0] = (int)round((new_pos[0] + prev_pos_row) / 2.0);
            new_pos[1] = (int)round((new_pos[1] + prev_pos_col) / 2.0);

            if (abs(new_pos[0] - prev_pos_row) <= 1 && abs(new_pos[1] - prev_pos_col) <= 1)
                break;
            new_colr_p = calc_p(img, new_pos[0], new_pos[1], row_window_size, col_window_size);
            new_bhat = bhatt_coefficient(new_colr_p, mode_q);
            delete[] new_colr_p;
        }

        if (abs(new_pos[0] - prev_pos_row) <= 1 && abs(new_pos[1] - prev_pos_col) <= 1)
            return new_pos;
        prev_pos_row = new_pos[0];
        prev_pos_col = new_pos[1];
    }
}

/**
 * Normalize color of image
 * @c (r',g',b')=(r,g,b)/(r+g+b)
 */
Mat img_norm_color(Mat img)
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

void generate_pngs(char *video_name = (char *)"mot.avi", int max_count = 40, int step = 1)
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
            sprintf(name, "imgs/f%02d.png", index++);
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
    const int num_frames = 1;
    char *frame_names[num_frames];
    for (int i = 0; i < num_frames; ++i)
    {
        frame_names[i] = new char[10];
        sprintf(frame_names[i], "./imgs/f%02d.png", i);
    }

    Mat frames[num_frames];
    for (int i = 0; i < num_frames; ++i)
    {
        frames[i] = imread(frame_names[i]);
        // frames[i] =img_norm_color(frames[i]);
    }

    int center_row = 300, center_col = 710, row_width = 100, col_width = 35;
    double *target_mode_q = calc_q(frames[0], center_row, center_col, row_width, col_width);
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
                newframe.at<Vec3b>(row, col)[0] = 255;
            }
        }
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