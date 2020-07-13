// Mean shift for video tracking
#include <opencv2/opencv.hpp>
#include <math.h>
#include <stdio.h>
#include <unistd.h>

#define PI 3.1415926
#define COLOR_BIN_WIDTH 15 // the with of RGB color bin
#define MEAN_SHIFT_MAX_ITER 50
#define EPSILON 0.0001
#define TOTAL_COLOR_BIN_NUM ((255 / COLOR_BIN_WIDTH) * 3) // assume RGB
#define CONVERT_RED_TO_BIN_INDEX(x) (x / COLOR_BIN_WIDTH + 2 * (255 / COLOR_BIN_WIDTH))
#define CONVERT_GREEN_TO_BIN_INDEX(x) (x / COLOR_BIN_WIDTH + (255 / COLOR_BIN_WIDTH))
#define CONVERT_BLUE_TO_BIN_INDEX(x) (x / COLOR_BIN_WIDTH)

using namespace cv;

double gaussian_kernel_profile(double x)
{
    return 1.0 / (2 * PI) * exp(-0.5 * x);
}

double gradient(double x)
{
    return 1.0 / (4 * PI) * exp(-0.5 * x);
}

/**
 * Calculate color p centered at y
 * The result contains result along row and along col
 */
double *calc_p(Mat img, int center_pos_row, int center_pos_col, int row_window_size, int col_window_size)
{
    // assume image is RGB 3 color
    double *res = new double[TOTAL_COLOR_BIN_NUM * 2];
    for (int i = 0; i < TOTAL_COLOR_BIN_NUM * 2; ++i)
        res[i] = 0;
    int row_low = center_pos_row - row_window_size;
    int row_high = center_pos_row + row_window_size;
    int col_low = center_pos_col - col_window_size;
    int col_high = center_pos_col + col_window_size;

    row_low = MAX(0, row_low);
    row_high = MIN(row_high, img.rows);
    col_low = MAX(0, col_low);
    col_high = MIN(col_high, img.cols);

    for (int row = row_low; row < row_high; ++row)
    {
        int r = (int)img.at<Vec3b>(row, center_pos_col)[2], g = (int)img.at<Vec3b>(row, center_pos_col)[1], b = (int)img.at<Vec3b>(row, center_pos_col)[0];
        res[CONVERT_RED_TO_BIN_INDEX(r)] += gaussian_kernel_profile(pow((row - center_pos_row) * 2.0 / row_window_size, 2));
        res[CONVERT_GREEN_TO_BIN_INDEX(g)] += gaussian_kernel_profile(pow((row - center_pos_row) * 2.0 / row_window_size, 2));
        res[CONVERT_BLUE_TO_BIN_INDEX(b)] += gaussian_kernel_profile(pow((row - center_pos_row) * 2.0 / row_window_size, 2));
    }

    for (int col = col_low; col < col_high; col++)
    {
        int r = (int)img.at<Vec3b>(center_pos_row, col)[2], g = (int)img.at<Vec3b>(center_pos_row, col)[1], b = (int)img.at<Vec3b>(center_pos_row, col)[0];
        res[TOTAL_COLOR_BIN_NUM + CONVERT_RED_TO_BIN_INDEX(r)] += gaussian_kernel_profile(pow((col - center_pos_col) * 2.0 / col_window_size, 2));
        res[TOTAL_COLOR_BIN_NUM + CONVERT_GREEN_TO_BIN_INDEX(g)] += gaussian_kernel_profile(pow((col - center_pos_col) * 2.0 / col_window_size, 2));
        res[TOTAL_COLOR_BIN_NUM + CONVERT_BLUE_TO_BIN_INDEX(b)] += gaussian_kernel_profile(pow((col - center_pos_col) * 2.0 / col_window_size, 2));
    }

    double norm_sum = 0;
    for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
    {
        norm_sum += res[i];
    }
    for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
    {
        res[i] /= norm_sum;
    }
    norm_sum = 0;
    for (int i = TOTAL_COLOR_BIN_NUM; i < 2 * TOTAL_COLOR_BIN_NUM; ++i)
    {
        norm_sum += res[i];
    }
    for (int i = TOTAL_COLOR_BIN_NUM; i < 2 * TOTAL_COLOR_BIN_NUM; ++i)
    {
        res[i] /= norm_sum;
    }
    return res;
}

double *calc_q(Mat img, int center_pos_row, int center_pos_col, int row_window_size, int col_window_size)
{
    return calc_p(img, center_pos_row, center_pos_col, row_window_size, col_window_size);
}

/**
 * Calculating the row and col bhat coefficient
 * @param row_coef  return value , the bhat coefficient of row pixles
 * @param col_coef return value, the bhat coefficient of col pixels
 */
void bhatt_coefficient(double *colr_p, double *model_q, double &row_coef, double &col_coef)
{
    row_coef = 0;
    col_coef = 0;
    for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
    {
        row_coef += sqrt(colr_p[i] * model_q[i] * 10000);
    }
    row_coef /= 100;
    for (int i = TOTAL_COLOR_BIN_NUM; i < 2 * TOTAL_COLOR_BIN_NUM; ++i)
    {
        col_coef += sqrt(colr_p[i] * model_q[i] * 10000);
    }
    col_coef /= 100;
}

/**
 * The weight for one pixel
 */
double weight(Vec3b pixel_colors, double *colr_p, double *model_q, int dim = 0)
{
    double w = 0;
    int r = (int)pixel_colors[2], g = (int)pixel_colors[1], b = (int)pixel_colors[0];
    if (model_q[CONVERT_RED_TO_BIN_INDEX(r) + dim * TOTAL_COLOR_BIN_NUM] < EPSILON)
    {
        if (colr_p[CONVERT_RED_TO_BIN_INDEX(r) + dim * TOTAL_COLOR_BIN_NUM] < EPSILON)
            w += 1;
    }
    else
    {
        if (colr_p[CONVERT_RED_TO_BIN_INDEX(r) + dim * TOTAL_COLOR_BIN_NUM] > EPSILON)
            w += sqrt(model_q[CONVERT_RED_TO_BIN_INDEX(r) + dim * TOTAL_COLOR_BIN_NUM] / colr_p[CONVERT_RED_TO_BIN_INDEX(r) + dim * TOTAL_COLOR_BIN_NUM]);
    }

    if (model_q[CONVERT_GREEN_TO_BIN_INDEX(g) + dim * TOTAL_COLOR_BIN_NUM] < EPSILON)
    {
        if (colr_p[CONVERT_GREEN_TO_BIN_INDEX(g) + dim * TOTAL_COLOR_BIN_NUM] < EPSILON)
            w += 1;
    }
    else
    {
        if (colr_p[CONVERT_GREEN_TO_BIN_INDEX(g) + dim * TOTAL_COLOR_BIN_NUM] > EPSILON)
            w += sqrt(model_q[CONVERT_GREEN_TO_BIN_INDEX(g) + dim * TOTAL_COLOR_BIN_NUM] / colr_p[CONVERT_GREEN_TO_BIN_INDEX(g) + dim * TOTAL_COLOR_BIN_NUM]);
    }

    if (model_q[CONVERT_BLUE_TO_BIN_INDEX(b) + dim * TOTAL_COLOR_BIN_NUM] < EPSILON)
    {
        if (colr_p[CONVERT_BLUE_TO_BIN_INDEX(b) + dim * TOTAL_COLOR_BIN_NUM] < EPSILON)
            w += 1;
    }
    else
    {
        if (colr_p[CONVERT_BLUE_TO_BIN_INDEX(b) + dim * TOTAL_COLOR_BIN_NUM] > EPSILON)
            w += sqrt(model_q[CONVERT_BLUE_TO_BIN_INDEX(b) + dim * TOTAL_COLOR_BIN_NUM] / colr_p[CONVERT_BLUE_TO_BIN_INDEX(b) + dim * TOTAL_COLOR_BIN_NUM]);
    }
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

    double nu_row = 0, nu_col = 0, de_row = 0, de_col = 0;
    printf("\t ---- center row = %d\n", center_pos_row);
    printf("\t color p = \n");
    for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
    {
        printf("%lf\t", colr_p[i]);
    }
    printf("\n\nmodel q = \n");
    for (int i = 0; i < TOTAL_COLOR_BIN_NUM; i++)
    {
        printf("%lf\t", model_q[i]);
    }
    printf("\n\n");
    for (int row = row_low; row < row_high; ++row)
    {
        printf("\t ---- weight of row %d = %lf, gradient =%lf, model q r=%lf, g=%lf,b=%lf, colr p r=%lf, g=%lf, b=%lf\n",
               row, weight(img.at<Vec3b>(row, center_pos_col), colr_p, model_q), gradient(pow((row - center_pos_row) * 1.0 / row_window_size, 2)),
               model_q[CONVERT_RED_TO_BIN_INDEX(img.at<Vec3b>(row, center_pos_col)[2])],
               model_q[CONVERT_GREEN_TO_BIN_INDEX(img.at<Vec3b>(row, center_pos_col)[1])],
               model_q[CONVERT_BLUE_TO_BIN_INDEX(img.at<Vec3b>(row, center_pos_col)[0])],
               colr_p[CONVERT_RED_TO_BIN_INDEX(img.at<Vec3b>(row, center_pos_col)[2])],
               colr_p[CONVERT_GREEN_TO_BIN_INDEX(img.at<Vec3b>(row, center_pos_col)[1])],
               colr_p[CONVERT_BLUE_TO_BIN_INDEX(img.at<Vec3b>(row, center_pos_col)[0])]);
        nu_row += weight(img.at<Vec3b>(row, center_pos_col), colr_p, model_q) * gradient(pow((row - center_pos_row) * 2.0 / row_window_size, 2)) * row;
        de_row += weight(img.at<Vec3b>(row, center_pos_col), colr_p, model_q) * gradient(pow((row - center_pos_row) * 2.0 / row_window_size, 2));
    }
    for (int col = col_low; col < col_high; ++col)
    {
        nu_col += weight(img.at<Vec3b>(center_pos_row, col), colr_p, model_q, 1) * gradient(pow((col - center_pos_col) * 2.0 / col_window_size, 2)) * col;
        de_col += weight(img.at<Vec3b>(center_pos_row, col), colr_p, model_q, 1) * gradient(pow((col - center_pos_col) * 2.0 / col_window_size, 2));
    }
    return Vec2i((int)round(nu_row / de_row), (int)round(nu_col / de_col));
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
    Vec2i ans_pos(0, 0);

    // update row position
    for (int epoch = 0; epoch < MEAN_SHIFT_MAX_ITER; ++epoch)
    {
        double *colr_p = calc_p(img, prev_pos_row, prev_pos_col, row_window_size, col_window_size);
        double prev_row_coef = 0, prev_col_coef = 0;
        bhatt_coefficient(colr_p, mode_q, prev_row_coef, prev_col_coef);
        Vec2i loc_z = new_loc_z(img, prev_pos_row, prev_pos_col, row_window_size, col_window_size, colr_p, mode_q);

        double *colr_p_z = calc_p(img, loc_z[0], loc_z[1], row_window_size, col_window_size);
        double z_row_coef = 0, z_col_coef = 0;
        bhatt_coefficient(colr_p_z, mode_q, z_row_coef, z_col_coef);
        printf("\t new row pos = %d, prev row pos=%d, coef new=%lf, coef prev=%lf\n", loc_z[0], prev_pos_row, z_row_coef, prev_row_coef);
        printf("\t prev color p=\n");
        for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
            printf("%lf\t", colr_p[i]);
        printf("\n\n new colr z p=\n");
        for (int i = 0; i < TOTAL_COLOR_BIN_NUM; ++i)
            printf("%lf\t", colr_p_z[i]);
        printf("\n");
        delete[] colr_p;
        // update row and col position
        while (z_row_coef < prev_row_coef)
        {
            loc_z[0] = (loc_z[0] + prev_pos_row) / 2;
            if (abs(loc_z[0] - prev_pos_row) <= 1)
                break;
            delete[] colr_p_z;
            colr_p_z = calc_p(img, loc_z[0], loc_z[1], row_window_size, col_window_size);
            bhatt_coefficient(colr_p_z, mode_q, z_row_coef, z_col_coef);
            printf("\t  while update, loc z new=(%d,%d), coef new=%lf, prev_coef=%lf\n", loc_z[0], loc_z[1], z_row_coef, prev_row_coef);
        }
        delete[] colr_p_z;
        if (abs(loc_z[0] - prev_pos_row) <= 1)
        {
            ans_pos[0] = prev_pos_row;
            loc_z[0] = prev_pos_row;
            break;
        }
        else
        {
            prev_pos_row = loc_z[0];
        }
    }

    // update col position
    for (int epoch = 0; epoch < MEAN_SHIFT_MAX_ITER; ++epoch)
    {
        double *colr_p = calc_p(img, prev_pos_row, prev_pos_col, row_window_size, col_window_size);
        double tmp = 0, prev_col_coef = 0;
        bhatt_coefficient(colr_p, mode_q, tmp, prev_col_coef);
        Vec2i loc_z = new_loc_z(img, prev_pos_row, prev_pos_col, row_window_size, col_window_size, colr_p, mode_q);

        double *colr_p_z = calc_p(img, loc_z[0], loc_z[1], row_window_size, col_window_size);
        double z_col_coef = 0;
        bhatt_coefficient(colr_p_z, mode_q, tmp, z_col_coef);
        printf("\t new col pos=%d, prev col pos=%d, new col coef=%lf. prev col coef=%lf\n", loc_z[1], prev_pos_col, z_col_coef, prev_col_coef);
        delete[] colr_p;

        while (z_col_coef < prev_col_coef)
        {
            loc_z[1] = (loc_z[1] + prev_pos_col) / 2;
            if (abs(loc_z[1] - prev_pos_col) <= 1)
                break;
            delete[] colr_p_z;
            colr_p_z = calc_p(img, loc_z[0], loc_z[1], row_window_size, col_window_size);
            bhatt_coefficient(colr_p_z, mode_q, tmp, z_col_coef);
        }
        delete[] colr_p_z;
        if (abs(loc_z[1] - prev_pos_col) <= 1)
        {
            ans_pos[1] = prev_pos_col;
            loc_z[1] = prev_pos_col;
            break;
        }
        else
        {
            prev_pos_col = loc_z[1];
        }
    }
    return ans_pos;
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

void generate_pngs(char *video_name = (char *)"mot.avi", int step = 5)
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
    }
}

int main(int argc, char const *argv[])
{
    const int num_frames = 20;
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

    int center_row = 260, center_col = 250, row_width = 50, col_width = 50;
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
                // newframe.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
                newframe.at<Vec3b>(row, col)[0] = 255;
            }
        }
        if (i > 18)
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